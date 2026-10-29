#!/usr/bin/env python3
"""Fit AM-ARM200 MuJoCo joint/actuator parameters to recorded step responses.

  arm_sysid_fit.py data/sysid/arm_XXXX.npz            # fit + report
  arm_sysid_fit.py data/sysid/arm_XXXX.npz --apply    # also write into the MJCF
  arm_sysid_fit.py --synthetic                        # self-test on sim-made data

Per joint it fits kp, damping, armature, frictionloss, forcerange and a command
delay by replaying the recorded commands through the combined model. It also
regresses steady-state servo current against the model's gravity torque: a good
linear fit (similar slope on shoulder_lift and elbow_flex) means the link
masses/COMs are consistent with the real arm.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from orca_teleop.am200.dynamics import dynamics_path_for, write_arm_delays
from orca_teleop.am200.sysid import (
    ARM_JOINTS,
    ArmReplay,
    JointParams,
    apply_params_to_mjcf,
    build_schedule,
    default_model_path,
)

# log-space bounds for kp, damping, armature, frictionloss, forcerange
_LO = np.log([1.0, 1e-3, 1e-4, 1e-4, 0.3])
_HI = np.log([500.0, 10.0, 0.5, 3.0, 30.0])
_FD_STEP = 0.02
_DELAY_GRID = np.arange(0.0, 0.0801, 0.002)
_TICK = 2 * np.pi / 4096
_MIN_TAU_RANGE = 0.15


def _to_x(p: JointParams) -> np.ndarray:
    x = np.log([
        p.kp,
        max(p.damping, 1e-3),
        max(p.armature, 1e-4),
        max(p.frictionloss, 1e-4),
        p.forcerange,
    ])
    return np.clip(x, _LO + 1e-9, _HI - 1e-9)


def _from_x(x: np.ndarray, delay_s: float) -> JointParams:
    kp, damping, armature, frictionloss, forcerange = np.exp(x)
    return JointParams(
        kp=float(kp),
        damping=float(damping),
        armature=float(armature),
        frictionloss=float(frictionloss),
        forcerange=float(forcerange),
        delay_s=float(delay_s),
    )


def _onsets(t: np.ndarray, cmd: np.ndarray, q: np.ndarray, j: int) -> np.ndarray:
    """Time from each command change on joint j until q_j moves by 3 encoder ticks."""
    out = []
    for c in np.flatnonzero(np.diff(cmd[:, j]) != 0) + 1:
        moved = np.flatnonzero(np.abs(q[c:, j] - q[c - 1, j]) > 3 * _TICK)
        out.append(t[c + moved[0]] - t[c] if len(moved) else np.nan)
    return np.asarray(out)


def _segment(data: dict, j: int) -> dict | None:
    rows = data["segment"] == j
    if rows.sum() < 20:
        return None
    t = data["t"][rows]
    return {
        "t": t - t[0],
        "cmd": data["cmd"][rows],
        "q": data["q"][rows],
        "cur": data["current_mA"][rows],
    }


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((a - b) ** 2)))


def _with_delay(p: JointParams, delay_s: float) -> JointParams:
    return JointParams(**{**p.as_dict(), "delay_s": float(delay_s)})


def fit_all(
    rep: ArmReplay,
    segs: dict[int, dict],
    fit_idx: list[int],
    params: list[JointParams],
    home: np.ndarray,
) -> tuple[list[JointParams], dict[int, float], dict[int, float]]:
    """Jointly fit all selected joints against every segment.

    Joints are gravity-coupled, so the sag of the *held* joints in another
    joint's segment carries information about their own stiffness too.
    Returns ``(params, rmse_before_by_joint, rmse_after_by_joint)``.
    """
    params = list(params)
    nx = 5

    def seg_resid(j: int, sets: list[list[JointParams]]) -> np.ndarray:
        s = segs[j]
        q, _ = rep.rollout(s["t"], s["cmd"], s["q"][0], s["t"], sets, home)
        return np.nan_to_num(q - s["q"][None], nan=1.0)  # (B, T, 3)

    def sets_from_x(xs: list[np.ndarray], delays: dict[int, float]) -> list[list[JointParams]]:
        out = []
        for x in xs:
            ps = list(params)
            for k, j in enumerate(fit_idx):
                ps[j] = _from_x(x[k * nx:(k + 1) * nx], delays[j])
            out.append(ps)
        return out

    def resid_many(sets: list[list[JointParams]]) -> np.ndarray:
        return np.concatenate(
            [seg_resid(j, sets).reshape(len(sets), -1) for j in segs], axis=1
        )

    def own_rmse(ps: list[JointParams]) -> dict[int, float]:
        return {j: _rmse(seg_resid(j, [ps])[0, :, j], 0.0) for j in segs}

    def estimate_delay(j: int) -> float:
        s = segs[j]
        ps = list(params)
        ps[j] = _with_delay(ps[j], 0.0)
        q_sim = seg_resid(j, [ps])[0] + s["q"]
        lag = _onsets(s["t"], s["cmd"], s["q"], j) - _onsets(s["t"], s["cmd"], q_sim, j)
        if not np.isfinite(lag).any():
            return 0.0
        return float(np.clip(np.nanmedian(lag), 0.0, _DELAY_GRID[-1]))

    def refine_delay(j: int) -> float:
        sets = []
        for dl in _DELAY_GRID:
            ps = list(params)
            ps[j] = _with_delay(ps[j], dl)
            sets.append(ps)
        cost = np.sum(seg_resid(j, sets)[:, :, j] ** 2, axis=1)
        return float(_DELAY_GRID[int(np.argmin(cost))])

    lo = np.tile(_LO, len(fit_idx))
    hi = np.tile(_HI, len(fit_idx))

    def fit_fixed_delay(x0: np.ndarray, delays: dict[int, float]) -> np.ndarray:
        cache: dict[bytes, np.ndarray] = {}

        def fun(x: np.ndarray) -> np.ndarray:
            key = x.tobytes()
            if key not in cache:
                cache[key] = resid_many(sets_from_x([x], delays))[0]
            return cache[key]

        def jac(x: np.ndarray) -> np.ndarray:
            steps = np.where(x + _FD_STEP <= hi, _FD_STEP, -_FD_STEP)
            xs = [x] + [x + np.eye(len(x))[k] * steps[k] for k in range(len(x))]
            r = resid_many(sets_from_x(xs, delays))
            return ((r[1:] - r[0]) / steps[:, None]).T

        sol = least_squares(fun, x0, jac=jac, bounds=(lo, hi), max_nfev=80, x_scale="jac")
        return sol.x

    before = own_rmse(params)
    x = np.concatenate([_to_x(params[j]) for j in fit_idx])
    delays = {j: estimate_delay(j) for j in fit_idx}
    for _ in range(6):
        x = fit_fixed_delay(x, delays)
        params = sets_from_x([x], delays)[0]
        new = {j: refine_delay(j) for j in fit_idx}
        if new == delays:
            break
        delays = new
        params = sets_from_x([x], delays)[0]
    return params, before, own_rmse(params)


def gravity_current_check(rep: ArmReplay, data: dict, home: np.ndarray) -> dict:
    """Regress steady-state current on model gravity torque, per joint.

    Model: current = k * tau_gravity + b + c * last_move_direction, where the
    direction term absorbs static-friction hysteresis of the geared servo.
    """
    t, cmd, q, cur = data["t"], data["cmd"], data["q"], data["current_mA"]
    change = np.flatnonzero(np.any(np.diff(cmd, axis=0) != 0, axis=1)) + 1
    bounds = np.concatenate([[0], change, [len(t)]])
    taus, curs, dirs = [], [], []
    last_dir = np.zeros(3)
    for a, b in zip(bounds[:-1], bounds[1:]):
        if a > 0:
            step = np.sign(cmd[a] - cmd[a - 1])
            last_dir = np.where(step != 0, step, last_dir)
        if b - a < 10:
            continue
        tail = slice(a + int(0.6 * (b - a)), b)
        if np.ptp(q[tail], axis=0).max() > 4 * _TICK:
            continue
        tau_g, ncon = rep.gravity_torque(q[tail].mean(axis=0), home)
        if ncon:
            continue
        taus.append(tau_g)
        curs.append(cur[tail].mean(axis=0))
        dirs.append(last_dir.copy())
    taus = np.asarray(taus).reshape(-1, 3)
    curs = np.asarray(curs).reshape(-1, 3)
    dirs = np.asarray(dirs).reshape(-1, 3)
    out = {}
    for i, name in enumerate(ARM_JOINTS):
        x, y, s = taus[:, i], curs[:, i], dirs[:, i]
        if len(x) < 4 or np.ptp(x) < _MIN_TAU_RANGE:
            out[name] = {
                "note": f"gravity torque range {np.ptp(x) if len(x) else 0:.2f} Nm over {len(x)} "
                f"settled holds; need >= {_MIN_TAU_RANGE} Nm (record larger --amp)"
            }
            continue
        A = np.stack([x, np.ones_like(x), s], axis=1)
        (k, b, c), *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - np.sum((y - A @ [k, b, c]) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-9)
        out[name] = {
            "mA_per_Nm": float(k),
            "offset_mA": float(b),
            "friction_mA": float(c),
            "r2": float(r2),
            "tau_range_Nm": [float(x.min()), float(x.max())],
            "n": int(len(x)),
        }
    return out


# -------------------------------------------------------------- synthetic


def make_synthetic(rep: ArmReplay, home: np.ndarray, seed: int = 0) -> tuple[dict, list[JointParams]]:
    rng = np.random.default_rng(seed)
    truth = [
        JointParams(kp=35.0, damping=0.8, armature=0.03, frictionloss=0.05, forcerange=4.0, delay_s=0.02),
        JointParams(kp=55.0, damping=1.2, armature=0.04, frictionloss=0.08, forcerange=6.0, delay_s=0.025),
        JointParams(kp=45.0, damping=0.6, armature=0.02, frictionloss=0.04, forcerange=5.0, delay_s=0.015),
    ]
    k_true = np.array([300.0, 300.0, 300.0])
    parts = {k: [] for k in ("t", "cmd", "q", "current_mA", "segment")}
    t_off = 0.0
    for j in range(3):
        ts, tg = build_schedule(j, [0.15, 0.3, 0.45, 0.6], "both" if j == 0 else "pos", 1.5)
        dur = ts[-1] + 1.5
        t_eval = np.cumsum(rng.uniform(0.003, 0.005, int(dur / 0.004)))
        t_eval = t_eval[t_eval < dur]
        idx = np.searchsorted(ts, t_eval, side="right") - 1
        cmd = tg[idx]
        q, tau = rep.simulate(t_eval, cmd, np.zeros(3), t_eval, truth, home)
        q = np.round(q / (2 * np.pi / 4096)) * (2 * np.pi / 4096)
        parts["t"].append(t_eval + t_off)
        parts["cmd"].append(cmd)
        parts["q"].append(q)
        parts["current_mA"].append(tau * k_true + rng.normal(0, 6.5, tau.shape))
        parts["segment"].append(np.full(len(t_eval), j))
        t_off += dur + 0.5
    return {k: np.concatenate(v) for k, v in parts.items()}, truth


# ------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", nargs="?")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--model", default=str(default_model_path()))
    parser.add_argument(
        "--home-qpos", type=float, nargs=3, default=[0.0, 0.0, 0.0],
        help="model qpos of the pose the arm was in at connect (default: model zero)",
    )
    parser.add_argument("--joints", nargs="+", default=list(ARM_JOINTS), choices=ARM_JOINTS)
    parser.add_argument("--out", default=None, help="result json (default next to recording)")
    parser.add_argument("--apply", action="store_true", help="write fitted values into --model")
    args = parser.parse_args()
    if not args.synthetic and not args.recording:
        parser.error("give a recording .npz or --synthetic")

    rep = ArmReplay(args.model)
    home = np.asarray(args.home_qpos)
    truth = None
    if args.synthetic:
        data, truth = make_synthetic(rep, home)
        print("synthetic data generated from known params")
    else:
        data = dict(np.load(args.recording))

    params = [rep.get_params(i) for i in range(3)]
    start = [JointParams(**p.as_dict()) for p in params]
    report: dict = {"model": str(args.model), "home_qpos": home.tolist(), "joints": {}}

    segs = {j: s for j in range(3) if (s := _segment(data, j)) is not None}
    fit_idx = [ARM_JOINTS.index(n) for n in args.joints if ARM_JOINTS.index(n) in segs]
    if not fit_idx:
        raise SystemExit("recording has no segments for the selected joints")
    t0 = time.time()
    params, before, after = fit_all(rep, segs, fit_idx, params, home)
    print(f"joint fit over {len(segs)} segments took {time.time() - t0:.0f}s")

    def fmt(p: JointParams) -> str:
        return (
            f"kp={p.kp:.1f} damping={p.damping:.3f} armature={p.armature:.4f} "
            f"frictionloss={p.frictionloss:.3f} forcerange={p.forcerange:.2f} delay={p.delay_s * 1e3:.0f}ms"
        )

    for j in fit_idx:
        name = ARM_JOINTS[j]
        print(
            f"[{name}] rmse {np.degrees(before[j]):.2f}° -> {np.degrees(after[j]):.2f}°\n"
            f"    fit   {fmt(params[j])}"
        )
        entry = {
            "before": start[j].as_dict(),
            "fitted": params[j].as_dict(),
            "rmse_deg_before": float(np.degrees(before[j])),
            "rmse_deg_after": float(np.degrees(after[j])),
        }
        if truth is not None:
            entry["truth"] = truth[j].as_dict()
            print(f"    truth {fmt(truth[j])}")
        report["joints"][name] = entry
    if truth is not None:
        floor = {
            ARM_JOINTS[j]: float(np.degrees(_rmse(
                rep.simulate(s["t"], s["cmd"], s["q"][0], s["t"], truth, home)[0][:, j], s["q"][:, j]
            )))
            for j, s in segs.items()
        }
        print("    rmse at truth (noise floor): " + ", ".join(f"{k} {v:.3f}°" for k, v in floor.items()))

    grav = gravity_current_check(rep, data, home)
    report["gravity_current"] = grav
    print("\ngravity-torque vs steady current (mass/COM consistency):")
    for name, g in grav.items():
        if "note" in g:
            print(f"  {name:14s} {g['note']}")
        else:
            print(
                f"  {name:14s} {g['mA_per_Nm']:.0f} mA/Nm  friction ±{abs(g['friction_mA']):.0f} mA  R²={g['r2']:.3f}  "
                f"τ range {g['tau_range_Nm'][0]:+.2f}..{g['tau_range_Nm'][1]:+.2f} Nm"
            )

    out = Path(args.out) if args.out else (
        Path(args.recording).with_suffix(".fit.json") if args.recording else None
    )
    if out:
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nsaved {out}")

    if args.apply:
        if args.synthetic:
            print("--apply ignored for --synthetic")
        else:
            model = Path(args.model)
            shutil.copy2(model, model.with_suffix(".xml.bak"))
            apply_params_to_mjcf(
                model, {n: params[ARM_JOINTS.index(n)] for n in report["joints"]}
            )
            dyn_path = dynamics_path_for(model)
            write_arm_delays(
                dyn_path, {n: params[ARM_JOINTS.index(n)].delay_s for n in report["joints"]}
            )
            print(f"applied to {model} (backup {model.with_suffix('.xml.bak').name})")
            print(f"command delays written to {dyn_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
