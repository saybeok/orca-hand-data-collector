"""Replay a hand capture through the retargeter and measure pinch fidelity.

For every frame the robot's thumb-to-fingertip distances (forward kinematics
of the retargeted joints) are compared with the operator's. Variants override
retarget-config values so settings can be compared on identical input:

    python scripts/eval_retarget_pinch.py ../data/hand_captures/pinch_01.npz \\
        --variant off:w_pinch=0 \\
        --variant current \\
        --variant strong:w_pinch=20,pinch_alpha_max=1.0 \\
        --variant wide:pinch_d1=3,pinch_d2=6

``pinch_d1`` / ``pinch_d2`` set the thresholds for all four fingers.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from orca_teleop.retargeting import utils as retargeter_utils
from orca_teleop.retargeting.adaptive_analytical import AdaptiveAnalyticalRetargeter
from orca_teleop.retargeting.retargeter import Retargeter, TargetPose

FINGERS = ("index", "middle", "ring", "pinky")
TIPS = np.array([4, 8, 12, 16, 20])
PINCH_CM = 3.0
OPEN_CM = 7.0


def _parse_variant(spec: str) -> tuple[str, dict[str, float]]:
    name, _, body = spec.partition(":")
    overrides = {}
    for item in filter(None, body.split(",")):
        key, _, value = item.partition("=")
        overrides[key.strip()] = float(value)
    return name, overrides


def _build(args, overrides: dict[str, float]) -> AdaptiveAnalyticalRetargeter:
    r = Retargeter.from_paths(
        model_path=args.hand_config,
        urdf_path=args.urdf,
        config_path=args.retarget_config,
        backend=getattr(args, "retargeter", "adaptive_analytical"),
    )
    overrides = dict(overrides)
    if getattr(args, "retargeter", "adaptive_analytical") == "dexpilot":
        if overrides:
            raise ValueError("DexPilot comparisons use its YAML profile; no adaptive variant overrides")
        return r
    d1, d2 = overrides.pop("pinch_d1", None), overrides.pop("pinch_d2", None)
    r._retarget_cfg.update(overrides)
    r._build_retarget_params()
    if d1 is not None:
        if not isinstance(r, AdaptiveAnalyticalRetargeter):
            raise ValueError("pinch_d1/d2 overrides are only supported by adaptive_analytical")
        r._pinch_d1[:] = d1
    if d2 is not None:
        if not isinstance(r, AdaptiveAnalyticalRetargeter):
            raise ValueError("pinch_d1/d2 overrides are only supported by adaptive_analytical")
        r._pinch_d2[:] = d2
    return r


def _robot_gaps_cm(r: AdaptiveAnalyticalRetargeter) -> np.ndarray:
    """Visible fingertip gaps (URDF tip frames, not the extra pad offsets)."""
    if hasattr(r, "_seq"):
        q = np.zeros(r._robot.dof)
        q[r._model_q_indices_np] = r._last_qpos_phys
        r._robot.compute_forward_kinematics(q)
        tips = np.array([
            r._robot.get_link_pose(r._robot.get_link_index(f"right_{f}_fingertip"))[:3, 3].copy()
            for f in ("thumb", *FINGERS)
        ]) * 100.0
        return np.linalg.norm(tips[1:] - tips[0], axis=1)
    offsets = getattr(r, "_contact_offsets", None)
    pos, _ = r._robot.compute_positions_and_jacobians(
        r._full_model_qpos(r._last_qpos_phys), r._computed_frame_indices, offsets
    )
    tips = pos[r._tip_indices] * 100.0
    return np.linalg.norm(tips[1:] - tips[0], axis=1)


def _retarget_frame_gaps_cm(r: AdaptiveAnalyticalRetargeter, kp: np.ndarray) -> np.ndarray:
    """Operator gaps as the pinch weighting sees them (normalized, auto-scaled)."""
    normalized = retargeter_utils.get_normalized_local_manohand_joint_pos(kp, "mediapipe")
    if hasattr(r, "_seq"):
        # Upstream projection tests the unscaled human distances, in metres.
        pts = normalized @ r._urdf_rotation.T
        return np.linalg.norm(pts[TIPS[1:]] - pts[TIPS[0]], axis=1) * 100.0
    if r._rotation_matrix is not None:
        normalized = normalized @ r._rotation_matrix.T
    pts = r._target_keypoints_in_urdf_frame(normalized)
    return np.linalg.norm(pts[TIPS[1:]] - pts[TIPS[0]], axis=1) * 100.0


def replay(r: AdaptiveAnalyticalRetargeter, capture) -> dict[str, np.ndarray]:
    human, seen, robot, commands, timestamps, frame_indices = [], [], [], [], [], []
    # Replay acquisition time, not how fast this CPU runs the optimizer.
    times = capture.get("timestamp_ns")
    if times is None:
        times = 1_000_000_000 + np.rint(np.arange(len(capture["keypoints"])) * 1e9 / 30).astype(np.int64)
    for frame_index, (kp, wrist, recal, timestamp) in enumerate(zip(
        capture["keypoints"], capture["wrist_angle_degrees"], capture["recalibrate"], times,
        strict=True,
    )):
        if recal:
            r.reset()
            continue
        try:
            out = r.retarget(TargetPose(
                joint_positions=kp, source="mediapipe", wrist_angle_degrees=float(wrist),
                timestamp_ns=int(timestamp),
            ))
        except (AssertionError, ValueError):
            continue
        if out is None:
            continue
        human.append(np.linalg.norm(kp[TIPS[1:]] - kp[TIPS[0]], axis=1) * 100.0)
        seen.append(_retarget_frame_gaps_cm(r, kp))
        robot.append(_robot_gaps_cm(r))
        commands.append(r._last_qpos_phys.copy())
        timestamps.append(timestamp)
        frame_indices.append(frame_index)
    return {
        "human": np.array(human), "seen": np.array(seen), "robot": np.array(robot),
        "q_rad": np.array(commands), "timestamp_ns": np.array(timestamps, dtype=np.int64),
        "frame_index": np.array(frame_indices, dtype=np.int64),
        "joint_ids": np.array(r._finger_joint_ids),
    }


def _summary(name: str, res: dict[str, np.ndarray]) -> None:
    print(f"\n== {name}  ({len(res['robot'])} frames)")
    if not len(res["robot"]):
        print("No retargeted frames; capture may be shorter than calibration.")
        return
    if "q_rad" in res and len(res["q_rad"]) > 1:
        dt = np.diff(res["timestamp_ns"]).astype(float) * 1e-9
        step = np.rad2deg(np.max(np.abs(np.diff(res["q_rad"], axis=0)), axis=1))
        valid = dt > 0
        if valid.any():
            print(f"max command step {step.max():.3f} deg; max velocity {(step[valid] / dt[valid]).max():.3f} deg/s")
    print(f"{'finger':7s} {'pinch fr':>8s} {'robot gap med':>13s} {'p90':>6s} {'<1cm':>6s} {'open gap med':>12s}")
    for i, finger in enumerate(FINGERS):
        pinch = res["human"][:, i] < PINCH_CM
        opened = res["human"][:, i] > OPEN_CM
        rg = res["robot"][:, i]
        if pinch.any():
            row = f"{pinch.sum():8d} {np.median(rg[pinch]):13.2f} {np.percentile(rg[pinch], 90):6.2f} {np.mean(rg[pinch] < 1.0):6.0%}"
        else:
            row = f"{0:8d} {'-':>13s} {'-':>6s} {'-':>6s}"
        open_med = f"{np.median(rg[opened]):12.2f}" if opened.any() else f"{'-':>12s}"
        print(f"{finger:7s} {row} {open_med}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--variant", action="append", default=[], help="NAME[:key=val,...]")
    parser.add_argument("--hand-config", default=os.environ.get("ORCA_MODEL_PATH"))
    parser.add_argument("--urdf", default=os.environ.get("ORCA_URDF_PATH"))
    parser.add_argument("--retarget-config", default=os.environ.get("ORCA_RETARGET_CONFIG"))
    parser.add_argument("--retargeter", default="adaptive_analytical", choices=["adaptive_analytical", "key_objectives", "dexpilot"])
    parser.add_argument("--limit-frames", type=int, help="Replay only the first N input frames")
    parser.add_argument("--output-dir", type=Path, help="Save command and gap traces for each variant")
    args = parser.parse_args()
    if args.limit_frames is not None and args.limit_frames <= 0:
        parser.error("--limit-frames must be positive")

    with np.load(args.capture) as loaded:
        n_frames = len(loaded["keypoints"])
        capture = {
            key: value[:args.limit_frames] if value.ndim and len(value) == n_frames else value
            for key, value in loaded.items()
        }
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = [_parse_variant(v) for v in (args.variant or ["current"])]

    first = None
    for name, overrides in variants:
        retargeter = _build(args, overrides)
        res = replay(retargeter, capture)
        first = first or res
        _summary(name, res)
        if args.output_dir:
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name) or "variant"
            np.savez_compressed(
                args.output_dir / f"{safe_name}.npz", **res,
                backend=np.array(args.retargeter),
                source_capture=np.array(str(args.capture.resolve())),
                resolved_config_json=np.array(json.dumps(retargeter.config, sort_keys=True)),
            )

    if not len(first["human"]):
        return 1
    h, s = first["human"][:, 0], first["seen"][:, 0]
    print("\noperator thumb-index gap (cm): raw Quest vs as seen by the pinch weighting")
    for q in (1, 5, 10, 25, 50):
        print(f"  p{q:<3d} raw {np.percentile(h, q):5.2f}   seen {np.percentile(s, q):5.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
