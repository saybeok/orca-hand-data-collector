#!/usr/bin/env python3
"""Record AM-ARM200 step responses for system identification.

Place the arm in the model zero pose (same start pose as teleop) before running:
connect-time angles become model qpos = 0. Each selected joint walks a
staircase of offsets up and back down to 0 while the others hold. Position,
velocity and current are logged as fast as the bus allows.

Keep a hand under the forearm: at the end (and on Ctrl+C) torque is disabled.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from orca_teleop.am200.arm_client import AmArmFeetech, default_arm_config_path
from orca_teleop.am200.sysid import ARM_JOINTS, build_schedule


def _parse_directions(items: list[str]) -> dict[str, str]:
    out = {j: "both" for j in ARM_JOINTS}
    for item in items:
        name, _, val = item.partition("=")
        if name not in out or val not in ("both", "pos", "neg"):
            raise SystemExit(f"bad --direction {item!r}; use e.g. shoulder_lift=pos")
        out[name] = val
    return out


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(default_arm_config_path()))
    parser.add_argument("--joints", nargs="+", default=list(ARM_JOINTS), choices=ARM_JOINTS)
    parser.add_argument(
        "--amp", type=float, nargs="+", default=[0.1, 0.2, 0.3],
        help="staircase levels (rad); larger levels make the gravity/mass check informative",
    )
    parser.add_argument(
        "--direction",
        nargs="*",
        default=[],
        help="per-joint side, e.g. shoulder_lift=pos elbow_flex=neg (default both)",
    )
    parser.add_argument("--hold-s", type=float, default=1.5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    directions = _parse_directions(args.direction)
    out = Path(args.out) if args.out else (
        root / "data" / "sysid" / f"arm_{datetime.now():%Y%m%d_%H%M%S}.npz"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    arm = AmArmFeetech(args.config)
    signs = arm._signs.copy()

    # Per-joint schedule, concatenated with a settle gap between joints.
    t_sched, tgt_sched, seg_sched = [], [], []
    t_off = 0.0
    for name in args.joints:
        j = ARM_JOINTS.index(name)
        ts, tg = build_schedule(j, args.amp, directions[name], args.hold_s)
        t_sched.append(ts + t_off)
        tgt_sched.append(tg)
        seg_sched.append(np.full(len(ts), j))
        t_off += ts[-1] + args.hold_s + 0.5
    t_sched = np.concatenate(t_sched)
    tgt_sched = np.concatenate(tgt_sched)
    seg_sched = np.concatenate(seg_sched)
    t_end = t_off
    print(f"joints={args.joints} amp={args.amp} dirs={directions} duration≈{t_end:.1f}s")

    arm.connect()
    home = arm._motor_home.copy()

    def _shutdown(*_a: object) -> None:
        print("\ninterrupt — returning to home, then torque OFF", flush=True)
        try:
            arm.write_model_rad(np.zeros(3))
            time.sleep(1.0)
        finally:
            arm.disconnect(disable_torque=True)
        sys.exit(130)

    signal.signal(signal.SIGINT, _shutdown)

    arm.write_model_rad(np.zeros(3))
    time.sleep(1.0)

    log_t, log_cmd, log_q, log_dq, log_cur, log_seg = [], [], [], [], [], []
    client = arm._client
    last_k = -1
    t0 = time.monotonic()
    while True:
        t = time.monotonic() - t0
        if t >= t_end:
            break
        k = int(np.searchsorted(t_sched, t, side="right") - 1)
        if k != last_k:
            arm.write_model_rad(tgt_sched[k])
            last_k = k
        r = client.read_position_velocity_current()
        log_t.append(time.monotonic() - t0)
        log_cmd.append(tgt_sched[k])
        log_q.append((r.position.astype(np.float64) - home) / signs)
        log_dq.append(r.velocity.astype(np.float64) / signs)
        log_cur.append(r.current.astype(np.float64) / signs)
        log_seg.append(seg_sched[k])

    arm.write_model_rad(np.zeros(3))
    time.sleep(1.0)
    arm.disconnect(disable_torque=True)

    t_arr = np.asarray(log_t)
    np.savez(
        out,
        t=t_arr,
        cmd=np.asarray(log_cmd),
        q=np.asarray(log_q),
        dq=np.asarray(log_dq),
        current_mA=np.asarray(log_cur),
        segment=np.asarray(log_seg),
        joint_names=np.asarray(ARM_JOINTS),
        motor_home=home,
        speed=arm._speed,
        max_torque=arm._max_torque,
    )
    rate = len(t_arr) / max(t_arr[-1], 1e-6)
    print(f"saved {out}  samples={len(t_arr)}  rate≈{rate:.0f} Hz")
    print("DONE — torque OFF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
