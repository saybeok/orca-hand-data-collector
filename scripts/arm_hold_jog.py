#!/usr/bin/env python3
"""Safe AM-ARM200 bring-up: hold all joints, tiny jog each axis, hold, exit.

Does not move the ORCA hand. Keep clear of the desk; Ctrl+C disables torque.
"""
from __future__ import annotations

import argparse
import signal
import sys
import time

import numpy as np

from orca_teleop.am200.arm_client import (
    AmArmFeetech,
    default_arm_config_path,
    load_arm_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(default_arm_config_path()))
    parser.add_argument("--delta", type=float, default=0.08, help="jog amplitude (rad)")
    parser.add_argument("--hold-s", type=float, default=2.0)
    parser.add_argument("--speed", type=int, default=120, help="low-speed direction check")
    args = parser.parse_args()

    cfg = load_arm_config(args.config)
    torque = int(cfg.get("jog_torque", 350))
    arm = AmArmFeetech(args.config)
    arm.connect()

    def _shutdown(*_args: object) -> None:
        print("\ninterrupt — torque OFF", flush=True)
        arm.disconnect(disable_torque=True)
        sys.exit(130)

    signal.signal(signal.SIGINT, _shutdown)

    print("holding current pose...", flush=True)
    arm.hold_current(torque=torque, speed=args.speed)
    time.sleep(args.hold_s)

    zero = np.zeros(3, dtype=np.float64)
    for i, name in enumerate(arm.joint_names):
        print(f"jog {name} ±{args.delta} rad", flush=True)
        for sign in (+1.0, -1.0, 0.0):
            q = zero.copy()
            q[i] = sign * args.delta
            arm.write_model_rad(q, torque=torque, speed=args.speed)
            time.sleep(0.7)
        arm.write_model_rad(zero, torque=torque, speed=args.speed)
        time.sleep(0.4)

    print(f"hold zero offset {args.hold_s}s then exit", flush=True)
    arm.write_model_rad(zero, torque=torque, speed=args.speed)
    time.sleep(args.hold_s)
    arm.disconnect(disable_torque=True)
    print("done — torque OFF", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
