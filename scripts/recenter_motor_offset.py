#!/usr/bin/env python3
"""Move a Feetech motor's encoder wrap point away from its working range.

HLS servos run single-turn (raw 0-4095). After re-tensioning, a spool can come
to rest so that its finger's travel straddles the 0/4095 seam; the servo cannot
cross it, so calibration records a sliver of a range (or a clipped hardstop).
This issues the servo's OFSCAL instruction so the *current* pose reads
mid-encoder (2048), leaving ~180 deg of margin on both sides. The firmware
computes and stores the EEPROM offset itself; HLS encodes that register
differently from the STS docs, so it is never written by hand here.

Before running: torque off, hold each listed finger joint half-bent (middle of
its travel). Afterwards the stored motor limits are stale — recalibrate.

    python scripts/recenter_motor_offset.py --motors 6 8 15            # dry run
    python scripts/recenter_motor_offset.py --motors 6 8 15 --apply
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import yaml
from orca_core.hardware.feetech.sms_sts import (
    SMS_STS_PRESENT_POSITION_L,
    SMS_STS_TORQUE_ENABLE,
)
from orca_core.hardware.feetech_client import FeetechClient

CENTER = 2048
TOLERANCE = 40
POS_SIGN_BIT = 15
WRIST_ID = 17


def _circular(delta: int) -> int:
    return (delta + 2048) % 4096 - 2048


def _check(result: int, error: int, what: str) -> None:
    if result != 0 or error != 0:
        raise RuntimeError(f"{what} failed (result={result}, error={error})")


class _Motor:
    def __init__(self, ph, motor_id: int) -> None:
        self.ph = ph
        self.id = motor_id

    def position(self) -> int:
        raw, result, error = self.ph.read2ByteTxRx(self.id, SMS_STS_PRESENT_POSITION_L)
        _check(result, error, f"read position of motor {self.id}")
        return self.ph.scs_tohost(raw, POS_SIGN_BIT)

    def set_current_as(self, position: int) -> None:
        _check(*self.ph.write1ByteTxRx(self.id, SMS_STS_TORQUE_ENABLE, 0), "torque off")
        _check(*self.ph.reOfsCal(self.id, position), f"OFSCAL on motor {self.id}")
        time.sleep(0.1)


def recenter(motor: _Motor, apply: bool) -> bool:
    pos = motor.position()
    shift = _circular(pos - CENTER)
    print(f"motor {motor.id:2d}: position {pos:5d}  ({shift:+5d} steps from center)")
    if abs(pos - CENTER) <= TOLERANCE:
        print("          already centered")
        return True
    if not apply:
        print(f"          would make this pose read {CENTER}")
        return True
    motor.set_current_as(CENTER)
    after = motor.position()
    if abs(after - CENTER) <= TOLERANCE:
        print(f"          position now {after}")
        return True
    print(f"          OFSCAL did not center the motor in the writable 0..4095 frame (reads {after}); do not run teleop")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--motors", type=int, nargs="+", required=True)
    parser.add_argument("--config", default=os.environ.get("ORCA_MODEL_PATH"))
    parser.add_argument("--apply", action="store_true", help="write EEPROM (default: dry run)")
    parser.add_argument(
        "--allow-wrist",
        action="store_true",
        help="permit the wrist motor (invalidates the wrist calibration too)",
    )
    args = parser.parse_args()
    if not args.config:
        parser.error("--config or $ORCA_MODEL_PATH is required")
    if WRIST_ID in args.motors and not args.allow_wrist:
        parser.error("motor 17 is the wrist; pass --allow-wrist to include it")

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    client = FeetechClient(args.motors, port=str(cfg["port"]), baudrate=int(cfg["baudrate"]))
    client.connect()
    ok = True
    try:
        client.set_torque_enabled(args.motors, False)
        for motor_id in args.motors:
            ok &= recenter(_Motor(client.packet_handler, motor_id), args.apply)
    finally:
        client.disconnect()

    if not args.apply:
        print("\ndry run only — re-run with --apply to write EEPROM")
    elif ok:
        print("\noffsets written — now recalibrate: ./scripts/hand.sh calibrate")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
