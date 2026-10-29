#!/usr/bin/env python3
"""Isolate a physical finger joint from Quest and the retargeter.

Uses the existing joint-to-motor calibration. Only the selected joint's
motor is opened, enabled, commanded and released; no automatic calibration
or neutral pose is run. Readback is a motor-derived estimate, not a link sensor.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import os
from pathlib import Path
import sys
import time

from orca_core import OrcaHand
from orca_core.hand_config import OrcaHandConfig

JOINT = "thumb_dip"
ROOT = Path(__file__).resolve().parents[1]


class ThumbProbe(OrcaHand):
    _persist_calibration = False

    def _persist_resolved_driver(self, existing):
        # A diagnostic must not rewrite the user's configuration.
        pass


def make_probe(config_path: str, joint: str = JOINT) -> ThumbProbe:
    cfg = OrcaHandConfig.from_config_path(config_path)
    if joint not in cfg.joint_ids or joint == "wrist":
        raise ValueError(f"손가락 관절 이름을 지정해 주세요: {joint}")
    motor = cfg.joint_to_motor_map[joint]
    subset = dataclasses.replace(
        cfg,
        motor_ids=[motor],
        joint_ids=[joint],
        joint_to_motor_map={joint: motor},
        joint_inversion_dict={joint: cfg.joint_inversion_dict.get(joint, False)},
        joint_roms_dict={joint: cfg.joint_roms_dict[joint]},
        neutral_position={joint: cfg.neutral_position[joint]},
        calibration_sequence=[],
        calibration_min_motor_span_rad={joint: cfg.calibration_min_motor_span_rad[joint]}
        if joint in cfg.calibration_min_motor_span_rad else {},
        joint_encoder_joints=None,
        use_joint_feedback=False,
    )
    hand = ThumbProbe(config=subset)
    if not hand.is_calibrated():
        raise RuntimeError(f"{joint} 캘리브레이션 값이 없습니다. 테스트를 중단합니다.")
    return hand


def read_thumb(hand: ThumbProbe) -> float:
    joint = hand.config.joint_ids[0]
    estimate = hand.get_joint_position().data.get(joint)
    if not hand.motor_client.last_read_ok:
        raise RuntimeError("모터 위치를 새로 읽지 못했습니다. 캐시 값으로 움직이지 않습니다.")
    if estimate is None or not math.isfinite(estimate):
        raise RuntimeError(f"{joint} 위치가 유효하지 않습니다.")
    return float(estimate)


def run_probe(hand: ThumbProbe, delta: float, speed: float, hold: float) -> None:
    joint = hand.config.joint_ids[0]
    if not all(math.isfinite(v) for v in (delta, speed, hold)):
        raise ValueError("이동량·속도·유지 시간은 유한한 수여야 합니다.")
    low, high = hand.config.joint_roms_dict[joint]
    if not 0 < abs(delta) <= high - low or not 0 < speed <= 30 or not 0 <= hold <= 10:
        raise ValueError(f"이동량: 설정 범위 ±{high-low:g}° 이내(0 제외), 속도: 0~30°/s, 유지: 0~10초")
    if hand.config.motor_type != "feetech":
        raise RuntimeError("이 확인 코드는 현재 Feetech 손용입니다.")

    motor = hand.config.joint_to_motor_map[joint]
    start = read_thumb(hand)
    if not low <= start <= high:
        raise RuntimeError(f"시작 추정 각도 {start:.1f}°가 ROM [{low}, {high}] 밖입니다.")
    target = min(high, max(low, start + delta))
    if abs(target - start) < 5:
        opposite = -40 if delta > 0 else 40
        raise RuntimeError(f"이 방향으로 남은 범위가 5° 미만입니다. --delta-deg {opposite}으로 반대 방향을 확인하세요.")

    hand.set_max_current(hand.config.max_current)
    # Preload the measured pose before enabling torque, avoiding an old goal.
    hand.set_joint_positions({joint: start})
    if hand.enable_torque():
        raise RuntimeError(f"ID {motor} 모터의 토크 활성화 응답을 받지 못했습니다.")
    print(f"{joint} → ID {motor} | 추정 각도 {start:.1f}° → {target:.1f}° → {start:.1f}°", flush=True)
    print("어느 관절이 어느 방향으로 움직이는지 봐 주세요. Ctrl+C로 중단할 수 있습니다.", flush=True)
    last_report = -math.inf

    def report(command: float) -> None:
        nonlocal last_report
        estimate = read_thumb(hand)
        now = time.monotonic()
        if now - last_report >= .5:
            print(f"ID {motor} | 명령={command:.1f}° | 모터 환산={estimate:.1f}° | 차이={command-estimate:+.1f}°", flush=True)
            last_report = now

    current = start
    for goal in (target, start):
        steps = math.ceil(abs(goal - current) / (speed * .05))
        origin = current
        for step in range(1, steps + 1):
            current = origin + (goal - origin) * step / steps
            hand.set_joint_positions({joint: current})
            time.sleep(.05)
            report(current)
        # Hold long enough to distinguish motion lag from failure to follow.
        for _ in range(math.ceil(hold / .1)):
            time.sleep(.1)
            report(current)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.environ.get("ORCA_MODEL_PATH", str(ROOT / "configs/orcahand-right-v1-feetech/config.yaml")))
    parser.add_argument("--joint", default=JOINT,
                        help="one configured finger joint; default: thumb_dip")
    parser.add_argument("--delta-deg", type=float, default=40)
    parser.add_argument("--speed-deg-s", type=float, default=15)
    parser.add_argument("--hold-s", type=float, default=3)
    args = parser.parse_args()
    hand = None
    try:
        hand = make_probe(args.config, joint=args.joint)
        ok, message = hand.connect(interactive=False)
        if not ok:
            raise RuntimeError(message)
        run_probe(hand, args.delta_deg, args.speed_deg_s, args.hold_s)
        print("왕복 완료. 선택한 모터의 토크를 해제합니다.", flush=True)
        return 0
    except KeyboardInterrupt:
        print("\n중단. 선택한 모터의 토크를 해제합니다.", flush=True)
        return 130
    except Exception as exc:
        print(f"테스트 중단: {exc}", file=sys.stderr)
        return 1
    finally:
        if hand is not None:
            hand.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
