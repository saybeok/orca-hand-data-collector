#!/usr/bin/env python3
"""Record physical pad contacts, then fit a separate retargeting profile.

Capture releases only finger motors (never the wrist or arm) and never enables
torque or writes goals. The operator confirms contact; software cannot sense it.
Fit is entirely offline. Motor-derived angles remain estimates, not link sensors.
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace
import json
import os
from pathlib import Path
import time

import numpy as np
import yaml
from orca_core import OrcaHand
from orca_teleop.retargeting.adaptive_analytical import AdaptiveAnalyticalRetargeter
from orca_teleop.retargeting.contact_calibration import FINGERS, TaughtContacts, fingerprint, fit_offsets, pose_frames

ROOT = Path(__file__).resolve().parents[1]


class ContactReader(OrcaHand):
    _persist_calibration = False

    def _persist_resolved_driver(self, existing):
        pass


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def stable_angles(hand, joints):
    samples = []
    for _ in range(10):
        angles = hand.get_joint_position().data
        if not hand.motor_client.last_read_ok:
            raise ValueError("모터 읽기 실패: 이번 자세는 저장하지 않았습니다.")
        q = [angles.get(j) for j in joints]
        if any(v is None or not np.isfinite(v) for v in q):
            raise ValueError("유효한 손가락 각도를 읽지 못했습니다.")
        samples.append(q)
        time.sleep(.05)
    if np.max(np.ptp(samples, axis=0)) > 1.0:
        raise ValueError("기록 중 손가락이 움직였습니다. 같은 자세로 유지해 주세요.")
    return dict(zip(joints, np.median(samples, axis=0).tolist(), strict=True))


def capture(args, r):
    identity = fingerprint(r)
    dataset = {"schema": 1, "fingerprint": identity, "samples": []}
    if args.samples.exists():
        dataset = json.loads(args.samples.read_text())
        if dataset.get("schema") != 1 or dataset.get("fingerprint") != identity:
            raise ValueError("기존 기록과 현재 모터 캘리/모델이 다릅니다. 새 --samples 경로를 지정하세요.")
    cfg = r._hand.config
    joints = r._finger_joint_ids
    cfg = replace(
        cfg, joint_ids=list(joints), motor_ids=[cfg.joint_to_motor_map[j] for j in joints],
        joint_to_motor_map={j: cfg.joint_to_motor_map[j] for j in joints},
        joint_inversion_dict={j: cfg.joint_inversion_dict.get(j, False) for j in joints},
        joint_roms_dict={j: cfg.joint_roms_dict[j] for j in joints},
        neutral_position={j: cfg.neutral_position[j] for j in joints},
        calibration_sequence=[], joint_encoder_joints=None, use_joint_feedback=False,
        calibration_min_motor_span_rad={j: v for j, v in cfg.calibration_min_motor_span_rad.items() if j in joints},
    )
    if cfg.motor_type != "feetech" or not cfg.baudrate or not cfg.port or cfg.port == "auto" or not os.path.exists(cfg.port):
        raise ValueError("실물 손의 고정된 Feetech 포트와 baudrate가 필요합니다.")
    print("텔레옵을 종료하고 손을 받쳐 주세요. 손가락 16개 모터만 토크를 해제합니다.")
    print("손목·팔 모터는 연결하지 않습니다. 손가락을 억지로 꺾지 마세요.")
    print("각 쌍마다 굽힘이 다른 접촉 자세 6개를 기록합니다. 손목만 돌리면 안 됩니다.")
    print("실제 집는 패드끼리 가볍게 닿게 하고 Enter. q: 부분 저장 후 종료.")
    hand = ContactReader(config=cfg)
    try:
        if not hand.is_calibrated():
            raise ValueError("먼저 손가락 모터 캘리브레이션을 완료해 주세요.")
        # Do not fall back to the arm adapter if the pinned hand port fails.
        error = hand._try_port(cfg.port, cfg)
        if error is not None:
            raise RuntimeError(str(error))
        stable_angles(hand, joints)  # Require fresh replies from the complete hand.
        if hand.disable_torque():
            raise RuntimeError("일부 손가락 토크 해제에 실패했습니다.")
        for finger, label in zip(FINGERS[1:], ("검지", "중지", "약지", "새끼"), strict=True):
            while sum(s["finger"] == finger for s in dataset["samples"]) < 6:
                count = sum(s["finger"] == finger for s in dataset["samples"])
                if input(f"엄지–{label} 접촉 {count+1}/6 [Enter 저장 / q 종료]: ").strip().lower() == "q":
                    return
                try:
                    angles = stable_angles(hand, joints)
                    pose_frames(r, angles)  # Validate ROM before keeping the sample.
                    relevant = [j for j in joints if j.startswith(("thumb_", finger + "_"))]
                    previous = [s for s in dataset["samples"] if s["finger"] == finger]
                    if any(max(abs(angles[j] - s["angles_deg"][j]) for j in relevant) < 3 for s in previous):
                        raise ValueError("이전 자세와 너무 비슷합니다. 엄지와 상대 손가락의 굽힘을 바꿔 주세요.")
                    dataset["samples"].append({
                        "finger": finger, "angles_deg": angles,
                        "operator_confirmed_contact": True, "timestamp_ns": time.time_ns(),
                    })
                    write_json(args.samples, dataset)
                    print("저장했습니다.")
                except ValueError as exc:
                    print(exc)
        print(f"기록 완료: {args.samples}\n다음: ./scripts/hand.sh contacts fit")
    finally:
        hand.disconnect()


def fit(args, r):
    dataset = json.loads(args.samples.read_text())
    if dataset.get("schema") != 1 or dataset.get("fingerprint") != fingerprint(r):
        raise ValueError("모터 캘리/매핑/모델이 기록 당시와 다릅니다. 새 접촉 기록이 필요합니다.")
    transformed = []
    for s in dataset["samples"]:
        if s.get("operator_confirmed_contact") is not True or s.get("finger") not in FINGERS[1:]:
            raise ValueError("실물 접촉을 확인하지 않은 표본입니다.")
        positions, rotations = pose_frames(r, s["angles_deg"])
        transformed.append({"finger": s["finger"], "positions_m": positions, "rotations": rotations})
    profile = fit_offsets(transformed)
    profile.update(fingerprint=fingerprint(r), source_samples=str(args.samples.resolve()))
    write_json(args.output, profile)
    cfg = r.config.copy()
    cfg["retarget"] = dict(cfg["retarget"], contact_calibration=str(args.output.resolve()))
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    for f in FINGERS[1:]:
        before = 1000*np.mean(profile["before_validation_errors_m"][f])
        after = 1000*np.mean(profile["validation_errors_m"][f])
        print(f"{f}: 계산에 쓰지 않은 검증 자세의 모델 오차 {before:.2f} → {after:.2f} mm")
    print(f"검증 통과. 별도 실행 설정: {args.output_config}")
    print("기록에 쓰지 않은 실물 자세에서도 접촉을 확인해야 합니다.")


def teach(args, r):
    """Export confirmed configurations without asserting a successful FK fit."""
    library = TaughtContacts(args.samples, r)
    dataset = json.loads(args.samples.read_text())
    dataset.update(
        method="taught_contact_poses",
        source_samples=str(args.samples.resolve()),
        source_sha256=hashlib.sha256(args.samples.read_bytes()).hexdigest(),
        limitations="Operator-confirmed passive contacts; powered contact and grasp forces are not verified",
    )
    write_json(args.output, dataset)
    cfg = r.config.copy()
    cfg["retarget"] = dict(cfg["retarget"], contact_pose_library=os.path.relpath(
        args.output.resolve(), args.output_config.resolve().parent))
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"실제 접촉 자세 {sum(len(v) for v in library.poses.values())}개로 핀치 기준 생성: {args.output_config}")
    print("접근 자세에 가까운 기록을 선택하고, 완전 핀치 시 엄지와 해당 손가락만 그 자세로 이동합니다.")
    print("기구학 보정 검증 통과를 뜻하지 않습니다. 실물 접촉 재현은 실행 후 확인해야 합니다.")


def fit_surface(args, r):
    """Fit four empirical clearances, keeping the continuous adaptive solver."""
    from orca_teleop.retargeting.surface_contacts import SkinContactGeometry, fit_surface_contacts

    dataset = json.loads(args.samples.read_text())
    profile = fit_surface_contacts(dataset, r, SkinContactGeometry(r))
    profile.update(source_samples=str(args.samples.resolve()),
                   source_sha256=hashlib.sha256(args.samples.read_bytes()).hexdigest())
    write_json(args.output, profile)
    cfg = r.config.copy()
    cfg['retarget'] = dict(cfg['retarget'], surface_contact_calibration=os.path.relpath(
        args.output.resolve(), args.output_config.resolve().parent))
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    for finger in FINGERS[1:]:
        before = np.mean(np.abs(profile['before_validation_signed_errors_m'][finger])) * 1000
        after = np.mean(np.abs(profile['validation_signed_errors_m'][finger])) * 1000
        maximum = np.max(np.abs(profile['validation_signed_errors_m'][finger])) * 1000
        print(f'{finger}: 보류한 두 자세의 표면거리 평균 오차 {before:.2f} → {after:.2f} mm; 최대 {maximum:.2f} mm')
    print(f'연속 리타겟팅 보정 후보 생성: {args.output_config}')
    print('기존 점 보정의 3 mm 검증 통과가 아닙니다. 별도 표면거리 기준은 5 mm입니다.')
    print('수동 접촉 기록의 일관성 검사이며, 실물 구동 중 접촉·힘·정밀도를 보장하지 않습니다.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("capture", "fit", "teach", "fit-surface"))
    p.add_argument("--samples", type=Path, default=ROOT / "data/contact_calibration/full_hand_contacts.json")
    p.add_argument("--output", type=Path)
    p.add_argument("--output-config", type=Path)
    p.add_argument("--hand-config", default=str(ROOT / "configs/orcahand-right-v1-feetech/config.yaml"))
    p.add_argument("--urdf", default=str(ROOT / "orcahand_description/v1/models/urdf/orcahand_right.urdf"))
    p.add_argument("--retarget-config", default=str(ROOT / "configs/adaptive_analytical_orca_v1_pinch_refined.yaml"))
    args = p.parse_args()
    outputs = {
        'teach': ('data/contact_calibration/full_hand_taught.json', 'configs/adaptive_analytical_orca_v1_pinch_taught.yaml'),
        'fit-surface': ('data/contact_calibration/full_hand_surface.json', 'configs/adaptive_analytical_orca_v1_pinch_calibrated.yaml'),
    }
    profile, config = outputs.get(args.mode, ('data/contact_calibration/full_hand_profile.json',
                                             'configs/adaptive_analytical_orca_v1_contact_calibrated.yaml'))
    args.output = args.output or ROOT / profile
    args.output_config = args.output_config or ROOT / config
    try:
        r = AdaptiveAnalyticalRetargeter(hand_config_path=args.hand_config, urdf_path=args.urdf, config_path=args.retarget_config)
        if any(r._retarget_cfg.get(k) for k in ('contact_calibration', 'contact_pose_library', 'surface_contact_calibration')):
            raise ValueError("기준 설정에는 기존 접촉 보정이 없어야 합니다.")
        {"capture": capture, "fit": fit, "teach": teach, "fit-surface": fit_surface}[args.mode](args, r)
        return 0
    except KeyboardInterrupt:
        print("\n중단했습니다. 저장된 자세는 다음 실행에서 이어집니다.")
        return 130
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        print(f"접촉 보정 중단: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
