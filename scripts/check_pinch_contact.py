#!/usr/bin/env python3
"""Compare a powered, operator-observed pinch with the same nominal FK pose.

Only thumb + one selected finger are opened/enabled. No wrist/arm, neutral,
calibration sweep, automatic closing, fit or operating-profile edits. Commands
are manual, bounded increments, with all seven parent/distal joints held.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from orca_core import OrcaHand
from orca_core.hand_config import OrcaHandConfig
from orca_teleop.retargeting.adaptive_analytical import AdaptiveAnalyticalRetargeter
from orca_teleop.retargeting.contact_calibration import FINGERS, fingerprint
from orca_teleop.retargeting.surface_contacts import SkinContactGeometry

ROOT = Path(__file__).resolve().parents[1]


class PairProbe(OrcaHand):
    _persist_calibration = False

    def _persist_resolved_driver(self, existing):
        pass


def make_probe(config_path, finger):
    if finger not in FINGERS[1:]:
        raise ValueError('Select index, middle, ring or pinky')
    cfg = OrcaHandConfig.from_config_path(config_path)
    joints = [j for j in cfg.joint_ids if j.startswith(('thumb_', finger+'_'))]
    if len(joints) != 7 or cfg.motor_type != 'feetech' or not cfg.port or cfg.port == 'auto':
        raise ValueError('고정 Feetech 손 포트와 엄지 4개 + 상대 손가락 3개 관절이 필요합니다.')
    cfg = replace(cfg, joint_ids=joints,
        motor_ids=[cfg.joint_to_motor_map[j] for j in joints],
        joint_to_motor_map={j: cfg.joint_to_motor_map[j] for j in joints},
        joint_inversion_dict={j: cfg.joint_inversion_dict[j] for j in joints},
        joint_roms_dict={j: cfg.joint_roms_dict[j] for j in joints},
        neutral_position={j: cfg.neutral_position[j] for j in joints},
        calibration_sequence=[], joint_encoder_joints=None, use_joint_feedback=False,
        calibration_min_motor_span_rad={j: v for j,v in cfg.calibration_min_motor_span_rad.items() if j in joints})
    hand = PairProbe(config=cfg)
    if not hand.is_calibrated():
        raise ValueError('선택한 손가락의 모터 캘리브레이션이 유효하지 않습니다.')
    return hand


def fresh_angles(hand):
    angles = hand.get_joint_position().data
    if not hand.motor_client.last_read_ok:
        raise RuntimeError('모터 읽기 실패. 캐시 값으로 이동하거나 저장하지 않습니다.')
    result = {}
    for joint in hand.config.joint_ids:
        value = angles.get(joint)
        lo, hi = hand.config.joint_roms_dict[joint]
        tolerance = encoder_tick_deg(hand, joint)
        if value is None or not math.isfinite(value) or not lo-tolerance <= value <= hi+tolerance:
            raise RuntimeError(f'{joint}: 엔코더 환산값 {value}가 ROM [{lo}, {hi}] 밖입니다.')
        result[joint] = float(value)
    return result


def encoder_tick_deg(hand, joint):
    """One Feetech encoder count in calibrated joint degrees (at most .25°)."""
    motor = hand.config.joint_to_motor_map[joint]
    ratio = abs(hand.joint_to_motor_ratios_dict[motor])
    if not math.isfinite(ratio) or ratio <= 0:
        raise RuntimeError(f'Invalid calibration ratio: {joint}')
    return min(.25, (2*math.pi/4096)/ratio + 1e-6)


def begin_hold(hand):
    measured = fresh_angles(hand)
    # Preserve raw feedback in snapshots. Only the command is clamped to the
    # existing ROM when a reading is one encoder count beyond an endpoint.
    pose = {j: float(np.clip(v, *hand.config.joint_roms_dict[j])) for j,v in measured.items()}
    hand.set_max_current(hand.config.max_current)
    hand.set_joint_positions(pose)  # Preload current pose, never an old goal.
    if hand.enable_torque():
        raise RuntimeError('선택한 손가락 토크 활성화 응답을 받지 못했습니다.')
    return pose


def jog(hand, command, joint, delta, speed=5.0):
    if joint not in command or not math.isfinite(delta) or not 0 < abs(delta) <= 5:
        raise ValueError('표시된 관절 번호와 -5~+5° 이동량(0 제외)을 지정하세요.')
    if not math.isfinite(speed) or not 0 < speed <= 10:
        raise ValueError('진단 속도는 0~10°/s 범위여야 합니다.')
    fresh_angles(hand)  # Fail before movement on stale/out-of-ROM feedback.
    goal = command[joint]+delta
    lo, hi = hand.config.joint_roms_dict[joint]
    if not lo <= goal <= hi:
        raise ValueError(f'{joint} 목표 {goal:.1f}°가 ROM [{lo}, {hi}] 밖입니다.')
    steps = math.ceil(abs(delta)/(speed*.05))
    origin = command[joint]
    result = dict(command)
    for n in range(1, steps+1):
        fresh_angles(hand)
        result[joint] = origin + delta*n/steps
        hand.set_joint_positions(result)
        time.sleep(.05)
    return result


def move_reference(hand, command, goal, speed=5.0):
    """Slowly replay a confirmed passive pose as a diagnostic, never an IK target."""
    if set(goal) != set(command) or not math.isfinite(speed) or not 0 < speed <= 10:
        raise ValueError('기준 자세는 선택한 7개 관절과 유효한 진단 속도가 필요합니다.')
    for j,v in goal.items():
        if not math.isfinite(v) or not hand.config.joint_roms_dict[j][0] <= v <= hand.config.joint_roms_dict[j][1]:
            raise ValueError(f'기준 자세가 ROM 밖입니다: {j}')
    steps = max(1, math.ceil(max(abs(goal[j]-command[j]) for j in command)/(speed*.05)))
    result = dict(command)
    for n in range(1, steps+1):
        fresh_angles(hand)
        result = {j: command[j]+(goal[j]-command[j])*n/steps for j in command}
        hand.set_joint_positions(result)
        time.sleep(.05)
    return result


def reference_poses(path, r, finger, joints):
    dataset = json.loads(path.read_text())
    if dataset.get('schema') != 1 or dataset.get('fingerprint') != fingerprint(r):
        raise ValueError('수동 접촉 기록이 현재 모터 캘리·모델과 다릅니다. 재생하지 않습니다.')
    rows = [s for s in dataset['samples'] if s.get('finger') == finger]
    if not rows or any(s.get('operator_confirmed_contact') is not True for s in rows):
        raise ValueError('확인된 수동 접촉 기준 자세가 없습니다.')
    poses = []
    for row in rows:
        values = {j: float(row['angles_deg'][j]) for j in joints}
        inspect_pose(r, r._surface_contacts or SkinContactGeometry(r), values, finger)
        poses.append(values)
    return poses


def inspect_pose(r, geometry, angles, finger):
    expected = [j for j in r._finger_joint_ids if j.startswith(('thumb_', finger+'_'))]
    if set(angles) != set(expected):
        raise ValueError('Snapshot must contain exactly the selected seven joints')
    q = r._neutral_qpos_phys.copy()
    for joint, value in angles.items():
        i = r._finger_joint_ids.index(joint)
        tolerance = np.deg2rad(encoder_tick_deg(r._hand, joint))
        if not math.isfinite(value) or not r._lower_phys[i]-tolerance <= np.deg2rad(value) <= r._upper_phys[i]+tolerance:
            raise ValueError(f'Invalid angle: {joint}')
        q[i] = np.deg2rad(value)
    positions, _ = r._robot.compute_positions_and_jacobians(
        r._full_model_qpos(q), r._computed_frame_indices, r._contact_offsets)
    pair = FINGERS[1:].index(finger)
    points = positions[r._tip_indices[[0, pair+1]]]
    result = geometry.inspect_pair(pair)
    result.update(nominal_tip_frame_gap_mm=float(np.linalg.norm(points[1]-points[0])*1000),
                  nominal_tip_points_world_mm=(points*1000).tolist(),
                  unrelated_joints='configured neutral for FK only; not measured or commanded')
    if r._surface_contacts is not None:
        bias = float(r._surface_contacts.biases[pair]*1000)
        goal = bias-float(r._surface_contact_closure_margin_cm[pair]*10)
        result.update(passive_clearance_bias_mm=bias, configured_closed_target_mm=goal,
                      configured_target_residual_mm=result['signed_hull_clearance_mm']-goal)
    return result


def save_snapshot(path, r, geometry, hand, command, finger, confirmed):
    samples = [fresh_angles(hand) for _ in range(1)]
    for _ in range(9):
        time.sleep(.05)
        samples.append(fresh_angles(hand))
    values = np.array([[s[j] for j in command] for s in samples])
    if np.max(np.ptp(values, axis=0)) > 1:
        raise ValueError('기록 중 손가락이 1° 넘게 움직였습니다. 같은 자세를 유지하세요.')
    measured = dict(zip(command, np.median(values, axis=0).tolist(), strict=True))
    identity = fingerprint(r)
    dataset = dict(schema=1, method='powered_contact_check', fingerprint=identity, samples=[])
    if path.exists():
        dataset = json.loads(path.read_text())
        if dataset.get('method') != 'powered_contact_check' or dataset.get('fingerprint') != identity:
            raise ValueError('기존 파일이 다른 진단/캘리 기록입니다. 새 --samples 경로를 지정하세요.')
    row = dict(timestamp_ns=time.time_ns(), finger=finger,
        operator_confirmed_pad_contact=bool(confirmed),
        contact_observation='operator_confirmed_touching' if confirmed else 'not_confirmed',
        command_deg=dict(command), encoder_estimated_deg=measured,
        encoder_is_independent_joint_sensor=False,
        command_model=inspect_pose(r, geometry, command, finger),
        encoder_model=inspect_pose(r, geometry, measured, finger))
    dataset['samples'].append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dataset, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('verify', 'audit'))
    p.add_argument('--execute', action='store_true')
    p.add_argument('--finger', choices=FINGERS[1:], default='index')
    p.add_argument('--hand-config', default=str(ROOT/'configs/orcahand-right-v1-feetech/config.yaml'))
    p.add_argument('--urdf', default=str(ROOT/'orcahand_description/v1/models/urdf/orcahand_right.urdf'))
    p.add_argument('--retarget-config', default=str(ROOT/'configs/adaptive_analytical_orca_v1_pinch_calibrated.yaml'))
    p.add_argument('--samples', type=Path, default=ROOT/'data/contact_calibration/powered_contacts.json')
    p.add_argument('--reference-samples', type=Path, default=ROOT/'data/contact_calibration/full_hand_contacts.json')
    args = p.parse_args(argv)
    hand = None
    try:
        r = AdaptiveAnalyticalRetargeter(hand_config_path=args.hand_config, urdf_path=args.urdf, config_path=args.retarget_config)
        geometry = r._surface_contacts or SkinContactGeometry(r)
        if args.mode == 'audit':
            dataset = json.loads(args.samples.read_text())
            if dataset.get('method') != 'powered_contact_check' or dataset.get('fingerprint') != fingerprint(r):
                raise ValueError('현재 모델·캘리와 일치하는 구동 접촉 진단 기록이 필요합니다.')
            for row in dataset['samples']:
                print(json.dumps(dict(finger=row['finger'],
                    operator_confirmed_pad_contact=row['operator_confirmed_pad_contact'],
                    command_model=inspect_pose(r, geometry, row['command_deg'], row['finger']),
                    encoder_model=inspect_pose(r, geometry, row['encoder_estimated_deg'], row['finger'])), ensure_ascii=False))
            return 0
        if not args.execute:
            print('실물 비교: ./scripts/hand.sh contacts verify --execute\n현재 자세에서 시작하며 입력 전에는 조그하지 않습니다.')
            return 0
        hand = make_probe(args.hand_config, args.finger)
        poses = reference_poses(args.reference_samples, r, args.finger, hand.config.joint_ids)
        if not os.path.exists(hand.config.port):
            raise ValueError(f'손 USB 포트가 없습니다: {hand.config.port}')
        error = hand._try_port(hand.config.port, hand.config)
        if error is not None:
            raise RuntimeError(str(error))
        command = begin_hold(hand)
        joints = list(command)
        print('현재 자세 유지. 엄지와 선택한 손가락 7개 모터만 사용합니다. 손목·팔은 연결하지 않습니다.', flush=True)
        for n, joint in enumerate(joints, 1):
            print(f'{n}: {joint} (ID {hand.config.joint_to_motor_map[joint]}) {command[joint]:.1f}°', flush=True)
        print(f'입력 예: 4 +2 / 4 -2 (번호, 이동각). pose 1~{len(poses)}: 수동 접촉 자세를 5°/s로 진단 재생. '
              'c: 실제 집는 면 접촉 확인 저장, n: 접촉 미확인 상태 저장, p: 상태, q: 토크 해제 종료', flush=True)
        while True:
            value = input('contact> ').strip().lower()
            if value == 'q':
                return 0
            try:
                if value in ('c', 'n'):
                    row = save_snapshot(args.samples, r, geometry, hand, command, args.finger, value == 'c')
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                elif value == 'p':
                    print(json.dumps(inspect_pose(r, geometry, fresh_angles(hand), args.finger), ensure_ascii=False), flush=True)
                elif value.startswith('pose '):
                    index = int(value.split()[1])
                    if index not in range(1, len(poses)+1):
                        raise ValueError('표시된 기준 자세 번호를 지정하세요.')
                    command = move_reference(hand, command, poses[index-1])
                    print('기준 자세 재생 완료. 실제 패드 접촉을 눈으로 확인하세요.', flush=True)
                else:
                    number, delta = value.split()
                    if int(number) not in range(1, len(joints)+1):
                        raise ValueError('표시된 관절 번호를 지정하세요.')
                    command = jog(hand, command, joints[int(number)-1], float(delta))
                    print('명령:', command, flush=True)
            except (ValueError, TypeError) as exc:
                print(exc, flush=True)
    except (KeyboardInterrupt, EOFError):
        print('중단: 선택한 7개 손가락 모터의 토크를 해제합니다.', flush=True)
        return 130
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f'접촉 비교 중단: {exc}', flush=True)
        return 1
    finally:
        if hand is not None:
            hand.disconnect()


if __name__ == '__main__':
    raise SystemExit(main())
