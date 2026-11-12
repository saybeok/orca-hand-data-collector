"""Check an installed collector without opening motors, cameras or Quest."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def initialize_arm_home(root):
    config_path = root / 'configs/am-arm200-3dof/config.yaml'
    config = yaml.safe_load(config_path.read_text())
    target = (config_path.parent / config['home_pose_file']).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Arm home must be inside this project.')
    if target.exists():
        return False  # Never replace a pose already taught on this installation.
    reference = root / 'configs/am-arm200-3dof/initial_pose.json'
    value = json.loads(reference.read_text())
    names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex']
    if (value.get('version') != 1 or value.get('joint_names') != names
            or value.get('motor_ids') != [config['joint_to_motor_map'][name] for name in names]
            or value.get('joint_signs') != [config['joint_signs'][name] for name in names]
            or value.get('port') != config['port']
            or len(value.get('motor_rad', [])) != 3
            or not all(math.isfinite(v) for v in value['motor_rad'])):
        raise ValueError('Bundled initial pose does not match this arm/configuration.')
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(reference.read_bytes())
        stream.flush()
        os.fsync(stream.fileno())
    return True


def check(root):
    import importlib.metadata
    for name in ('mujoco', 'pinocchio', 'nlopt', 'aiohttp', 'av', 'lerobot'):
        importlib.import_module(name)
    from orca_teleop.am200.composer import Am200ActionComposer
    from orca_teleop.am200.arm_client import AmArmFeetech
    from orca_teleop.collector_app import Collector
    from orca_teleop.retargeting.adaptive_analytical import AdaptiveAnalyticalRetargeter

    subprocess.run(['/usr/bin/python3', '-c',
        'import tkinter, numpy, cv2, yaml; from PIL import Image, ImageTk; from Xlib import display'], check=True)
    model = root / 'orca_am200/robot_model/am_arm200_orca_right.xml'
    composer = Am200ActionComposer.from_model_path(model,
        hand_config_path=root / 'configs/orcahand-right-v1-feetech/config.yaml')
    if composer.joint_map.dof != 20:
        raise ValueError('Expected the 20-DoF AM arm + ORCA model.')
    # Constructor and file validation only; no connect/read/write calls.
    arm = AmArmFeetech(root / 'configs/am-arm200-3dof/config.yaml')
    if arm._load_home_pose() is None:
        raise ValueError('Saved arm initial pose is missing; run setup with --initialize.')
    AdaptiveAnalyticalRetargeter(
        hand_config_path=str(root / 'configs/orcahand-right-v1-feetech/config.yaml'),
        urdf_path=str(root / 'orcahand_description/v1/models/urdf/orcahand_right.urdf'),
        config_path=str(root / 'configs/adaptive_analytical_orca_v1_pinch_calibrated.yaml'))
    collector = Collector(root)
    if not collector.settings['task'].strip():
        raise ValueError('Task instruction is empty.')
    print('모터·카메라 연결 없이 설치 점검 완료: 20-DoF 모델, 초기 자세, 수집 설정, GUI/인코더 의존성.')
    print('작업 지시:', collector.settings['task'])
    for package in ['lerobot', 'mujoco', 'pin', 'torch']:
        print(f'{package}: {importlib.metadata.version(package)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initialize', action='store_true')
    args = parser.parse_args()
    if args.initialize:
        created = initialize_arm_home(ROOT)
        print('같은 로봇의 저장된 초기 자세를 복원했습니다.' if created else '기존 초기 자세를 유지합니다.')
    check(ROOT)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'설치 점검 실패: {error}', file=sys.stderr)
        raise SystemExit(1)
