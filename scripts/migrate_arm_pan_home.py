#!/usr/bin/env python3
"""Explicitly match taught-home pan polarity metadata to the arm config.

No serial devices are opened and no motor positions are changed. The config
must already specify the desired shoulder_pan sign (+1 or -1); only an
otherwise identical home with the opposite pan sign is accepted. Without
--apply this only reports the proposed metadata update.
"""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex']


def migrate(config_path, *, apply=False):
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    signs = [float(cfg['joint_signs'][j]) for j in JOINTS]
    if any(s not in (-1, 1) for s in signs):
        raise ValueError('Arm joint signs must be +1 or -1')
    home_path = (config_path.parent/cfg['home_pose_file']).resolve()
    if not home_path.exists():
        return {'status': 'no_saved_home', 'path': str(home_path)}
    original = home_path.read_bytes()
    home = json.loads(original)
    if (home.get('version') != 1 or home.get('joint_names') != JOINTS
            or home.get('motor_ids') != [cfg['joint_to_motor_map'][j] for j in JOINTS]
            or home.get('port') != cfg['port']):
        raise ValueError('Home belongs to a different arm/configuration; not modified')
    pose = home.get('motor_rad', [])
    if len(pose) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in pose):
        raise ValueError('Home must contain three finite motor positions')
    if home.get('joint_signs') == signs:
        return {'status': 'already_matches', 'path': str(home_path)}
    if home.get('joint_signs') != [-signs[0], *signs[1:]]:
        raise ValueError('Changes beyond shoulder_pan polarity detected; not modified')
    result = {'status': 'proposed', 'path': str(home_path),
              'old_signs': home['joint_signs'], 'new_signs': signs,
              'motor_rad_unchanged': pose}
    if not apply:
        return result
    backup = home_path.with_name(f'{home_path.stem}.before_pan_fix_{time.time_ns()}.json')
    backup.write_bytes(original)
    old_sign = home['joint_signs'][0]
    home['joint_signs'] = signs
    home['pan_polarity_migration'] = {'old_sign': old_sign, 'new_sign': signs[0],
                                     'updated_at_unix_s': time.time(),
                                     'motor_positions_changed': False}
    fd, temporary = tempfile.mkstemp(prefix=home_path.name+'.', dir=home_path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(home, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        if home_path.read_bytes() != original:
            raise RuntimeError('Home changed during migration; not replaced')
        os.replace(temporary, home_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {**result, 'status': 'updated', 'backup': str(backup)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm-config', type=Path, default=ROOT/'configs/am-arm200-3dof/config.yaml')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(migrate(args.arm_config, apply=args.apply), indent=2, ensure_ascii=False))
