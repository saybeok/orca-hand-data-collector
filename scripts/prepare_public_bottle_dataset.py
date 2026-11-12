"""Prepare the frozen, curated 75-episode collection for Hub publication.

Writes only a derived LeRobot dataset and conversion artifacts; never uploads.
Fresh conversions contain only selected recordings, including at video level.
"""
import argparse
from collections import Counter
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import uuid

import numpy as np

from orca_teleop.collection_control import atomic_json
from orca_teleop.deferred_recording import capture_fingerprint, convert_capture

PROJECT = Path(__file__).resolve().parents[1]


def identity(episode):
    return (episode['rows_sha256'], tuple(sorted((k, v['sha256']) for k, v in episode['videos'].items())))


def prepare(inventory_path, previous_capture, output):
    inventory = json.loads(inventory_path.read_text())
    sources = [Path(item['path']) for item in inventory['sessions']]
    output = output.resolve()
    roots = []
    records = []
    arrays = {'observation.state': [], 'action': []}
    reference = None
    tasks = set()
    for source, expected in zip(sources, inventory['sessions'], strict=True):
        if capture_fingerprint(source) != expected['fingerprint']:
            raise ValueError(f'Frozen source changed; inspect before publishing: {source}')
        capture = json.loads((source / 'capture.json').read_text())
        session = json.loads((source / 'collection_session.json').read_text())
        signature = {
            'features': capture['features'], 'fps': capture['fps'],
            'joint_ids': session['joint_ids'], 'angle_units': session['angle_units'],
            'arm_reference': session['arm_reference'],
            'hardware_sha256': {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                for name in ('collection_hand_config.yaml', 'collection_calibration.yaml',
                             'collection_arm_config.yaml', 'collection_arm_home.json')},
        }
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError(f'Incompatible robot/camera/calibration snapshot: {source}')
        paths = sorted(source.glob('episode-*/episode.json'))
        if len(paths) != expected['episodes']:
            raise ValueError('Saved episode count changed.')
        episodes = []
        for index, path in enumerate(paths):
            episode = json.loads(path.read_text())
            if episode['index'] != index or episode['frames'] / capture['fps'] < 20:
                raise ValueError(f'Noncontiguous or short episode: {path}')
            with np.load(path.parent / 'rows.npz', allow_pickle=False) as rows:
                for key in arrays:
                    value = rows[key]
                    if value.shape != (episode['frames'], 20) or not np.isfinite(value).all():
                        raise ValueError(f'Invalid 20-DoF data: {path}')
                    arrays[key].append(value.copy())
                tasks.update(str(t) for t in np.unique(rows['task']))
            episodes.append(episode)
            records.append({'index': len(records), 'source_session': source.name,
                            'source_episode': index, 'frames': episode['frames']})
        roots.append((source, episodes))
    if len(records) != 75 or tasks != {'move the bottle.'}:
        raise ValueError(f'Expected exactly 75 bottle episodes, got {len(records)} / {tasks}')
    previous = [json.loads(p.read_text()) for p in sorted(previous_capture.glob('episode-*/episode.json'))]
    first_three = [episode for _, episodes in roots[:3] for episode in episodes]
    if Counter(map(identity, previous)) != Counter(map(identity, first_three)) or len(previous) != 29:
        raise ValueError('Existing converted 29 episodes do not match the first three frozen sessions.')
    local_ids = list(map(identity, first_three))
    if len(set(local_ids + list(map(identity, roots[3][1])))) != 75:
        raise ValueError('Duplicate source episodes detected.')

    public_manifest = {'format': 'orca-public-collection-v1', 'episodes': records,
        'source_sessions': [{'name': p.name, 'episodes': len(e), 'fingerprint': capture_fingerprint(p)}
                            for p, e in roots],
        'minimum_episode_seconds': 20, 'task': 'move the bottle.', 'fps': reference['fps'],
        'normalization_scope': 'All 75 published episodes; refit on a chosen training split for held-out evaluation.'}
    if output.exists():
        if json.loads((output / 'collection_manifest.json').read_text()) != public_manifest:
            raise ValueError('Existing output has different provenance; refusing overwrite.')
        summary = json.loads((output / 'preparation_summary.json').read_text())
        if summary.get('strict_video_selection'):
            logging.info('Reusing verified public dataset: %s', output)
            return output

    converted = []
    logging.info('Sources verified: 75 episodes. Fresh video selection excludes all discarded footage.')
    for source, _ in roots:
        last = [None]
        def progress(**fields):
            current = fields.get('episode')
            if current != last[0]:
                logging.info('Encoding %s episode %s/%s', source.name, current, fields.get('total'))
                last[0] = current
        converted.append(convert_capture(source, progress))

    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.compute_stats import compute_episode_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import load_stats, write_stats

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name('.' + output.name + '-preparing-' + uuid.uuid4().hex)
    try:
        aggregate_datasets(
            repo_ids=[json.loads((p / 'capture.json').read_text())['repo_id'] for p, _ in roots],
            roots=converted, aggr_repo_id='local/orca-hand-bottle',
            aggr_root=staging, video_files_size_in_mb=1,
        )
        numeric = {key: np.concatenate(values) for key, values in arrays.items()}
        stats = load_stats(staging)
        stats.update(compute_episode_stats(numeric, reference['features']))
        write_stats(stats, staging)
        for key in numeric:
            for quantile in ['q01', 'q99']:
                if np.asarray(stats[key][quantile]).shape != (20,) or not np.isfinite(stats[key][quantile]).all():
                    raise ValueError('Pi05 quantile statistics are missing/invalid.')
        atomic_json(staging / 'collection_manifest.json', public_manifest)
        atomic_json(staging / 'robot_metadata.json', {
            'robot': 'AM-ARM200 3-DoF + ORCA Hand v1 Feetech right 17-DoF',
            'joint_order': reference['joint_ids'], 'angle_units': reference['angle_units'],
            'arm_reference': reference['arm_reference'],
            'hand_coordinate_convention': 'Canonical model radians, including model ref offsets; use CanonicalJointMap for hardware conversion.',
            'hardware_sha256': reference['hardware_sha256'],
        })
        hardware = staging / 'robot_configuration'
        hardware.mkdir()
        # Publish calibration/geometry conventions; omit workstation paths and USB port identities.
        import yaml
        for name in ['collection_hand_config.yaml', 'collection_calibration.yaml', 'collection_arm_config.yaml']:
            data = yaml.safe_load((sources[0] / name).read_text())
            if isinstance(data, dict):
                for key in ['port', 'home_pose_file', 'config_path', 'calibration_path']:
                    data.pop(key, None)
            (hardware / name.removeprefix('collection_')).write_text(yaml.safe_dump(data, sort_keys=False))
        frame_count = sum(e['frames'] for e in records)
        (staging / 'README.md').write_text(f'''---
pretty_name: ORCA Hand and AM-ARM200 Bottle Demonstrations
language:
- en
tags:
- lerobot
- robotics
- imitation-learning
configs:
- config_name: default
  data_files:
  - split: train
    path: data/*/*.parquet
---

# ORCA Hand + AM-ARM200: move the bottle

75 operator-saved physical-robot demonstrations, {frame_count:,} synchronized
frames at 30 Hz. Task: `move the bottle.`

- Robot: AM-ARM200 arm (3 DoF) + right ORCA Hand v1 Feetech (17 DoF).
- RGB cameras: bottom, wrist and RealSense, each 640x480.
- `observation.state`: measured 20-joint canonical model angles in radians.
- `action`: last successfully dispatched 20-joint target in the same frame.
- Format: LeRobot v3.0, generated with LeRobot 0.4.4; MP4 videos and Parquet rows.
- Joint order, arm zero/signs and hand conversion convention: `robot_metadata.json`.
- Source-session provenance: `collection_manifest.json`.

Failed retries were discarded during recording. Episodes shorter than 20 seconds
were excluded. Successful task completion is operator-labelled; no automated
success scoring is claimed. Camera synchronization is recorder-side sampling,
not hardware-triggered exposure synchronization. Hand angles are derived from
motor encoders, not independent link-angle sensors.

## Training

Use the dataset repository ID with LeRobot. Data remain 20-dimensional and in
canonical model radians: do not convert them to degrees or manually pad saved
rows. Pi05 input preparation handles image resizing and dimensional padding.
`meta/stats.json` includes exact numerical `q01`/`q99` over all published rows.
For held-out evaluation, fit normalization on the chosen training split only.
No held-out performance or physical rollout success rate is reported here.

The `train` metadata split exposes all 75 episodes. Future collection revisions
should be pinned by commit or dataset snapshot when reporting training results.
''')
        ds = LeRobotDataset('local/orca-hand-bottle', root=staging, video_backend='pyav')
        if ds.num_episodes != 75 or ds.num_frames != frame_count:
            raise ValueError('Merged episode/frame count mismatch.')
        if {int(i) for i in ds.hf_dataset['episode_index']} != set(range(75)):
            raise ValueError('Merged indices are not contiguous.')
        for i, record in enumerate(records):
            if int(ds.meta.episodes[i]['length']) != record['frames']:
                raise ValueError('Merged episode length mismatch.')
        # Validate source/merged numeric rows and exact decoded image pixels across the merge boundary.
        for index in [0, 28, 29, 74]:
            record = records[index]
            source_slot = next(i for i, (p, _) in enumerate(roots) if p.name == record['source_session'])
            src_root = converted[source_slot]
            old_index = record['source_episode']
            original = LeRobotDataset('local/source', root=src_root, video_backend='pyav')
            a = original[int(original.meta.episodes[old_index]['dataset_from_index'])]
            b = ds[int(ds.meta.episodes[index]['dataset_from_index'])]
            for key in ['observation.state', 'action', *ds.meta.video_keys]:
                np.testing.assert_array_equal(a[key].numpy(), b[key].numpy())
        # A public artifact must not carry hidden, unindexed rejected footage.
        import av
        actual_frames = 0
        for path in (staging / 'videos').rglob('*.mp4'):
            with av.open(str(path)) as container:
                count = container.streams.video[0].frames
                if count <= 0:
                    raise ValueError(f'Missing video frame count: {path}')
                actual_frames += count
        if actual_frames != frame_count * 3:
            raise ValueError('Video streams contain missing or unindexed frames.')
        atomic_json(staging / 'preparation_summary.json', {
            'episodes': 75, 'frames': frame_count, 'fps': 30, 'state_dim': 20, 'action_dim': 20,
            'cameras': ['bottom', 'wrist', 'realsense'], 'source_encoding': 'JPEG quality 95',
            'numeric_quantiles': ['q01', 'q99'], 'upload_performed': False,
            'strict_video_selection': True, 'encoded_video_frames': actual_frames,
        })
        if output.exists():
            previous_output = output.with_name('.' + output.name + '-superseded-' + uuid.uuid4().hex)
            output.rename(previous_output)
            try:
                staging.rename(output)
            except BaseException:
                previous_output.rename(output)
                raise
            shutil.rmtree(previous_output)
        else:
            staging.rename(output)
        logging.info('PUBLIC DATASET READY: %s | 75 episodes, %d frames', output, frame_count)
        return output
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inventory', type=Path,
                        default=PROJECT / 'data/act/external_dataset_inventory.json',
                        help='Frozen source inventory, including verified recovery copies if needed.')
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ.setdefault('HF_DATASETS_CACHE', str(PROJECT / 'data/act/.hf_cache'))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s | %(message)s')
    prepare(args.inventory,
            PROJECT / 'data/act/bottle_20261108_min20/capture', args.output)


if __name__ == '__main__':
    main()
