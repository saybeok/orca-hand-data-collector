"""Assemble explicitly selected captures for local ACT training; never upload."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import uuid
from pathlib import Path

import numpy as np

from orca_teleop.collection_control import atomic_json
from orca_teleop.deferred_recording import capture_fingerprint, convert_capture


def prepare(sources: list[Path], destination: Path) -> Path:
    sources = [source.resolve() for source in sources]
    if not sources or len(sources) != len(set(sources)):
        raise ValueError('Select distinct capture session folders.')
    sessions, records = [], []
    reference = None
    for source in sources:
        capture = json.loads((source / 'capture.json').read_text())
        session = json.loads((source / 'collection_session.json').read_text())
        identity = {
            'capture': {k: capture[k] for k in ('format', 'fps', 'features', 'image_encoding', 'jpeg_quality')},
            'session': {k: session[k] for k in ('joint_ids', 'angle_units', 'arm_reference')},
            'hardware': {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                         for name in ('collection_hand_config.yaml', 'collection_calibration.yaml',
                                      'collection_arm_config.yaml', 'collection_arm_home.json')},
        }
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError(f'Camera/joint/calibration/arm-reference mismatch: {source}')
        paths = sorted(source.glob('episode-*/episode.json'))
        if not paths:
            raise ValueError(f'No saved episodes: {source}')
        group = []
        for old_index, path in enumerate(paths):
            episode = json.loads(path.read_text())
            if episode['index'] != old_index or episode['frames'] <= 0:
                raise ValueError(f'Invalid episode index or empty capture: {path}')
            with np.load(path.parent / 'rows.npz', allow_pickle=False) as rows:
                for key in ('observation.state', 'action'):
                    if rows[key].shape != (episode['frames'], 20) or not np.isfinite(rows[key]).all():
                        raise ValueError(f'Invalid 20-DoF rows: {path} / {key}')
                tasks = sorted(set(str(t) for t in rows['task']))
            new_index = len(records)
            record = {'index': new_index, 'source': str(source), 'source_index': old_index,
                      'frames': episode['frames'], 'tasks': tasks}
            records.append(record)
            group.append(new_index)
        sessions.append({'source': str(source), 'fingerprint': capture_fingerprint(source),
                         'episodes': group})
    rng = random.Random(42)
    validation = sorted(rng.choice(s['episodes']) for s in sessions if len(s['episodes']) >= 2)
    manifest = {'format': 'orca-act-preparation-v1', 'sessions': sessions, 'episodes': records,
                'fps': reference['capture']['fps'], 'angle_units': reference['session']['angle_units'],
                'train_episodes': [r['index'] for r in records if r['index'] not in validation],
                'validation_episodes': validation,
                'note': 'Task success is operator-labelled; no automatic semantic filtering. No upload.'}
    destination = destination.resolve()
    if destination.exists():
        previous = json.loads((destination / 'collection_act_preparation.json').read_text())
        if previous != manifest:
            raise ValueError('Prepared capture differs from selected sources; choose a new destination.')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.with_name(f'.{destination.name}-preparing-{uuid.uuid4().hex}')
        staging.mkdir()
        try:
            capture = json.loads((sources[0] / 'capture.json').read_text())
            capture['repo_id'] = 'local/orca-bottle-act-20261108'
            atomic_json(staging / 'capture.json', capture)
            for path in sources[0].glob('collection_*'):
                if path.is_file():
                    shutil.copyfile(path, staging / path.name)
            for record in records:
                original = Path(record['source']) / f"episode-{record['source_index']:06d}"
                target = staging / f"episode-{record['index']:06d}"
                target.mkdir()
                episode = json.loads((original / 'episode.json').read_text())
                episode['index'] = record['index']
                for name in ['rows.npz', *(v['file'] for v in episode['videos'].values())]:
                    if Path(name).name != name:
                        raise ValueError('Unsafe capture filename.')
                    # Immutable payloads share disk blocks; JSON has independent indices.
                    os.link(original / name, target / name)
                atomic_json(target / 'episode.json', episode)
            atomic_json(staging / 'collection_act_preparation.json', manifest)
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    logging.info('Selected %d episodes, %d frames; train=%d, held-out=%d.',
                 len(records), sum(r['frames'] for r in records),
                 len(manifest['train_episodes']), len(validation))
    last = [None]
    def progress(**fields):
        episode = fields.get('episode')
        if episode != last[0]:
            logging.info('Converting episode %s/%s', episode, fields.get('total'))
            last[0] = episode
    output = convert_capture(destination, progress)
    # The stock loader filters episodes but otherwise uses whole-dataset stats.
    # Keep held-out numeric observations/actions out of ACT's normalizer too.
    from lerobot.datasets.compute_stats import compute_episode_stats
    from lerobot.datasets.utils import load_stats, write_stats

    stats = load_stats(output)
    backup = output / 'meta' / 'stats_all_episodes.json'
    if not backup.exists():
        shutil.copyfile(output / 'meta' / 'stats.json', backup)
    numeric = {}
    for key in ('observation.state', 'action'):
        arrays = []
        for index in manifest['train_episodes']:
            with np.load(destination / f'episode-{index:06d}' / 'rows.npz', allow_pickle=False) as rows:
                arrays.append(rows[key].copy())
        numeric[key] = np.concatenate(arrays)
    stats.update(compute_episode_stats(numeric, reference['capture']['features']))
    write_stats(stats, output)
    atomic_json(output / 'collection_act_split.json', {
        'train_episodes': manifest['train_episodes'], 'validation_episodes': validation,
        'numeric_normalization_episodes': manifest['train_episodes'],
        'visual_normalization': 'Fixed ImageNet statistics (dataset.use_imagenet_stats=true)',
    })
    logging.info('Local LeRobot dataset ready: %s', output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True, help='JSON list of capture-session paths.')
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s | %(message)s')
    project = Path(__file__).resolve().parents[2]
    sources = [project / path for path in json.loads(args.sources.read_text())]
    prepare(sources, args.destination)


if __name__ == '__main__':
    main()
