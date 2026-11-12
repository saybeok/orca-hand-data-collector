"""Quarantine short raw captures and reindex saved episodes, without converting."""
import argparse
from contextlib import ExitStack
from datetime import datetime
import fcntl
import json
from pathlib import Path
import uuid

from orca_teleop.collection_control import atomic_json
from orca_teleop.deferred_recording import capture_fingerprint


def inventory(base, minimum):
    result = []
    for header in sorted(base.glob('*/capture.json')):
        root = header.parent
        config = json.loads(header.read_text())
        episodes = [json.loads(p.read_text()) for p in sorted(root.glob('episode-*/episode.json'))]
        if [e['index'] for e in episodes] != list(range(len(episodes))):
            raise ValueError(f'Noncontiguous source indices: {root}')
        excluded = [e['index'] for e in episodes if e['frames'] / config['fps'] < minimum]
        if excluded and (root / 'lerobot').exists():
            raise ValueError(f'Already converted; use derived-dataset curation: {root}')
        if list(root.glob('.attempt-*')):
            raise ValueError(f'Unfinished recording buffer; stop recording before curation: {root}')
        result.append({'root': str(root), 'fps': config['fps'], 'episodes': episodes,
                       'excluded': excluded, 'fingerprint': capture_fingerprint(root)})
    return result


def curate_session(item, backup, minimum):
    root = Path(item['root'])
    excluded = set(item['excluded'])
    metadata = item['episodes']
    mapping = {old: new for new, old in enumerate(i for i in range(len(metadata)) if i not in excluded)}
    manifest = {'minimum_seconds': minimum, 'source': str(root), 'source_fingerprint': item['fingerprint'],
                'original_episode_metadata': metadata, 'kept_old_to_new': mapping,
                'excluded': [{'old_index': i, 'seconds': metadata[i]['frames'] / item['fps'],
                              'recovery_path': str(backup / f'episode-{i:06d}')} for i in sorted(excluded)]}
    backup.mkdir(parents=True, exist_ok=False)
    atomic_json(backup / 'manifest.json', manifest)
    current = {i: root / f'episode-{i:06d}' for i in range(len(metadata))}
    tag = uuid.uuid4().hex
    try:
        for old in sorted(excluded):
            target = backup / f'episode-{old:06d}'
            current[old].rename(target)
            current[old] = target
        for old, new in mapping.items():
            if old != new:
                target = root / f'.reindex-{tag}-{old:06d}'
                current[old].rename(target)
                current[old] = target
        for old, new in mapping.items():
            target = root / f'episode-{new:06d}'
            if current[old] != target:
                current[old].rename(target)
                current[old] = target
            atomic_json(target / 'episode.json', dict(metadata[old], index=new))
        remaining = sorted(root.glob('episode-*/episode.json'))
        assert len(remaining) == len(mapping)
        for i, path in enumerate(remaining):
            episode = json.loads(path.read_text())
            assert episode['index'] == i and episode['frames'] / item['fps'] >= minimum
            for name in ['rows.npz', *(v['file'] for v in episode['videos'].values())]:
                assert (path.parent / name).is_file()
        manifest.update(state='complete', kept_episodes=len(mapping),
                        current_fingerprint=capture_fingerprint(root))
        atomic_json(backup / 'manifest.json', manifest)
        atomic_json(root / 'collection_curation.json', manifest)
        return manifest
    except BaseException:
        # Recover original names and indices even after a partial renumber.
        for old, path in current.items():
            target = root / f'.rollback-{tag}-{old:06d}'
            path.rename(target)
            current[old] = target
        for old, path in current.items():
            target = root / f'episode-{old:06d}'
            path.rename(target)
            atomic_json(target / 'episode.json', metadata[old])
        manifest['state'] = 'rolled_back'
        atomic_json(backup / 'manifest.json', manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--backup-root', type=Path, required=True)
    parser.add_argument('--minimum-seconds', type=float, default=20)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    minimum = args.minimum_seconds
    if not 0 < minimum < float('inf'):
        parser.error('Minimum duration must be positive and finite.')
    base = args.source.resolve()
    items = inventory(base, minimum)
    if not items:
        raise ValueError('No capture sessions found.')
    summary = {'source': str(base), 'minimum_seconds': minimum,
               'before': sum(len(i['episodes']) for i in items),
               'excluded': sum(len(i['excluded']) for i in items),
               'sessions': [{'name': Path(i['root']).name, 'before': len(i['episodes']),
                             'excluded': i['excluded']} for i in items]}
    summary['after'] = summary['before'] - summary['excluded']
    if args.execute and summary['excluded']:
        batch = args.backup_root.resolve() / datetime.now().strftime('%Y%m%d_%H%M%S')
        if batch.is_relative_to(base):
            raise ValueError('Backup must be outside the capture collection.')
        with ExitStack() as stack:
            for item in items:
                if item['excluded']:
                    lock = stack.enter_context((Path(item['root']) / '.conversion.lock').open('a'))
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if capture_fingerprint(Path(item['root'])) != item['fingerprint']:
                        raise ValueError('Source changed during preflight.')
            batch.mkdir(parents=True, exist_ok=False)
            atomic_json(batch / 'batch_plan.json', summary)
            for item in items:
                if item['excluded']:
                    curate_session(item, batch / Path(item['root']).name, minimum)
            verified = inventory(base, minimum)
            assert sum(len(i['episodes']) for i in verified) == summary['after']
            assert not any(i['excluded'] for i in verified)
            summary['backup'] = str(batch)
            summary['state'] = 'complete'
            atomic_json(batch / 'batch_result.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
