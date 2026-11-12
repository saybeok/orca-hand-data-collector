"""Exclude short saved captures, preserving completed training artifacts."""
import argparse
import json
import os
import shutil
import uuid
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minimum-seconds', type=float, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.minimum_seconds) or args.minimum_seconds <= 0:
        parser.error('Minimum duration must be positive and finite.')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ.setdefault('HF_DATASETS_CACHE', str(args.output.resolve().parent / '.hf_cache'))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.dataset_tools import _copy_and_reindex_data, _copy_and_reindex_episodes_metadata
    from lerobot.datasets.compute_stats import compute_episode_stats
    from lerobot.datasets.utils import load_stats, write_stats
    from orca_teleop.collection_control import atomic_json
    from orca_teleop.deferred_recording import capture_fingerprint

    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    capture = json.loads((source / 'capture.json').read_text())
    metadata = [json.loads(p.read_text()) for p in sorted(source.glob('episode-*/episode.json'))]
    if [e['index'] for e in metadata] != list(range(len(metadata))):
        raise ValueError('Source indices must be contiguous.')
    excluded = [e['index'] for e in metadata if e['frames'] / capture['fps'] < args.minimum_seconds]
    kept = [e['index'] for e in metadata if e['index'] not in excluded]
    if not excluded or not kept:
        raise ValueError('No exclusions, or no episodes would remain.')
    mapping = {old: new for new, old in enumerate(kept)}
    source_hash = capture_fingerprint(source)
    staging = output.with_name('.' + output.name + '-curating-' + uuid.uuid4().hex)
    staging.mkdir(parents=True)
    try:
        root = staging / 'capture'
        root.mkdir()
        repo_id = capture['repo_id'] + f'-min{args.minimum_seconds:g}'
        atomic_json(root / 'capture.json', dict(capture, repo_id=repo_id))
        for p in source.glob('collection_*'):
            if p.is_file():
                shutil.copyfile(p, root / p.name)
        for old, new in mapping.items():
            original = source / f'episode-{old:06d}'
            target = root / f'episode-{new:06d}'
            target.mkdir()
            episode = dict(metadata[old], index=new)
            for name in ['rows.npz', *(v['file'] for v in episode['videos'].values())]:
                if Path(name).name != name:
                    raise ValueError('Unsafe filename.')
                os.link(original / name, target / name)
            atomic_json(target / 'episode.json', episode)
        split = json.loads((source / 'lerobot/collection_act_split.json').read_text())
        for key in ['train_episodes', 'validation_episodes', 'numeric_normalization_episodes']:
            split[key] = [mapping[i] for i in split[key] if i in mapping]
        if not split['train_episodes']:
            raise ValueError('No training episodes remain.')
        manifest = json.loads((root / 'collection_act_preparation.json').read_text())
        manifest['episodes'] = [dict(e, index=mapping[e['index']], previous_combined_index=e['index'])
                                for e in manifest['episodes'] if e['index'] in mapping]
        for group in manifest['sessions']:
            group['episodes'] = [mapping[i] for i in group['episodes'] if i in mapping]
        manifest.update(train_episodes=split['train_episodes'], validation_episodes=split['validation_episodes'])
        atomic_json(root / 'collection_act_preparation.json', manifest)
        src = LeRobotDataset(capture['repo_id'], root=source / 'lerobot', video_backend='pyav')
        dst = LeRobotDatasetMetadata.create(repo_id=repo_id, fps=src.meta.fps,
            features=src.meta.features, robot_type=src.meta.robot_type, root=root / 'lerobot', use_videos=True)
        videos = {}
        for old, new in mapping.items():
            ep = src.meta.episodes[old]
            videos[new] = {k: v for k, v in ep.items() if k.startswith('videos/')}
            for key in src.meta.video_keys:
                name = src.meta.video_path.format(video_key=key,
                    chunk_index=ep[f'videos/{key}/chunk_index'], file_index=ep[f'videos/{key}/file_index'])
                target = dst.root / name
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.link(src.root / name, target)
        data = _copy_and_reindex_data(src, dst, mapping)
        _copy_and_reindex_episodes_metadata(src, dst, mapping, data, videos)
        shutil.copyfile(dst.root / 'meta/stats.json', dst.root / 'meta/stats_all_episodes.json')
        stats = load_stats(dst.root)
        numeric = {}
        for key in ['observation.state', 'action']:
            values = []
            for index in split['train_episodes']:
                with np.load(root / f'episode-{index:06d}/rows.npz', allow_pickle=False) as rows:
                    values.append(rows[key].copy())
            numeric[key] = np.concatenate(values)
        stats.update(compute_episode_stats(numeric, capture['features']))
        write_stats(stats, dst.root)
        for p in root.glob('collection_*'):
            if p.is_file():
                shutil.copyfile(p, dst.root / p.name)
        atomic_json(dst.root / 'collection_act_split.json', split)
        total_frames = sum(metadata[i]['frames'] for i in kept)
        atomic_json(dst.root / 'conversion.json', {'source_sha256': capture_fingerprint(root),
            'episodes': len(kept), 'frames': total_frames, 'source_encoding': 'JPEG quality 95',
            'method': 'Filtered data/metadata; video streams shared without re-encoding'})
        report = {'minimum_duration_seconds': args.minimum_seconds,
            'source_capture': str(source), 'source_fingerprint': source_hash,
            'active_capture': str(output / 'capture'), 'active_dataset': str(output / 'capture/lerobot'),
            'kept_episodes': len(kept), 'kept_frames': total_frames,
            'excluded': [{'combined_index': i, 'seconds': metadata[i]['frames'] / capture['fps'],
                'reason': 'short_duration', 'recovery_path': str(source / f'episode-{i:06d}')} for i in excluded],
            'train_episodes': len(split['train_episodes']), 'validation_episodes': len(split['validation_episodes']),
            'discarded_retries': 'Already discarded by the recorder; none present in saved captures.',
            'note': 'Original dataset kept for completed checkpoint reproducibility. No retraining/upload.'}
        atomic_json(root / 'collection_curation.json', report)
        atomic_json(dst.root / 'collection_curation.json', report)
        verify = LeRobotDataset(repo_id, root=dst.root, video_backend='pyav')
        assert verify.num_episodes == len(kept) and verify.num_frames == total_frames
        assert {int(i) for i in verify.hf_dataset['episode_index']} == set(range(len(kept)))
        assert all(verify.meta.episodes[i]['length'] / capture['fps'] >= args.minimum_seconds for i in range(len(kept)))
        # Check both sides of each removed episode for shifted video/row alignment.
        for old in sorted({i for x in excluded for i in (x - 1, x + 1)} & set(kept)):
            a = src[int(src.meta.episodes[old]['dataset_from_index'])]
            b = verify[int(verify.meta.episodes[mapping[old]]['dataset_from_index'])]
            for key in ['observation.state', 'action', *src.meta.video_keys]:
                np.testing.assert_array_equal(a[key].numpy(), b[key].numpy())
        assert capture_fingerprint(source) == source_hash
        staging.rename(output)
        atomic_json(output.parent / 'active_dataset.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == '__main__':
    main()
