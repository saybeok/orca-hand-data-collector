"""Append audited bottle demonstrations while retaining published collection versions."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import uuid

import numpy as np

from orca_teleop.collection_control import atomic_json
from orca_teleop.deferred_recording import capture_fingerprint, convert_capture

PROJECT = Path(__file__).resolve().parents[1]
PUBLIC_FOLDERS = ('data', 'meta', 'videos', 'robot_configuration')
PUBLIC_FILES = ('README.md', 'collection_manifest.json', 'robot_metadata.json')


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def public_files(root):
    paths = {str(p.relative_to(root)): p for folder in PUBLIC_FOLDERS
             for p in (root / folder).rglob('*') if p.is_file()}
    paths.update({name: root / name for name in PUBLIC_FILES})
    return paths


def validate_extension(manifest, previous, parent):
    count = len(previous['episodes'])
    if len(manifest['episodes']) <= count or manifest['episodes'][:count] != previous['episodes']:
        raise ValueError('Extension must preserve every previous episode in its original order.')
    if manifest.get('parent_collection') != {
        'repo_id': parent['repo_id'], 'commit': parent['commit'],
        'tag': parent['tag'], 'episodes': count,
    }:
        raise ValueError('Extension has different parent provenance.')


def verify_files(root, expected):
    for name, wanted in expected.items():
        if sha256(root / name) != wanted:
            raise ValueError(f'Prepared public file changed: {name}')


def verify_parent(root, parent):
    for file in parent['files']:
        if file['path'] == '.gitattributes':
            continue
        path = root / file['path']
        if not path.is_file() or path.stat().st_size != file['size']:
            raise ValueError(f'Published parent file missing or changed: {path}')
        if file['sha256']:
            actual = sha256(path)
            expected = file['sha256']
        else:
            digest = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
            with path.open('rb') as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            actual, expected = digest.hexdigest(), file['blob_id']
        if actual != expected:
            raise ValueError(f'Parent does not match published content: {path}')


def parent_identities(previous, source_base):
    if 'episode_identity_sha256' in previous:
        return previous['episode_identity_sha256']
    result = []
    for record in previous['episodes']:
        episode = read(source_base / record['source_session'] /
                       f"episode-{record['source_episode']:06d}/episode.json")
        identity = (episode['rows_sha256'], tuple(sorted(
            (key, value['sha256']) for key, value in episode['videos'].items())))
        result.append(hashlib.sha256(json.dumps(identity).encode()).hexdigest())
    return result


def prepare(previous_root, audit_path, parent_path, output):
    parent, audit = read(parent_path), read(audit_path)
    if audit['issues']:
        raise ValueError('New recordings failed the checksum audit; cannot publish.')
    previous = read(previous_root / 'collection_manifest.json')
    info = read(previous_root / 'meta/info.json')
    if info['total_episodes'] != parent['episodes']:
        raise ValueError('Local and remote parent episode counts differ.')
    logging.info('Verifying local parent against public file hashes.')
    verify_parent(previous_root, parent)
    identities = parent_identities(previous, Path('/media/lee/My Book/orcahand_data'))
    if len(identities) != len(previous['episodes']) or len(set(identities)) != len(identities):
        raise ValueError('Invalid or duplicate parent episode identities.')
    records = list(previous['episodes'])
    seen = set(identities)
    source_sessions = list(previous['source_sessions'])
    for session in audit['sessions']:
        source = Path(session['path'])
        if capture_fingerprint(source) != session['fingerprint']:
            raise ValueError(f'Frozen new source changed: {source}')
        for episode in session['episodes']:
            identity = episode['identity_sha256']
            if identity in seen:
                raise ValueError('New episode duplicates a previously selected demonstration.')
            seen.add(identity)
            identities.append(identity)
            records.append({'index': len(records), 'source_session': session['name'],
                            'source_episode': episode['source_episode'], 'frames': episode['frames']})
        source_sessions.append({'name': session['name'], 'episodes': len(session['episodes']),
                                'fingerprint': session['fingerprint']})
    count = len(records)
    if count != parent['episodes'] + audit['episodes']:
        raise ValueError('Extension episode count mismatch.')
    match = re.fullmatch(r'collection-v(\d+)-\d+', parent['tag'])
    if not match:
        raise ValueError('Unknown parent collection tag.')
    tag = f'collection-v{int(match[1]) + 1}-{count}'
    manifest = {
        **previous, 'episodes': records, 'source_sessions': source_sessions,
        'parent_collection': {'repo_id': parent['repo_id'], 'commit': parent['commit'],
                              'tag': parent['tag'], 'episodes': parent['episodes']},
        'episode_identity_sha256': identities,
        'selection_policy': {
            'parent_episodes': parent['episodes'], 'parent_minimum_seconds': 20,
            'added_episodes': audit['episodes'], 'added_minimum_seconds': 0,
            'reason': 'Operator explicitly requested all 90 episodes from the three latest sessions, including 13 under 20 seconds.',
        },
        'minimum_episode_seconds': 0,
        'normalization_scope': f'All {count} published episodes; refit on a chosen training split for held-out evaluation.',
    }
    validate_extension(manifest, previous, parent)
    if output.exists():
        if read(output / 'collection_manifest.json') != manifest:
            raise ValueError('Existing extension output has different provenance.')
        verify_files(output, read(output / 'preparation_summary.json')['file_sha256'])
        logging.info('Reusing verified extension: %s', output)
        return
    converted = [previous_root]
    for session in audit['sessions']:
        last = [None]
        def progress(**fields):
            if fields.get('episode') != last[0]:
                last[0] = fields.get('episode')
                logging.info('Encoding %s episode %s/%s', session['name'], last[0], fields.get('total'))
        converted.append(convert_capture(Path(session['path']), progress))
        if capture_fingerprint(Path(session['path'])) != session['fingerprint']:
            raise ValueError('New source changed during conversion.')
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.compute_stats import compute_episode_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import load_stats, write_stats
    import av
    import pyarrow.parquet as pq

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name('.' + output.name + '-preparing-' + uuid.uuid4().hex)
    try:
        aggregate_datasets(repo_ids=['local/previous', *[
            read(Path(s['path']) / 'capture.json')['repo_id'] for s in audit['sessions']]],
            roots=converted, aggr_repo_id='local/orca-hand-bottle', aggr_root=staging,
            video_files_size_in_mb=1)
        # Keep every previously published video byte-identical; only metadata/numeric tables expand.
        for video in (previous_root / 'videos').rglob('*.mp4'):
            if sha256(video) != sha256(staging / video.relative_to(previous_root)):
                raise ValueError('Aggregation changed a previously published video.')
        table = pq.read_table(sorted((staging / 'data').rglob('*.parquet')),
                              columns=['observation.state', 'action'])
        numeric = {key: np.asarray(table[key].to_pylist()) for key in ['observation.state', 'action']}
        frames = sum(record['frames'] for record in records)
        if any(value.shape != (frames, 20) or not np.isfinite(value).all() for value in numeric.values()):
            raise ValueError('Merged 20-DoF values are missing or invalid.')
        stats = load_stats(staging)
        stats.update(compute_episode_stats(numeric, info['features']))
        write_stats(stats, staging)
        atomic_json(staging / 'collection_manifest.json', manifest)
        shutil.copyfile(previous_root / 'robot_metadata.json', staging / 'robot_metadata.json')
        shutil.copytree(previous_root / 'robot_configuration', staging / 'robot_configuration')
        card = (previous_root / 'README.md').read_text()
        card = card.replace(f"{parent['episodes']} operator-saved", f'{count} operator-saved')
        card = card.replace(f"{info['total_frames']:,} synchronized", f'{frames:,} synchronized')
        card = card.replace(f"all {parent['episodes']} episodes", f'all {count} episodes')
        card = card.replace('Episodes shorter than 20 seconds\nwere excluded.',
            f'The initial {parent["episodes"]} episodes excluded recordings shorter than 20 seconds.\n'
            f'This revision appends all {audit["episodes"]} operator-selected recordings from three\n'
            'new sessions, including 13 shorter than 20 seconds, as explicitly requested by the operator.')
        card += (f'\n## Collection revision\n\n`{tag}` contains {count} episodes and {frames:,} frames.\n'
                 f'The first {parent["episodes"]} episodes retain their original indices and video bytes.\n'
                 f'The previous `{parent["tag"]}` snapshot remains available.\n')
        (staging / 'README.md').write_text(card)
        ds = LeRobotDataset('local/orca-hand-bottle', root=staging, video_backend='pyav')
        if ds.num_episodes != count or ds.num_frames != frames:
            raise ValueError('Merged episode/frame count mismatch.')
        if {int(i) for i in ds.hf_dataset['episode_index']} != set(range(count)):
            raise ValueError('Merged episode indices are not contiguous.')
        if any(int(ds.meta.episodes[i]['length']) != record['frames'] for i, record in enumerate(records)):
            raise ValueError('Merged episode duration mismatch.')
        offset = 0
        for root in converted:
            original = LeRobotDataset('local/source', root=root, video_backend='pyav')
            for index in sorted({0, original.num_episodes - 1}):
                a = original[int(original.meta.episodes[index]['dataset_from_index'])]
                b = ds[int(ds.meta.episodes[offset + index]['dataset_from_index'])]
                for key in ['observation.state', 'action', *ds.meta.video_keys]:
                    np.testing.assert_array_equal(a[key].numpy(), b[key].numpy())
            offset += original.num_episodes
        encoded_frames = 0
        for video in (staging / 'videos').rglob('*.mp4'):
            with av.open(str(video)) as container:
                n = container.streams.video[0].frames
                if n <= 0:
                    raise ValueError('Video frame count missing.')
                encoded_frames += n
        if encoded_frames != frames * 3:
            raise ValueError('Video streams contain missing or unindexed footage.')
        files = {name: sha256(path) for name, path in public_files(staging).items()}
        atomic_json(staging / 'preparation_summary.json', {
            'episodes': count, 'added_episodes': audit['episodes'], 'frames': frames,
            'fps': 30, 'strict_video_selection': True, 'encoded_video_frames': encoded_frames,
            'collection_tag': tag, 'file_sha256': files,
        })
        staging.rename(output)
        logging.info('PUBLIC EXTENSION READY: %s | %d episodes, %d frames', output, count, frames)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def publish(root, parent_path):
    from huggingface_hub import HfApi, hf_hub_download
    parent = read(parent_path)
    manifest, info, stats, summary = [read(root / p) for p in (
        'collection_manifest.json', 'meta/info.json', 'meta/stats.json', 'preparation_summary.json')]
    count = info['total_episodes']
    if not summary['strict_video_selection'] or summary['encoded_video_frames'] != info['total_frames'] * 3:
        raise ValueError('Public video selection has not been verified.')
    if count != summary['episodes'] or count != len(manifest['episodes']):
        raise ValueError('Prepared episode counts disagree.')
    for key in ['observation.state', 'action']:
        if info['features'][key]['shape'] != [20]:
            raise ValueError('Unexpected robot state/action dimension.')
        for q in ['q01', 'q99']:
            if np.asarray(stats[key][q]).shape != (20,) or not np.isfinite(stats[key][q]).all():
                raise ValueError('Pi05 quantile statistics are missing.')
    verify_files(root, summary['file_sha256'])
    (PROJECT / 'data/act').mkdir(parents=True, exist_ok=True)
    api = HfApi()
    repo_id = parent['repo_id']
    if api.whoami()['name'] != repo_id.split('/')[0]:
        raise ValueError('Logged-in account differs from the approved repository owner.')
    cache = PROJECT / 'data/act/.hub_verification_cache'
    previous = read(hf_hub_download(repo_id, 'collection_manifest.json', repo_type='dataset',
                                   revision=parent['commit'], cache_dir=cache))
    validate_extension(manifest, previous, parent)
    head = api.dataset_info(repo_id)
    refs = {tag.name: tag.target_commit for tag in api.list_repo_refs(repo_id, repo_type='dataset').tags}
    tag = summary['collection_tag']
    if any(refs.get(name) != target for name, target in parent['tags'].items() if name != 'v3.0'):
        raise ValueError('An existing collection tag changed; reconcile before publishing.')
    if head.sha == parent['commit']:
        if tag in refs:
            raise ValueError('New immutable tag already exists on a different publication.')
        expected = set(public_files(root))
        existing = set(api.list_repo_files(repo_id, repo_type='dataset', revision=head.sha)) - {'.gitattributes'}
        if not existing <= expected:
            raise ValueError('Publication would leave stale, unindexed remote files.')
        commit = api.upload_folder(repo_id=repo_id, repo_type='dataset', folder_path=root,
            parent_commit=parent['commit'],
            commit_message=f'Append {summary["added_episodes"]} bottle demonstrations ({count} total)',
            commit_description='Preserve previous collection snapshot and episode order; all 90 new operator-selected episodes included; refresh full-collection quantile statistics.',
            allow_patterns=[*PUBLIC_FILES, *[folder + '/**' for folder in PUBLIC_FOLDERS]])
        revision = commit.oid
    else:
        current = read(hf_hub_download(repo_id, 'collection_manifest.json', repo_type='dataset',
                                      revision=head.sha, cache_dir=cache))
        if current != manifest:
            raise ValueError('Remote parent changed; refusing to overwrite another collection update.')
        revision = head.sha  # Resume final verification/tagging after an already completed upload.
    if tag in refs and refs[tag] != revision:
        raise ValueError('Immutable collection tag points to a different revision.')
    if 'v3.0' in refs and refs['v3.0'] != revision:
        api.delete_tag(repo_id, tag='v3.0', repo_type='dataset')
    api.create_tag(repo_id, tag='v3.0', revision=revision, repo_type='dataset', exist_ok=True)
    api.create_tag(repo_id, tag=tag, revision=revision, repo_type='dataset', exist_ok=True)
    remote = HfApi(token=False).dataset_info(repo_id, revision=revision, files_metadata=True)
    if remote.private:
        raise ValueError('Uploaded dataset is not public.')
    remote_files = {file.rfilename: file for file in remote.siblings}
    expected = public_files(root)
    if set(remote_files) - {'.gitattributes'} != set(expected):
        raise ValueError('Remote file selection differs from the verified local dataset.')
    for name, path in expected.items():
        file = remote_files[name]
        if file.lfs:
            if file.lfs.sha256 != summary['file_sha256'][name]:
                raise ValueError(f'Remote file hash mismatch: {name}')
        else:
            digest = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
            digest.update(path.read_bytes())
            if digest.hexdigest() != file.blob_id:
                raise ValueError(f'Remote metadata hash mismatch: {name}')
    final_refs = {t.name: t.target_commit for t in api.list_repo_refs(repo_id, repo_type='dataset').tags}
    if any(final_refs.get(name) != target for name, target in parent['tags'].items() if name != 'v3.0'):
        raise ValueError('Prior collection tags were not preserved.')
    receipt = {'repo_id': repo_id, 'url': f'https://huggingface.co/datasets/{repo_id}',
               'private': False, 'commit': revision, 'parent_commit': parent['commit'],
               'episodes': count, 'added_episodes': summary['added_episodes'], 'frames': info['total_frames'],
               'dataset_tag': tag, 'lerobot_version_tag': 'v3.0',
               'anonymous_public_access_verified': True, 'remote_content_hashes_verified': len(expected),
               'previous_collection_tags_preserved': True}
    atomic_json(PROJECT / f'data/act/public_bottle_{count}_upload_receipt.json', receipt)
    atomic_json(PROJECT / 'data/act/publication_status.json', {'status': 'complete', **receipt})
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'publish'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--previous', type=Path,
        default=Path('/media/lee/My Book/orcahand_lerobot/orca-hand-bottle-v1-75'))
    parser.add_argument('--audit', type=Path, default=PROJECT / 'data/act/publication_latest3_audit.json')
    parser.add_argument('--parent', type=Path, default=PROJECT / 'data/act/publication_extension_parent.json')
    args = parser.parse_args()
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s | %(message)s')
    if args.operation == 'prepare':
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ.setdefault('HF_DATASETS_CACHE', str(PROJECT / 'data/act/.hf_cache'))
        prepare(args.previous, args.audit, args.parent, args.output)
    else:
        os.environ.pop('HF_HUB_OFFLINE', None)
        publish(args.output, args.parent)


if __name__ == '__main__':
    main()
