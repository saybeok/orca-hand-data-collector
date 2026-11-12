"""Publish the prepared 75-episode LeRobot dataset and verify its remote metadata."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError, EntryNotFoundError

from orca_teleop.collection_control import atomic_json

PROJECT = Path(__file__).resolve().parents[1]


def publish(root: Path, repo_id: str):
    info = json.loads((root / 'meta/info.json').read_text())
    manifest = json.loads((root / 'collection_manifest.json').read_text())
    stats = json.loads((root / 'meta/stats.json').read_text())
    summary = json.loads((root / 'preparation_summary.json').read_text())
    if not summary.get('strict_video_selection') or summary['encoded_video_frames'] != info['total_frames'] * 3:
        raise ValueError('Public video selection has not been verified.')
    if info['total_episodes'] != 75 or len(manifest['episodes']) != 75:
        raise ValueError('Expected the frozen 75-episode dataset.')
    for key in ['observation.state', 'action']:
        if info['features'][key]['shape'] != [20]:
            raise ValueError('Unexpected robot state/action dimensions.')
        for q in ['q01', 'q99']:
            if np.asarray(stats[key][q]).shape != (20,) or not np.isfinite(stats[key][q]).all():
                raise ValueError('Pi05 quantile stats are missing.')
    (PROJECT / 'data/act').mkdir(parents=True, exist_ok=True)
    api = HfApi()
    account = api.whoami()['name']
    if repo_id.split('/')[0] != account:
        raise ValueError(f'Expected the logged-in user namespace: {account}')
    try:
        api.dataset_info(repo_id)
        files = set(api.list_repo_files(repo_id, repo_type='dataset'))
        if files - {'.gitattributes', 'README.md'}:
            existing = hf_hub_download(repo_id, 'collection_manifest.json', repo_type='dataset',
                cache_dir=PROJECT / 'data/act/.hub_verification_cache')
            if json.loads(Path(existing).read_text()) != manifest:
                raise ValueError('Existing repository has different data; refusing overwrite.')
    except RepositoryNotFoundError:
        api.create_repo(repo_id, repo_type='dataset', private=False)
    except EntryNotFoundError as exc:
        raise ValueError('Existing repository has no matching provenance; refusing overwrite.') from exc
    api.update_repo_visibility(repo_id, repo_type='dataset', private=False)
    commit = api.upload_folder(
        repo_id=repo_id, repo_type='dataset', folder_path=root,
        commit_message='Publish 75 curated AM-ARM200 + ORCA bottle demonstrations',
        commit_description='30 Hz, 3 RGB views, 20-DoF radians; episodes under 20 seconds excluded; numerical q01/q99 included.',
        allow_patterns=['README.md', 'data/**', 'meta/**', 'videos/**',
                        'collection_manifest.json', 'robot_metadata.json', 'robot_configuration/**'],
    )
    refs = api.list_repo_refs(repo_id, repo_type='dataset')
    tags = {tag.name: tag.target_commit for tag in refs.tags}
    if 'v3.0' in tags and tags['v3.0'] != commit.oid:
        api.delete_tag(repo_id, tag='v3.0', repo_type='dataset')
    api.create_tag(repo_id, tag='v3.0', revision=commit.oid, repo_type='dataset', exist_ok=True)
    if 'collection-v1-75' in tags and tags['collection-v1-75'] != commit.oid:
        raise ValueError('Immutable collection tag already points to another commit; inspect before replacing.')
    api.create_tag(repo_id, tag='collection-v1-75', revision=commit.oid, repo_type='dataset', exist_ok=True)
    remote_path = hf_hub_download(repo_id, 'meta/info.json', repo_type='dataset', revision=commit.oid,
                                 cache_dir=PROJECT / 'data/act/.hub_verification_cache')
    if json.loads(Path(remote_path).read_text()) != info:
        raise ValueError('Uploaded metadata does not match local dataset.')
    remote = api.dataset_info(repo_id, revision=commit.oid)
    if remote.private:
        raise ValueError('Repository is not public.')
    remote_files = set(api.list_repo_files(repo_id, repo_type='dataset', revision=commit.oid))
    expected = {str(p.relative_to(root)) for folder in ['data', 'meta', 'videos', 'robot_configuration']
                for p in (root / folder).rglob('*') if p.is_file()}
    expected.update(['README.md', 'collection_manifest.json', 'robot_metadata.json'])
    if not expected <= remote_files:
        raise ValueError(f'Missing uploaded files: {expected - remote_files}')
    receipt = {'repo_id': repo_id, 'url': f'https://huggingface.co/datasets/{repo_id}',
               'private': False, 'commit': commit.oid, 'episodes': info['total_episodes'],
               'frames': info['total_frames'], 'dataset_tag': 'collection-v1-75',
               'lerobot_version_tag': 'v3.0', 'verified_file_count': len(expected)}
    atomic_json(PROJECT / 'data/act/public_bottle_75_upload_receipt.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--repo-id', required=True)
    args = parser.parse_args()
    os.environ.pop('HF_HUB_OFFLINE', None)
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    publish(args.dataset_root.resolve(), args.repo_id)


if __name__ == '__main__':
    main()
