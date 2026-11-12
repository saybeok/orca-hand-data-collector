import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    'public_bottle_upload', Path(__file__).parents[2] / 'scripts/upload_public_bottle_dataset.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def dataset(root):
    root.mkdir()
    for folder in ['meta', 'data', 'videos', 'robot_configuration']:
        (root / folder).mkdir()
    info = {'total_episodes': 75, 'total_frames': 70000,
            'features': {'action': {'shape': [20]}, 'observation.state': {'shape': [20]}}}
    (root / 'meta/info.json').write_text(json.dumps(info))
    (root / 'meta/stats.json').write_text(json.dumps({k: {'q01': [0] * 20, 'q99': [1] * 20}
                                                   for k in ['action', 'observation.state']}))
    (root / 'collection_manifest.json').write_text(json.dumps({'episodes': list(range(75))}))
    (root / 'preparation_summary.json').write_text(json.dumps(
        {'strict_video_selection': True, 'encoded_video_frames': 210000}))
    (root / 'README.md').write_text('Dataset card')
    (root / 'robot_metadata.json').write_text('{}')
    (root / 'private_raw.frames').write_bytes(b'must not be uploaded')
    return root


def test_rejects_unindexed_rejected_footage_before_any_network(tmp_path, monkeypatch):
    root = dataset(tmp_path / 'dataset')
    (root / 'preparation_summary.json').write_text(json.dumps(
        {'strict_video_selection': False, 'encoded_video_frames': 211728}))
    monkeypatch.setattr(module, 'HfApi', lambda: pytest.fail('No network before validation'))
    with pytest.raises(ValueError, match='video selection'):
        module.publish(root, 'owner/orca-hand-bottle')


def test_rejects_wrong_account_namespace(tmp_path, monkeypatch):
    root = dataset(tmp_path / 'dataset')
    monkeypatch.setattr(module, 'HfApi', lambda: SimpleNamespace(whoami=lambda: {'name': 'owner'}))
    with pytest.raises(ValueError, match='logged-in user namespace'):
        module.publish(root, 'someone-else/orca-hand-bottle')


def test_public_upload_allowlist_and_remote_verification(tmp_path, monkeypatch):
    root = dataset(tmp_path / 'dataset')
    calls = {}
    class API:
        def whoami(self):
            return {'name': 'owner'}
        def dataset_info(self, repo, **kwargs):
            if not kwargs:
                raise module.RepositoryNotFoundError('not created')
            return SimpleNamespace(private=False)
        def create_repo(self, repo, **kwargs):
            calls['create'] = kwargs
        def update_repo_visibility(self, repo, **kwargs):
            calls['visibility'] = kwargs
        def upload_folder(self, **kwargs):
            calls['upload'] = kwargs
            return SimpleNamespace(oid='test-commit')
        def list_repo_refs(self, *args, **kwargs):
            return SimpleNamespace(tags=[])
        def create_tag(self, *args, **kwargs):
            calls.setdefault('tags', []).append(kwargs['tag'])
        def list_repo_files(self, *args, **kwargs):
            return [str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()]
    monkeypatch.setattr(module, 'HfApi', API)
    monkeypatch.setattr(module, 'hf_hub_download', lambda *a, **k: str(root / 'meta/info.json'))
    monkeypatch.setattr(module, 'PROJECT', tmp_path)
    module.publish(root, 'owner/orca-hand-bottle')
    assert calls['create']['private'] is False
    assert calls['visibility']['private'] is False
    assert 'private_raw.frames' not in calls['upload']['allow_patterns']
    assert 'data/**' in calls['upload']['allow_patterns']
    assert set(calls['tags']) == {'v3.0', 'collection-v1-75'}
    receipt = json.loads((tmp_path / 'data/act/public_bottle_75_upload_receipt.json').read_text())
    assert receipt['commit'] == 'test-commit' and receipt['episodes'] == 75
