import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    'public_bottle_extension', Path(__file__).parents[2] / 'scripts/extend_public_bottle_dataset.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_extension_preserves_previous_episode_order():
    parent = {'repo_id': 'owner/bottle', 'commit': 'old', 'tag': 'collection-v1-1'}
    previous = {'episodes': [{'index': 0, 'source_episode': 7}]}
    manifest = {'episodes': [{'index': 0, 'source_episode': 8}, {'index': 1}],
                'parent_collection': {**parent, 'episodes': 1}}
    with pytest.raises(ValueError, match='original order'):
        module.validate_extension(manifest, previous, parent)


@pytest.fixture
def publication(tmp_path, monkeypatch):
    root = tmp_path / 'dataset'
    root.mkdir()
    for folder in module.PUBLIC_FOLDERS:
        (root / folder).mkdir()
    previous = {'episodes': [{'index': 0, 'frames': 30}]}
    parent = {'repo_id': 'owner/bottle', 'commit': 'old', 'tag': 'collection-v1-1',
              'episodes': 1, 'tags': {'collection-v1-1': 'old', 'v3.0': 'old'}}
    manifest = {'episodes': [*previous['episodes'], {'index': 1, 'frames': 40}],
                'parent_collection': {k: parent[k] for k in ['repo_id', 'commit', 'tag', 'episodes']}}
    info = {'total_episodes': 2, 'total_frames': 70,
            'features': {key: {'shape': [20]} for key in ['action', 'observation.state']}}
    payloads = {'collection_manifest.json': manifest, 'meta/info.json': info,
                'robot_metadata.json': {}, 'meta/stats.json': {
                    key: {'q01': [0] * 20, 'q99': [1] * 20} for key in info['features']}}
    for name, payload in payloads.items():
        (root / name).write_text(json.dumps(payload))
    (root / 'README.md').write_text('Public dataset')
    (root / 'data/rows.parquet').write_bytes(b'example-numeric-table')
    (root / 'private.frames').write_bytes(b'never publish raw capture')
    summary = {'episodes': 2, 'added_episodes': 1, 'strict_video_selection': True,
               'encoded_video_frames': 210, 'collection_tag': 'collection-v2-2',
               'file_sha256': {name: module.sha256(path) for name, path in module.public_files(root).items()}}
    (root / 'preparation_summary.json').write_text(json.dumps(summary))
    parent_path, previous_path = tmp_path / 'parent.json', tmp_path / 'previous.json'
    parent_path.write_text(json.dumps(parent))
    previous_path.write_text(json.dumps(previous))
    calls = {}
    tags = dict(parent['tags'])

    class API:
        head = 'old'
        def whoami(self):
            return {'name': 'owner'}
        def dataset_info(self, repo, **kwargs):
            if not kwargs:
                return SimpleNamespace(sha=self.head)
            siblings = []
            for name, path in module.public_files(root).items():
                blob = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode() + path.read_bytes()).hexdigest()
                siblings.append(SimpleNamespace(rfilename=name, lfs=None, blob_id=blob))
            return SimpleNamespace(private=False, siblings=siblings)
        def list_repo_refs(self, *args, **kwargs):
            return SimpleNamespace(tags=[SimpleNamespace(name=n, target_commit=c) for n, c in tags.items()])
        def list_repo_files(self, *args, **kwargs):
            return ['.gitattributes', *module.PUBLIC_FILES, 'meta/info.json', 'meta/stats.json']
        def upload_folder(self, **kwargs):
            calls['upload'] = kwargs
            return SimpleNamespace(oid='new')
        def delete_tag(self, repo, *, tag, **kwargs):
            tags.pop(tag)
        def create_tag(self, repo, *, tag, revision, **kwargs):
            tags[tag] = revision
    api = API()
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda **kwargs: api)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download',
                        lambda *args, **kwargs: str(previous_path))
    monkeypatch.setattr(module, 'PROJECT', tmp_path)
    return root, parent_path, api, calls, tags


def test_append_uses_atomic_parent_commit_and_preserves_old_tag(publication, tmp_path):
    root, parent_path, api, calls, tags = publication
    module.publish(root, parent_path)
    assert calls['upload']['parent_commit'] == 'old'
    assert 'private.frames' not in calls['upload']['allow_patterns']
    assert tags == {'collection-v1-1': 'old', 'collection-v2-2': 'new', 'v3.0': 'new'}
    receipt = json.loads((tmp_path / 'data/act/public_bottle_2_upload_receipt.json').read_text())
    assert receipt['episodes'] == 2 and receipt['added_episodes'] == 1
    assert receipt['anonymous_public_access_verified']


def test_local_file_tampering_rejected_before_publication(publication):
    root, parent_path, api, calls, tags = publication
    (root / 'data/rows.parquet').write_bytes(b'modified')
    with pytest.raises(ValueError, match='Prepared public file changed'):
        module.publish(root, parent_path)
    assert 'upload' not in calls


def test_concurrent_remote_update_is_not_overwritten(publication):
    root, parent_path, api, calls, tags = publication
    api.head = 'someone-elses-new-revision'
    with pytest.raises(ValueError, match='Remote parent changed'):
        module.publish(root, parent_path)
    assert 'upload' not in calls
    assert tags['collection-v1-1'] == 'old'


def test_existing_immutable_release_tag_is_not_replaced(publication):
    root, parent_path, api, calls, tags = publication
    tags['collection-v2-2'] = 'different-commit'
    with pytest.raises(ValueError, match='immutable tag already exists'):
        module.publish(root, parent_path)
    assert 'upload' not in calls
