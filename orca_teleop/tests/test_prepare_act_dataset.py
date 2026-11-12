import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    'prepare_act_dataset', Path(__file__).parents[1] / 'scripts' / 'prepare_act_dataset.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_source(root, offset=0):
    root.mkdir()
    features = {key: {'dtype': 'float32', 'shape': [20]}
                for key in ('observation.state', 'action')}
    capture = {'format': 'orca-jpeg-capture-v1', 'fps': 30, 'features': features,
               'image_encoding': 'JPEG', 'jpeg_quality': 95, 'repo_id': 'local/source'}
    (root / 'capture.json').write_text(json.dumps(capture))
    (root / 'collection_session.json').write_text(json.dumps({
        'joint_ids': list(range(20)), 'angle_units': 'canonical_model_radians',
        'arm_reference': {'motor_home_rad': [0, 0, 0]}}))
    for name in ('collection_hand_config.yaml', 'collection_calibration.yaml',
                 'collection_arm_config.yaml', 'collection_arm_home.json'):
        (root / name).write_text('same calibration')
    for index in range(2):
        episode = root / f'episode-{index:06d}'
        episode.mkdir()
        values = np.full((3, 20), offset + index, dtype=np.float32)
        np.savez(episode / 'rows.npz', **{'observation.state': values, 'action': values,
                                        'task': np.array(['move the bottle.'] * 3)})
        (episode / 'episode.json').write_text(json.dumps({
            'index': index, 'frames': 3, 'videos': {}, 'rows_sha256': 'test'}))
    return root


def fake_conversion(root, progress):
    from lerobot.datasets.compute_stats import compute_episode_stats
    from lerobot.datasets.utils import write_stats
    output = root / 'lerobot'
    if not output.exists():
        output.mkdir()
        values = np.zeros((12, 20), dtype=np.float32)
        features = json.loads((root / 'capture.json').read_text())['features']
        write_stats(compute_episode_stats({'action': values, 'observation.state': values}, features), output)
    return output


def test_combines_without_mutating_sources_and_excludes_validation_from_stats(tmp_path, monkeypatch):
    from lerobot.datasets.utils import load_stats
    a, b = make_source(tmp_path / 'a'), make_source(tmp_path / 'b', 100)
    original = (b / 'episode-000001/episode.json').read_bytes()
    monkeypatch.setattr(module, 'convert_capture', fake_conversion)
    destination = tmp_path / 'combined'
    output = module.prepare([a, b], destination)
    manifest = json.loads((destination / 'collection_act_preparation.json').read_text())
    assert len(manifest['episodes']) == 4
    assert set(manifest['train_episodes']).isdisjoint(manifest['validation_episodes'])
    assert len(manifest['train_episodes']) == len(manifest['validation_episodes']) == 2
    assert [json.loads(p.read_text())['index'] for p in sorted(destination.glob('episode-*/episode.json'))] == list(range(4))
    assert (destination / 'episode-000003/rows.npz').stat().st_ino == (b / 'episode-000001/rows.npz').stat().st_ino
    assert (b / 'episode-000001/episode.json').read_bytes() == original
    selected = [manifest['episodes'][i] for i in manifest['train_episodes']]
    expected = np.mean([100 * (Path(r['source']) == b) + r['source_index'] for r in selected])
    stats = load_stats(output)
    np.testing.assert_allclose(stats['action']['mean'], expected)
    assert stats['action']['count'].item() == 6
    assert module.prepare([a, b], destination) == output


def test_rejects_mixed_arm_reference_before_creating_output(tmp_path):
    a, b = make_source(tmp_path / 'a'), make_source(tmp_path / 'b')
    path = b / 'collection_session.json'
    data = json.loads(path.read_text())
    data['arm_reference']['motor_home_rad'][0] = 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='arm-reference mismatch'):
        module.prepare([a, b], tmp_path / 'combined')
    assert not (tmp_path / 'combined').exists()


def test_rejects_duplicate_source(tmp_path):
    source = make_source(tmp_path / 'a')
    with pytest.raises(ValueError, match='distinct'):
        module.prepare([source, source], tmp_path / 'combined')
