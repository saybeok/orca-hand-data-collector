"""Capture commits are fast and durable; conversion preserves aligned rows/video."""
import json
from pathlib import Path

import numpy as np
import pytest

from orca_teleop.deferred_recording import DeferredCapture, capture_roots, convert_capture


def make_capture(root, cameras=3):
    features = {k: {'dtype': 'float32', 'shape': (20,), 'names': None}
                for k in ('observation.state', 'action')}
    features.update({f'observation.images.cam{i}': {
        'dtype': 'video', 'shape': (64, 64, 3), 'names': ['height', 'width', 'channels'],
    } for i in range(cameras)})
    return DeferredCapture(root=root, repo_id='test/capture', fps=30, features=features)


def frame(capture, index):
    result = {'observation.state': np.full(20, index, np.float32),
              'action': np.full(20, index + .5, np.float32), 'task': 'move the eclipse case'}
    for key, ft in capture.features.items():
        if ft['dtype'] == 'video':
            # Detect RGB/BGR swaps even after JPEG and H.264 encoding.
            result[key] = np.full(ft['shape'], [200, 40 + index, 10], np.uint8)
    return result


@pytest.fixture(autouse=True)
def isolated_hf_cache(tmp_path, monkeypatch):
    from tqdm import tqdm
    monkeypatch.setattr(tqdm, 'monitor_interval', 0)
    monkeypatch.setattr('datasets.config.HF_DATASETS_CACHE', tmp_path / 'hf-cache')


def test_capture_save_discard_and_convert_multiple_episodes(tmp_path):
    import av
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    capture = make_capture(tmp_path / 'session')
    for i in range(8):
        capture.add_frame(frame(capture, i))
    capture.clear_episode_buffer()
    assert not list(capture.root.glob('.attempt-*'))
    for count in (6, 3):
        for i in range(count):
            capture.add_frame(frame(capture, i))
        capture.save_episode()
    assert capture.num_episodes == 2 and capture.num_frames == 9
    assert not list(capture.root.rglob('*.mp4'))
    assert capture_roots(tmp_path) == [capture.root]
    capture.finalize()
    output = convert_capture(capture.root)
    assert output == capture.root / 'lerobot'
    assert convert_capture(capture.root) == output  # restart/retry is idempotent
    assert len(list(capture.root.glob('episode-*'))) == 2
    for path in output.rglob('*.mp4'):
        with av.open(str(path)) as container:
            frames = list(container.decode(video=0))
        assert len(frames) == 9
        np.testing.assert_allclose([f.time for f in frames], np.arange(9)/30)
        rgb = frames[0].to_ndarray(format='rgb24').mean(axis=(0, 1))
        assert rgb[0] > 180 and rgb[2] < 25
    loaded = LeRobotDataset('test/capture', root=output, video_backend='pyav')
    assert loaded.num_episodes == 2 and loaded.num_frames == 9
    np.testing.assert_allclose(loaded[5]['action'].numpy(), 5.5)
    np.testing.assert_allclose(loaded[6]['action'].numpy(), .5)
    assert loaded[0]['task'] == 'move the eclipse case'


def test_corrupt_capture_rejects_conversion_and_preserves_source_for_retry(tmp_path):
    capture = make_capture(tmp_path / 'session', cameras=1)
    for i in range(3):
        capture.add_frame(frame(capture, i))
    capture.save_episode()
    capture.finalize()
    source = next(capture.root.glob('episode-*/*.frames'))
    original = source.read_bytes()
    source.write_bytes(original[:-15])
    with pytest.raises(ValueError, match='체크섬'):
        convert_capture(capture.root)
    assert source.exists() and not (capture.root / 'lerobot').exists()
    assert not list(capture.root.glob('.lerobot-converting-*'))
    source.write_bytes(original)
    assert convert_capture(capture.root).is_dir()


def test_failed_commit_never_counts_success_or_changes_previous_episode(tmp_path, monkeypatch):
    capture = make_capture(tmp_path / 'session', cameras=1)
    capture.add_frame(frame(capture, 0))
    capture.save_episode()
    previous = (capture.root / 'episode-000000/episode.json').read_bytes()
    capture.add_frame(frame(capture, 1))
    monkeypatch.setattr('orca_teleop.deferred_recording.os.fsync',
                        lambda fd: (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError, match='disk full'):
        capture.save_episode()
    assert capture.num_episodes == 1 and not (capture.root / 'episode-000001').exists()
    capture.finalize()
    assert previous == (capture.root / 'episode-000000/episode.json').read_bytes()
    assert not list(capture.root.glob('.attempt-*'))


def test_unfinished_trial_is_not_discovered_or_converted(tmp_path):
    capture = make_capture(tmp_path / 'session')
    capture.add_frame(frame(capture, 0))
    assert capture_roots(tmp_path) == []
    with pytest.raises(ValueError, match='성공 저장'):
        convert_capture(capture.root)
    capture.finalize()


def test_existing_conversion_is_not_overwritten_when_capture_changes(tmp_path):
    capture = make_capture(tmp_path / 'session', cameras=0)
    capture.add_frame(frame(capture, 0))
    capture.save_episode()
    capture.finalize()
    output = convert_capture(capture.root)
    manifest = capture.root / 'capture.json'
    data = json.loads(manifest.read_text())
    data['repo_id'] = 'different/capture'
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='덮어쓰지'):
        convert_capture(capture.root)
    assert (output / 'meta/info.json').exists()


def test_conversion_retry_cleans_interrupted_staging_and_rejects_concurrent_job(tmp_path):
    import fcntl
    capture = make_capture(tmp_path / 'session', cameras=0)
    capture.add_frame(frame(capture, 0))
    capture.save_episode()
    capture.finalize()
    orphan = capture.root / ('.lerobot-converting-' + 'a' * 32)
    orphan.mkdir()
    (orphan / 'incomplete').write_text('interrupted')
    with (capture.root / '.conversion.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(ValueError, match='다른 프로세스'):
            convert_capture(capture.root)
        assert orphan.exists()
    assert convert_capture(capture.root).is_dir()
    assert not orphan.exists()


def test_recorder_uses_deferred_path_and_checks_save_failure(tmp_path):
    import importlib.util
    import queue
    script = Path(__file__).resolve().parents[1] / 'scripts/record_dataset.py'
    spec = importlib.util.spec_from_file_location('deferred_test_recorder', script)
    recorder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recorder)
    capture = make_capture(tmp_path / 'session', cameras=1)
    q, results = queue.Queue(), queue.Queue()
    q.put(frame(capture, 1))
    q.put(recorder._SAVE_EPISODE)
    q.put(None)
    recorder._recorder_loop(capture, q, results)
    assert results.get_nowait() and capture.num_episodes == 1
    recorder._discard_recording_episode(capture)
    capture.finalize()


def test_app_selects_deferred_and_does_not_launch_conversion_during_teleop(tmp_path):
    from orca_teleop.collector_app import Collector
    collector = Collector(tmp_path)
    collector.control_dir = tmp_path / 'controls'
    collector.dataset_root = tmp_path / 'dataset'
    command = collector.build_command([])
    assert command[command.index('--video-storage') + 1] == 'deferred'
    collector.phase = 'preparing'
    with pytest.raises(ValueError, match='세션 종료'):
        collector.convert(tmp_path)
