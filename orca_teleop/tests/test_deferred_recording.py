"""Capture commits are fast and durable; conversion preserves aligned rows/video."""
import json
from pathlib import Path

import numpy as np
import pytest

from orca_teleop.deferred_recording import DeferredCapture, capture_roots, convert_capture


def make_capture(root, cameras=3, isolated=False):
    features = {k: {'dtype': 'float32', 'shape': (20,), 'names': None}
                for k in ('observation.state', 'action')}
    features.update({f'observation.images.cam{i}': {
        'dtype': 'video', 'shape': (64, 64, 3), 'names': ['height', 'width', 'channels'],
    } for i in range(cameras)})
    from orca_teleop.capture_process import ProcessDeferredCapture
    cls = ProcessDeferredCapture if isolated else DeferredCapture
    return cls(root=root, repo_id='test/capture', fps=30, features=features)


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


def test_offline_conversion_retries_full_encoder_queue_without_dropping_frames(tmp_path, monkeypatch):
    import queue
    import av
    from lerobot.datasets.video_utils import StreamingVideoEncoder
    original = StreamingVideoEncoder.start_episode
    full_events = []

    def start_episode(encoder, *args, **kwargs):
        original(encoder, *args, **kwargs)
        for key, frames in encoder._frame_queues.items():
            put = frames.put
            attempts = [0]
            def delayed_put(item, block=True, timeout=None, *, put=put, attempts=attempts, key=key):
                if item is not None and attempts[0] < 2:
                    attempts[0] += 1
                    full_events.append(key)
                    raise queue.Full
                return put(item, block=block, timeout=timeout)
            frames.put = delayed_put
    monkeypatch.setattr(StreamingVideoEncoder, 'start_episode', start_episode)
    capture = make_capture(tmp_path / 'session', cameras=1)
    for i in range(4):
        capture.add_frame(frame(capture, i))
    capture.save_episode()
    capture.finalize()
    output = convert_capture(capture.root)
    assert len(full_events) == 2
    with av.open(str(next((output / 'videos').rglob('*.mp4')))) as container:
        frames = list(container.decode(video=0))
    assert len(frames) == 4
    np.testing.assert_allclose([f.time for f in frames], np.arange(4) / 30)


def test_offline_encoder_backpressure_reports_crashed_worker():
    from types import SimpleNamespace
    from orca_teleop.recording_storage import CheckedStreamingEncoder
    def crashed(*args):
        raise RuntimeError('Encoder worker crashed: disk error')
    encoder = SimpleNamespace(_episode_active=True,
        _threads={'camera': SimpleNamespace(is_alive=lambda: False)}, feed_frame=crashed)
    checked = CheckedStreamingEncoder(encoder,
        {'camera': {'dtype': 'video', 'shape': (4, 4, 3)}}, 30, wait_for_encoder=True)
    with pytest.raises(RuntimeError, match='disk error'):
        checked.feed_frame('camera', np.zeros((4, 4, 3), np.uint8))


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


def test_process_capture_flushes_all_cameras_and_discards_only_unfinished_attempt(tmp_path):
    import av
    import os
    capture = make_capture(tmp_path / 'session', isolated=True)
    assert capture.worker_pid != os.getpid()
    try:
        for i in range(8):
            capture.add_frame(frame(capture, i))
        capture.clear_episode_buffer()
        for count in (6, 3):
            for i in range(count):
                row = frame(capture, i)
                capture.add_frame(row)
                # Camera drivers and callers may release/reuse these buffers.
                row['observation.images.cam0'][:] = 0
            capture.save_episode()
        assert capture.num_episodes == 2 and capture.num_frames == 9
        capture.add_frame(frame(capture, 99))
    finally:
        capture.finalize()
    assert capture._process.poll() == 0
    assert not list(capture.root.glob('.attempt-*'))
    output = convert_capture(capture.root)
    for path in output.rglob('*.mp4'):
        with av.open(str(path)) as container:
            frames = list(container.decode(video=0))
        assert len(frames) == 9
        np.testing.assert_allclose([f.time for f in frames], np.arange(9)/30)
        assert frames[0].to_ndarray(format='rgb24')[:, :, 0].mean() > 180


def test_process_write_error_is_propagated_and_never_counts_a_success(tmp_path):
    capture = make_capture(tmp_path / 'session', isolated=True)
    row = frame(capture, 0)
    row['action'] = np.zeros(19, np.float32)
    capture.add_frame(row)
    try:
        with pytest.raises(RuntimeError, match='invalid numeric capture data'):
            capture.save_episode()
        assert capture.num_episodes == 0
    finally:
        try:
            capture.finalize()
        except RuntimeError:
            pass
    assert capture._process.poll() is not None
    assert not list(capture.root.glob('episode-*'))


def test_terminated_capture_process_keeps_previously_saved_trials(tmp_path):
    capture = make_capture(tmp_path / 'session', isolated=True, cameras=1)
    capture.add_frame(frame(capture, 0))
    capture.save_episode()
    previous = next(capture.root.glob('episode-*/*.frames')).read_bytes()
    capture._process.terminate()
    capture._process.wait(timeout=3)
    try:
        with pytest.raises(RuntimeError):
            capture.add_frame(frame(capture, 1))
    finally:
        try:
            capture.finalize()
        except RuntimeError:
            pass
    assert next(capture.root.glob('episode-*/*.frames')).read_bytes() == previous
    assert convert_capture(capture.root).is_dir()
