"""Exercise actual LeRobot asynchronous image writing without cameras or motors."""
import importlib.util
import queue
import threading
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def recorder():
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'record_dataset.py'
    spec = importlib.util.spec_from_file_location('image_storage_recorder', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def make_dataset(tmp_path, recorder, monkeypatch):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from tqdm import tqdm

    # The library's process-wide progress monitor outlives individual datasets.
    monkeypatch.setattr(tqdm, 'monitor_interval', 0)
    monkeypatch.setattr('datasets.config.HF_DATASETS_CACHE', tmp_path / 'hf-cache')

    datasets = []

    def create(cameras=('bottom', 'wrist', 'realsense'), video_storage='images'):
        features = {
            'observation.state': {'dtype': 'float32', 'shape': (20,), 'names': None},
            'action': {'dtype': 'float32', 'shape': (20,), 'names': None},
        }
        features.update({f'observation.images.{name}': {
            'dtype': 'video', 'shape': (64, 64, 3), 'names': ['height', 'width', 'channels'],
        } for name in cameras})
        dataset = recorder._create_recording_dataset(
            LeRobotDataset, repo_id='test/async-images', fps=30,
            features=features, root=tmp_path / f'dataset-{len(datasets)}',
            video_storage=video_storage,
        )
        datasets.append(dataset)
        return dataset

    yield create
    for dataset in datasets:
        dataset.stop_image_writer()
        dataset.finalize()


def frame(dataset, index):
    row = {
        'observation.state': np.full(20, index / 100, dtype=np.float32),
        'action': np.full(20, index / 100, dtype=np.float32),
        'task': 'move the hwatu card.',
    }
    for key, feature in dataset.features.items():
        if feature['dtype'] == 'video':
            row[key] = np.full(feature['shape'], index * 20, dtype=np.uint8)
    return row


def run_episode(recorder, dataset, length=6):
    frames, finalized, failed = queue.Queue(), queue.Queue(), threading.Event()
    for index in range(length):
        frames.put(frame(dataset, index))
    frames.put(recorder._SAVE_EPISODE)
    frames.put(None)
    recorder._recorder_loop(dataset, frames, finalized, failed)
    return finalized.get_nowait(), failed


def test_three_camera_episode_keeps_all_rows_and_video_frames(recorder, make_dataset):
    import av

    dataset = make_dataset()
    assert dataset.image_writer.num_threads == 12
    assert dataset.image_writer.num_processes == 0
    succeeded, failed = run_episode(recorder, dataset)
    assert succeeded and not failed.is_set()
    assert dataset.num_episodes == 1 and dataset.num_frames == 6
    videos = list(dataset.root.rglob('*.mp4'))
    assert len(videos) == 3
    for video in videos:
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            assert float(stream.average_rate) == 30
            assert sum(1 for _ in container.decode(video=0)) == 6


def test_streaming_keeps_all_camera_frames_and_numeric_rows_without_pngs(recorder, make_dataset):
    import av

    dataset = make_dataset(video_storage='streaming')
    assert dataset.image_writer is None
    assert dataset.vcodec == 'h264'
    succeeded, failed = run_episode(recorder, dataset, length=9)
    assert succeeded and not failed.is_set()
    assert dataset.num_frames == 9 and dataset.num_episodes == 1
    assert not list(dataset.root.rglob('*.png'))
    for video in dataset.root.rglob('*.mp4'):
        with av.open(str(video)) as container:
            assert container.streams.video[0].codec_context.name == 'h264'
            frames = list(container.decode(video=0))
            assert len(frames) == 9
            np.testing.assert_allclose([f.time for f in frames], np.arange(9)/30)


def test_streaming_discard_then_save_does_not_keep_old_trial_frames(recorder, make_dataset):
    import av

    dataset = make_dataset(('wrist',), video_storage='streaming')
    for index in range(9):
        dataset.add_frame(frame(dataset, index))
    recorder._discard_recording_episode(dataset)
    assert not list(dataset.root.rglob('*.mp4'))
    succeeded, failed = run_episode(recorder, dataset, length=3)
    assert succeeded and not failed.is_set()
    assert dataset.num_frames == 3
    with av.open(str(next(dataset.root.rglob('*.mp4')))) as container:
        assert sum(1 for _ in container.decode(video=0)) == 3


def test_streaming_frame_drop_rejects_trial_instead_of_counting_success(recorder, make_dataset):
    dataset = make_dataset(('wrist',), video_storage='streaming')
    dataset.add_frame(frame(dataset, 0))
    dataset._streaming_encoder._encoder._dropped_frames['observation.images.wrist'] = 1
    q, finalized, failed = queue.Queue(), queue.Queue(), threading.Event()
    q.put(recorder._SAVE_EPISODE)
    q.put(None)
    recorder._recorder_loop(dataset, q, finalized, failed)
    assert finalized.get_nowait() is False and failed.is_set()
    assert dataset.num_episodes == 0


def test_streaming_final_frame_loss_is_rejected_before_success(recorder, make_dataset):
    dataset = make_dataset(('wrist',), video_storage='streaming')
    dataset.add_frame(frame(dataset, 0))
    dataset.add_frame(frame(dataset, 1))
    encoder = dataset._streaming_encoder
    # Simulate a worker that silently loses its final frame after queueing it.
    encoder._counts['observation.images.wrist'] += 1
    dataset.episode_buffer['size'] = 3
    with pytest.raises(RuntimeError, match='프레임 누락'):
        encoder.finish_episode()
    assert dataset.num_episodes == 0


def test_streaming_single_frame_episode_has_image_statistics(recorder, make_dataset):
    dataset = make_dataset(('wrist',), video_storage='streaming')
    succeeded, failed = run_episode(recorder, dataset, length=1)
    assert succeeded and not failed.is_set()
    assert dataset.num_frames == 1
    stats = dataset.meta.stats['observation.images.wrist']
    assert stats['mean'].shape == (3, 1, 1)
    assert np.isfinite(stats['std']).all()


def test_streaming_multiple_saved_episodes_keep_frame_count_and_timestamps(recorder, make_dataset):
    import av

    dataset = make_dataset(('wrist',), video_storage='streaming')
    assert run_episode(recorder, dataset, length=6)[0]
    assert run_episode(recorder, dataset, length=3)[0]
    assert dataset.num_episodes == 2 and dataset.num_frames == 9
    with av.open(str(next(dataset.root.rglob('*.mp4')))) as container:
        frames = list(container.decode(video=0))
        assert len(frames) == 9
        np.testing.assert_allclose([f.time for f in frames], np.arange(9)/30, atol=1e-4)


def test_streaming_dataset_reloads_for_training(recorder, make_dataset):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = make_dataset(('wrist',), video_storage='streaming')
    assert run_episode(recorder, dataset, length=6)[0]
    dataset.finalize()
    loaded = LeRobotDataset(repo_id=dataset.repo_id, root=dataset.root, video_backend='pyav')
    assert len(loaded) == 6
    np.testing.assert_allclose(loaded[5]['action'].numpy(), .05)
    assert tuple(loaded[5]['observation.images.wrist'].shape) == (3, 64, 64)
    loaded.finalize()


def test_streaming_invalid_video_is_rejected_before_writing_more_rows(recorder, make_dataset, monkeypatch):
    dataset = make_dataset(('wrist',), video_storage='streaming')
    assert run_episode(recorder, dataset, length=3)[0]
    inner = dataset._streaming_encoder._encoder
    original = inner.finish_episode
    def broken_video():
        results = original()
        for path, _ in results.values():
            Path(path).write_bytes(b'corrupt mp4')
        return results
    monkeypatch.setattr(inner, 'finish_episode', broken_video)
    before = dataset.latest_episode['index'].copy()
    succeeded, failed = run_episode(recorder, dataset, length=6)
    assert not succeeded and failed.is_set()
    assert dataset.num_episodes == 1 and dataset.num_frames == 3
    np.testing.assert_array_equal(dataset.latest_episode['index'], before)


@pytest.mark.parametrize('failure', ['missing', 'corrupt'])
def test_png_write_failure_never_commits_episode(recorder, make_dataset, monkeypatch, failure):
    from PIL import Image

    dataset = make_dataset(('wrist',))

    def fail_save(image, path, **kwargs):
        if failure == 'missing':
            # LeRobot catches this exception inside its image writer.
            raise OSError('disk full')
        Path(path).write_bytes(b'incomplete PNG')

    monkeypatch.setattr(Image.Image, 'save', fail_save)
    succeeded, failed = run_episode(recorder, dataset)
    assert not succeeded and failed.is_set()
    assert dataset.num_episodes == 0 and dataset.num_frames == 0


def test_save_waits_for_pending_image_writes(recorder, make_dataset, monkeypatch):
    import lerobot.datasets.image_writer as module

    dataset = make_dataset(('wrist',))
    started, release = threading.Event(), threading.Event()
    original = module.write_image

    def slow_write(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, 'write_image', slow_write)
    dataset.add_frame(frame(dataset, 0))
    assert started.wait(2)
    checked = threading.Event()

    def validate():
        recorder._validate_recorded_images(dataset)
        checked.set()

    worker = threading.Thread(target=validate)
    worker.start()
    try:
        assert not checked.wait(0.05)
    finally:
        release.set()
        worker.join(timeout=5)
    assert checked.is_set()
    dataset.clear_episode_buffer()


def test_overloaded_image_writer_aborts_instead_of_growing_memory(recorder, make_dataset, monkeypatch):
    dataset = make_dataset(('wrist',))
    monkeypatch.setattr(dataset.image_writer.queue, 'qsize', lambda: 64)
    succeeded, failed = run_episode(recorder, dataset)
    assert not succeeded and failed.is_set()
    assert dataset.num_episodes == 0


def test_shorter_retry_does_not_keep_discarded_video_frames(recorder, make_dataset):
    import av

    dataset = make_dataset(('wrist',))
    frames, finalized = queue.Queue(), queue.Queue()
    for index in range(9):
        frames.put(frame(dataset, index))
    frames.put(recorder._DISCARD_EPISODE)
    frames.put(None)
    recorder._recorder_loop(dataset, frames, finalized)
    assert finalized.get_nowait() is True
    assert not list((dataset.root / 'images').rglob('*.png'))
    succeeded, failed = run_episode(recorder, dataset, length=3)
    assert succeeded and not failed.is_set() and dataset.num_frames == 3
    with av.open(str(next(dataset.root.rglob('*.mp4')))) as container:
        assert sum(1 for _ in container.decode(video=0)) == 3


def test_no_camera_collection_needs_no_image_writer(recorder, make_dataset):
    dataset = make_dataset(())
    assert dataset.image_writer is None
    succeeded, failed = run_episode(recorder, dataset)
    assert succeeded and not failed.is_set()
    assert dataset.num_frames == 6
