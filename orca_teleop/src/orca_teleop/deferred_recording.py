"""Lightweight, durable capture files; LeRobot conversion runs after teleop."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from orca_teleop.collection_control import atomic_json

FORMAT = 'orca-jpeg-capture-v1'
HEADER = struct.Struct('<I')


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class DeferredCapture:
    """Record JPEG-95 frames and joint rows without video encoding or stats.

    Only an atomically renamed episode directory counts as a success. Unfinished
    attempts can be discarded without touching any earlier saved episode.
    """

    image_writer = None
    _streaming_encoder = None

    def __init__(self, *, repo_id, fps, features, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.features, self.fps = features, fps
        self._video = {k: tuple(v['shape']) for k, v in features.items() if v['dtype'] == 'video'}
        self._numeric = {k: v for k, v in features.items() if v['dtype'] not in ('image', 'video')}
        if any(v['dtype'] == 'image' for v in features.values()):
            raise ValueError('Deferred capture expects video camera features.')
        self._pool = ThreadPoolExecutor(max_workers=max(1, len(self._video)), thread_name_prefix='capture-jpeg')
        self.num_episodes = self.num_frames = 0
        self._directory = None
        self._files, self._hashes = {}, {}
        atomic_json(self.root / 'capture.json', {
            'format': FORMAT, 'repo_id': repo_id, 'fps': fps, 'features': features,
            'image_encoding': 'JPEG', 'jpeg_quality': 95,
        })
        with (self.root / 'capture.json').open('rb') as file:
            os.fsync(file.fileno())
        _sync_directory(self.root)
        self._reset_buffer()

    def _reset_buffer(self):
        self.episode_buffer = {'size': 0, 'episode_index': self.num_episodes}
        self._rows = {k: [] for k in self._numeric}
        self._tasks = []

    @staticmethod
    def _jpeg(image):
        import cv2
        ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError('수집 영상 JPEG 저장에 실패했습니다.')
        return encoded.tobytes()

    def add_frame(self, frame):
        if set(frame) != set(self.features) | {'task'} or not isinstance(frame['task'], str):
            raise ValueError('Capture frame features/task mismatch.')
        for key, ft in self._numeric.items():
            value = np.asarray(frame[key])
            if value.shape != tuple(ft['shape']) or value.dtype != np.dtype(ft['dtype']) or not np.isfinite(value).all():
                raise ValueError(f'{key}: invalid numeric capture data')
        for key, shape in self._video.items():
            value = frame[key]
            if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != np.uint8:
                raise ValueError(f'{key}: expected uint8 RGB {shape}')
        if self._directory is None:
            self._directory = self.root / f'.attempt-{uuid.uuid4().hex}'
            self._directory.mkdir()
            self._files = {key: (self._directory / f'camera-{i}.frames').open('wb')
                           for i, key in enumerate(self._video)}
            self._hashes = {key: hashlib.sha256() for key in self._video}
        encoded = list(self._pool.map(self._jpeg, (frame[key] for key in self._video)))
        for key, jpeg in zip(self._video, encoded):
            record = HEADER.pack(len(jpeg)) + jpeg
            self._files[key].write(record)
            self._hashes[key].update(record)
        for key in self._numeric:
            self._rows[key].append(frame[key].copy())
        self._tasks.append(frame['task'])
        self.episode_buffer['size'] += 1

    def prepare_episode(self):
        if not self._tasks or any(len(v) != len(self._tasks) for v in self._rows.values()):
            raise ValueError('Empty or incomplete capture episode.')

    def save_episode(self):
        self.prepare_episode()
        count = self.episode_buffer['size']
        videos = {}
        for key, file in self._files.items():
            file.flush()
            os.fsync(file.fileno())
            videos[key] = {'file': Path(file.name).name, 'sha256': self._hashes[key].hexdigest()}
            file.close()
        self._files.clear()
        arrays = {key: np.stack(rows) for key, rows in self._rows.items()}
        arrays['task'] = np.asarray(self._tasks, dtype=str)
        rows_path = self._directory / 'rows.npz'
        with rows_path.open('wb') as file:
            np.savez(file, **arrays)
            file.flush()
            os.fsync(file.fileno())
        atomic_json(self._directory / 'episode.json', {
            'index': self.num_episodes, 'frames': count, 'videos': videos,
            'rows_sha256': hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        })
        with (self._directory / 'episode.json').open('rb') as file:
            os.fsync(file.fileno())
        _sync_directory(self._directory)
        self._directory.rename(self.root / f'episode-{self.num_episodes:06d}')
        self._directory = None
        _sync_directory(self.root)
        self.num_episodes += 1
        self.num_frames += count
        self._reset_buffer()

    def clear_episode_buffer(self):
        for file in self._files.values():
            file.close()
        self._files.clear()
        if self._directory is not None:
            shutil.rmtree(self._directory)
            self._directory = None
        self._reset_buffer()

    def stop_image_writer(self):
        pass

    def finalize(self):
        self.clear_episode_buffer()
        self._pool.shutdown(wait=True)


def capture_fingerprint(root):
    paths = [root / 'capture.json', *sorted(root.glob('episode-*/episode.json'))]
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()


def capture_roots(source):
    source = Path(source)
    roots = [source] if (source / 'capture.json').is_file() else sorted(
        p.parent for p in source.glob('*/capture.json'))
    return [root for root in roots if list(root.glob('episode-*/episode.json'))]


def _read_jpeg(file, shape):
    import cv2
    raw = file.read(HEADER.size)
    if len(raw) != HEADER.size:
        raise ValueError('수집 영상 프레임 헤더가 누락되었습니다.')
    length, = HEADER.unpack(raw)
    # JPEG must be comfortably smaller than this bound at the recorded size.
    if not 0 < length <= int(np.prod(shape)) * 4 + 65536:
        raise ValueError('Invalid capture frame length.')
    raw = file.read(length)
    if len(raw) != length:
        raise ValueError('수집 영상 프레임이 잘렸습니다.')
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape != tuple(shape):
        raise ValueError('수집 영상 프레임이 손상되었습니다.')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def convert_capture(root, progress=lambda **fields: None):
    """Validate all sources, convert into staging, then publish atomically.

    No robot/camera is opened. Input capture files are never modified/deleted.
    Existing converted data is reused only when its source fingerprint matches.
    """
    import fcntl
    root = Path(root)
    with (root / '.conversion.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('이 수집본은 이미 다른 프로세스에서 변환 중입니다.') from exc
        for orphan in root.glob('.lerobot-converting-*'):
            suffix = orphan.name.removeprefix('.lerobot-converting-')
            if len(suffix) == 32 and all(c in '0123456789abcdef' for c in suffix) and not orphan.is_symlink():
                shutil.rmtree(orphan)
        return _convert_capture_locked(root, progress)


def _convert_capture_locked(root, progress):
    from contextlib import ExitStack
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from orca_teleop.recording_storage import CheckedStreamingEncoder

    root = Path(root)
    capture = json.loads((root / 'capture.json').read_text())
    if capture['format'] != FORMAT:
        raise ValueError('Unknown capture format.')
    fingerprint = capture_fingerprint(root)
    output = root / 'lerobot'
    if output.exists():
        marker = output / 'conversion.json'
        if marker.exists() and json.loads(marker.read_text()).get('source_sha256') == fingerprint:
            return output
        raise ValueError(f'기존 변환 결과와 원본이 다릅니다. 덮어쓰지 않습니다: {output}')
    episodes = sorted(root.glob('episode-*/episode.json'))
    if not episodes:
        raise ValueError('성공 저장된 시연이 없습니다.')
    staging = root / f'.lerobot-converting-{uuid.uuid4().hex}'
    dataset = None
    try:
        features = {key: dict(ft, shape=tuple(ft['shape'])) for key, ft in capture['features'].items()}
        video = {key: ft for key, ft in features.items() if ft['dtype'] == 'video'}
        dataset = LeRobotDataset.create(repo_id=capture['repo_id'], fps=capture['fps'],
            features=features, root=staging, use_videos=True, vcodec='h264',
            streaming_encoding=bool(video), encoder_threads=2, encoder_queue_maxsize=30,
            image_writer_processes=0, image_writer_threads=0)
        if video:
            dataset._streaming_encoder = CheckedStreamingEncoder(
                dataset._streaming_encoder, features, capture['fps'], wait_for_encoder=True)
        for index, path in enumerate(episodes):
            episode = json.loads(path.read_text())
            if episode['index'] != index:
                raise ValueError('수집 에피소드 순서가 잘못되었습니다.')
            directory = path.parent
            rows_path = directory / 'rows.npz'
            if _sha256_file(rows_path) != episode['rows_sha256']:
                raise ValueError('관절 데이터 체크섬이 다릅니다.')
            if set(episode['videos']) != set(video):
                raise ValueError('수집 카메라 목록이 다릅니다.')
            with ExitStack() as stack:
                rows = stack.enter_context(np.load(rows_path, allow_pickle=False))
                expected = set(features) - set(video) | {'task'}
                if set(rows.files) != expected or any(len(rows[key]) != episode['frames'] for key in rows.files):
                    raise ValueError('관절 데이터와 영상 프레임 수가 다릅니다.')
                files = {}
                for key, info in episode['videos'].items():
                    # Capture filenames are generated locally, never arbitrary paths.
                    if Path(info['file']).name != info['file']:
                        raise ValueError('Invalid capture filename.')
                    source = directory / info['file']
                    if _sha256_file(source) != info['sha256']:
                        raise ValueError(f'{key}: 수집 영상 체크섬이 다릅니다.')
                    files[key] = stack.enter_context(source.open('rb'))
                for i in range(episode['frames']):
                    frame = {key: rows[key][i] for key in expected}
                    frame['task'] = str(frame['task'])
                    for key in files:
                        frame[key] = _read_jpeg(files[key], video[key]['shape'])
                    dataset.add_frame(frame)
                    if i % capture['fps'] == 0:
                        progress(episode=index + 1, total=len(episodes), frames=i + 1,
                                 episode_frames=episode['frames'], source=str(root))
                if any(file.read(1) for file in files.values()):
                    raise ValueError('영상에 초과 프레임이 있습니다.')
            if video:
                dataset._streaming_encoder.prepare_episode(episode['frames'])
            dataset.save_episode()
            progress(episode=index + 1, total=len(episodes), frames=episode['frames'],
                     episode_frames=episode['frames'], source=str(root))
        dataset.finalize()
        for source in root.glob('collection_*'):
            if source.is_file():
                shutil.copyfile(source, staging / source.name)
        atomic_json(staging / 'conversion.json', {
            'source_sha256': fingerprint, 'episodes': dataset.num_episodes,
            'frames': dataset.num_frames, 'source_encoding': 'JPEG quality 95',
        })
        staging.rename(output)
        _sync_directory(root)
        return output
    finally:
        try:
            if dataset is not None:
                dataset.finalize()
        finally:
            if staging.exists():
                shutil.rmtree(staging)
