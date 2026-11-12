"""Strict validation around LeRobot's built-in streaming video encoder."""
import shutil
from pathlib import Path
import queue
import time


class CheckedStreamingEncoder:
    """Encode during recording; refuse to commit dropped or incomplete videos.

    LeRobot normally warns when its bounded encoder queue drops frames. Robot
    demonstrations need exact alignment, so any such drop invalidates the trial.
    Completed MP4s are decoded and checked before episode metadata is committed.
    """

    def __init__(self, encoder, features, fps, *, wait_for_encoder=False):
        self._encoder = encoder
        self._shapes = {key: tuple(ft['shape']) for key, ft in features.items()
                        if ft['dtype'] == 'video'}
        self._fps = fps
        self._counts = {}
        self._first_images = {}
        self._prepared_results = None
        self._wait_for_encoder = wait_for_encoder

    def start_episode(self, video_keys, temp_dir):
        self.cancel_episode()
        self._encoder.start_episode(video_keys=video_keys, temp_dir=temp_dir)
        self._counts = {key: 0 for key in video_keys}
        self._first_images.clear()

    def check(self, expected_frames=None):
        if any(self._encoder._dropped_frames.values()):
            raise RuntimeError('영상 인코딩이 밀려 프레임이 누락되었습니다. 현재 시도를 폐기합니다.')
        if expected_frames is not None and any(
                self._counts.get(key, 0) != expected_frames for key in self._shapes):
            raise RuntimeError('카메라 영상 수와 관절 데이터 프레임 수가 다릅니다.')

    def feed_frame(self, video_key, image):
        if self._wait_for_encoder:
            self._feed_frame_waiting(video_key, image)
        else:
            self._encoder.feed_frame(video_key, image)
        self.check()
        if self._counts[video_key] == 0:
            self._first_images[video_key] = image.copy()
        self._counts[video_key] += 1

    def _feed_frame_waiting(self, video_key, image):
        """Offline conversion can pause its producer instead of dropping input."""
        if not self._encoder._episode_active:
            raise RuntimeError('No active episode. Call start_episode() first.')
        payload = image.copy()
        deadline = time.monotonic() + 120
        while True:
            if not self._encoder._threads[video_key].is_alive():
                # Preserve LeRobot's worker error reporting instead of hiding a crash.
                self._encoder.feed_frame(video_key, image)
                raise RuntimeError(f'{video_key}: 영상 인코더가 종료되었습니다.')
            try:
                self._encoder._frame_queues[video_key].put(payload, timeout=0.1)
                return
            except queue.Full:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'{video_key}: 영상 인코더가 120초 동안 응답하지 않습니다.')

    def prepare_episode(self, expected_frames):
        """Validate before LeRobot writes any rows for the new episode."""
        self.check(expected_frames)
        if self._prepared_results is None:
            self._prepared_results = self._finish_and_validate()

    def finish_episode(self):
        if self._prepared_results is not None:
            results, self._prepared_results = self._prepared_results, None
            return results
        return self._finish_and_validate()

    def _finish_and_validate(self):
        self.check()
        results = self._encoder.finish_episode()
        try:
            if set(results) != set(self._shapes):
                raise RuntimeError('완료된 카메라 영상이 누락되었습니다.')
            import av

            for key, (path, stats) in results.items():
                if stats is None and self._counts[key] != 1:
                    raise RuntimeError(f'{key}: 영상 인코더가 정상 종료되지 않았습니다.')
                count = 0
                with av.open(str(path)) as container:
                    stream = container.streams.video[0]
                    stream.codec_context.thread_count = 2
                    if stream.average_rate is None or float(stream.average_rate) != self._fps:
                        raise RuntimeError(f'{key}: 저장된 영상 FPS가 다릅니다.')
                    for frame in container.decode(video=0):
                        if (frame.height, frame.width, 3) != self._shapes[key]:
                            raise RuntimeError(f'{key}: 저장된 영상 해상도가 다릅니다.')
                        if frame.time is None or abs(frame.time - count / self._fps) > 1e-4:
                            raise RuntimeError(f'{key}: 영상 프레임 타임스탬프가 다릅니다.')
                        count += 1
                if count != self._counts[key]:
                    raise RuntimeError(f'{key}: 영상 프레임 누락 ({count}/{self._counts[key]})')
                if stats is None:
                    # LeRobot intentionally omits worker stats for one-frame
                    # episodes. Compute them from the retained first frame.
                    from lerobot.datasets.compute_stats import (
                        RunningQuantileStats, auto_downsample_height_width,
                    )
                    tracker = RunningQuantileStats()
                    image = auto_downsample_height_width(self._first_images[key].transpose(2, 0, 1))
                    tracker.update(image.transpose(1, 2, 0).reshape(-1, 3))
                    results[key] = (path, tracker.get_statistics())
        except Exception:
            for path, _ in results.values():
                shutil.rmtree(Path(path).parent, ignore_errors=True)
            raise
        return results

    def cancel_episode(self):
        self._encoder.cancel_episode()
        if self._prepared_results is not None:
            for path, _ in self._prepared_results.values():
                shutil.rmtree(Path(path).parent, ignore_errors=True)
            self._prepared_results = None
        self._counts.clear()
        self._first_images.clear()

    def close(self):
        self.cancel_episode()
