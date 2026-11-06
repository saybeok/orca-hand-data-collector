"""Isolate JPEG capture I/O from the teleoperation interpreter's GIL.

Only this private socketpair is pickled; it is never a network listener. The OS
socket has bounded buffers and all parent I/O has deadlines. Frame data cannot
accumulate in an unbounded multiprocessing feeder queue.
"""
from __future__ import annotations

import os
import pickle
import select
import socket
import struct
import subprocess
import sys
import traceback
from pathlib import Path

from orca_teleop.deferred_recording import DeferredCapture

_HEADER = struct.Struct('!Q')


def _send(sock, message):
    data = pickle.dumps(message, protocol=5)
    sock.sendall(_HEADER.pack(len(data)))
    sock.sendall(data)


def _exact(sock, length):
    data = bytearray()
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            raise EOFError('Capture process connection closed.')
        data.extend(chunk)
    return data


def _receive(sock):
    length, = _HEADER.unpack(_exact(sock, _HEADER.size))
    if length > 256 * 1024 * 1024:
        raise ValueError('Capture IPC message exceeds the size limit.')
    return pickle.loads(_exact(sock, length))


class ProcessDeferredCapture(DeferredCapture):
    """Same capture format, with JPEG workers/file writes in a child process."""

    def __init__(self, *, repo_id, fps, features, root):
        self.root, self.fps, self.features = Path(root), fps, features
        self.num_episodes = self.num_frames = 0
        self.episode_buffer = {'size': 0, 'episode_index': 0}
        self._closed = False
        self._accepted = self._written = 0
        self._socket, child = socket.socketpair()
        self._socket.settimeout(15)
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        try:
            self._process = subprocess.Popen(
                [sys.executable, '-m', 'orca_teleop.capture_process', '--worker-fd', str(child.fileno())],
                pass_fds=(child.fileno(),), env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )
        except BaseException:
            self._socket.close()
            raise
        finally:
            child.close()
        try:
            self._request('init', dict(repo_id=repo_id, fps=fps, features=features, root=root))
        except BaseException:
            self._shutdown()
            raise

    @property
    def worker_pid(self):
        return self._process.pid

    def _reply(self):
        try:
            kind, payload = _receive(self._socket)
        except (OSError, EOFError) as exc:
            raise RuntimeError(f'수집 저장 프로세스 응답 실패: {exc}') from exc
        if kind == 'error':
            raise RuntimeError(f'수집 저장 프로세스 실패: {payload}')
        if kind == 'frame':
            self._written += 1
        return kind, payload

    def check(self):
        if self._closed:
            raise RuntimeError('수집 저장 프로세스가 종료되었습니다.')
        while select.select([self._socket], [], [], 0)[0]:
            kind, _ = self._reply()
            if kind != 'frame':
                raise RuntimeError(f'Unexpected capture reply: {kind}')
        if self._process.poll() is not None:
            raise RuntimeError(f'수집 저장 프로세스가 종료되었습니다 (code={self._process.returncode}).')

    def _send(self, op, payload=None):
        try:
            _send(self._socket, (op, payload))
        except (OSError, EOFError) as exc:
            raise RuntimeError(f'수집 저장 프로세스 전송 실패: {exc}') from exc

    def _request(self, op, payload=None):
        self._send(op, payload)
        while True:
            kind, result = self._reply()
            if kind == op:
                return result
            if kind != 'frame':
                raise RuntimeError(f'Unexpected capture reply: {kind}')

    def add_frame(self, frame):
        self.check()
        self._send('frame', frame)
        self._accepted += 1
        self.episode_buffer['size'] += 1

    def prepare_episode(self):
        self.check()
        if self.episode_buffer['size'] < 1:
            raise ValueError('Empty capture episode.')

    def save_episode(self):
        self.prepare_episode()
        result = self._request('save')
        if self._written != self._accepted:
            raise RuntimeError('수집 프로세스의 영상·관절 프레임 수가 다릅니다.')
        self.num_episodes, self.num_frames = result
        self.episode_buffer = {'size': 0, 'episode_index': self.num_episodes}

    def clear_episode_buffer(self):
        self._request('discard')
        self.episode_buffer = {'size': 0, 'episode_index': self.num_episodes}

    def _shutdown(self):
        self._closed = True
        self._socket.close()
        try:
            self._process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()

    def finalize(self):
        if self._closed:
            return
        try:
            self._request('close')
        finally:
            self._shutdown()


def _worker(sock):
    dataset = None
    try:
        import cv2
        # This changes only the writer process. Avoid nested OpenCV pools for
        # tiny RGB conversions; camera/retargeter threads retain their settings.
        cv2.setNumThreads(1)
        while True:
            op, value = _receive(sock)
            if op == 'init':
                dataset = DeferredCapture(**value)
                _send(sock, ('init', None))
            elif op == 'frame':
                dataset.add_frame(value)
                _send(sock, ('frame', None))
            elif op == 'save':
                dataset.save_episode()
                _send(sock, ('save', (dataset.num_episodes, dataset.num_frames)))
            elif op == 'discard':
                dataset.clear_episode_buffer()
                _send(sock, ('discard', None))
            elif op == 'close':
                dataset.finalize()
                _send(sock, ('close', None))
                return
            else:
                raise ValueError(f'Unknown capture operation: {op}')
    except EOFError:
        pass  # Parent exited: discard only the unfinished trial.
    except BaseException:
        try:
            _send(sock, ('error', traceback.format_exc()))
        except OSError:
            pass
    finally:
        if dataset is not None:
            dataset.finalize()
        sock.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-fd', type=int, required=True)
    args = parser.parse_args()
    _worker(socket.socket(fileno=args.worker_fd))
