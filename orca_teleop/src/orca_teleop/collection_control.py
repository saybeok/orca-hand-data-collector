"""Local, file-based controls for the desktop collector (no network listener).

The GUI never opens a motor bus. The recorder owns hardware and cameras and
publishes small preview images independently of its dataset sampling clock.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def send_command(directory: Path, command: str) -> None:
    commands = directory / 'commands'
    commands.mkdir(parents=True, exist_ok=True)
    atomic_json(commands / f'{time.time_ns():020d}_{uuid.uuid4().hex}.json', {'command': command})


class CollectionControl:
    def __init__(self, directory: Path, stop_event: threading.Event,
                 advance_event: threading.Event) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'commands').mkdir(exist_ok=True)
        self.stop_event = stop_event
        self.advance_event = advance_event
        self.start_event = threading.Event()
        self.discard_event = threading.Event()
        self._closed = threading.Event()
        self._lock = threading.RLock()
        self._state = {'phase': 'connecting', 'saved': 0, 'frames': 0, 'seconds': 0}
        self._images = {}
        self._image_revision = 0
        try:
            self._parent_pid = json.loads((directory / 'launch.json').read_text()).get('parent_pid')
        except (OSError, ValueError):
            self._parent_pid = None
        self._thread = threading.Thread(target=self._loop, name='collection-controls', daemon=True)
        self.update()
        self._thread.start()

    def update(self, **fields) -> None:
        with self._lock:
            self._state.update(fields)
            atomic_json(self.directory / 'status.json', self._state)

    def observe(self, observation) -> None:
        with self._lock:
            self._images = dict(observation.images)
            self._image_revision += 1

    def apply_command(self, command: str) -> bool:
        with self._lock:
            phase = self._state['phase']
            if command == 'stop':
                # Unlike terminal q, UI stop must never label a partial trial
                # as a successful demonstration.
                self.discard_event.set()
                self.stop_event.set()
                self.advance_event.set()
                self.update(phase='stopping')
                return True
            if command == 'record' and phase == 'ready':
                self.update(phase='starting')
                self.start_event.set()
                return True
            if command in ('save', 'discard') and phase == 'recording':
                if command == 'discard':
                    self.discard_event.set()
                self.update(phase='finalizing')
                self.advance_event.set()
                return True
            return False

    def _loop(self) -> None:
        revision = -1
        last_preview = 0.0
        while not self._closed.wait(0.05):
            if self._parent_pid and not self.stop_event.is_set():
                try:
                    os.kill(int(self._parent_pid), 0)
                except ProcessLookupError:
                    self.apply_command('stop')
                except PermissionError:
                    pass
            for path in sorted((self.directory / 'commands').glob('*.json')):
                try:
                    command = json.loads(path.read_text())['command']
                    self.apply_command(command)
                except (OSError, ValueError, KeyError):
                    pass
                finally:
                    path.unlink(missing_ok=True)
            if time.monotonic() - last_preview < 0.2:
                continue
            with self._lock:
                images = dict(self._images)
                current_revision = self._image_revision
            if current_revision == revision:
                continue
            try:
                import cv2
                for name, rgb in images.items():
                    # Dataset feature names must never become arbitrary paths.
                    if not name.replace('_', '').isalnum():
                        continue
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    h, w = bgr.shape[:2]
                    # Retain the full 640x480 recording feed. Large desktop
                    # panels should not enlarge an already downsampled image.
                    if w > 1280:
                        bgr = cv2.resize(bgr, (1280, max(1, round(h * 1280 / w))))
                    ok, encoded = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok:
                        target = self.directory / f'{name}.jpg'
                        temporary = target.with_suffix('.jpg.partial')
                        temporary.write_bytes(encoded.tobytes())
                        temporary.replace(target)
            except Exception:
                # A preview failure must not drop a dataset frame or block a
                # stop command. The GUI also displays preview age.
                pass
            revision = current_revision
            last_preview = time.monotonic()

    def close(self) -> None:
        self._closed.set()
        self._thread.join(timeout=2)

    def finish(self, saved: int) -> None:
        with self._lock:
            if self._state['phase'] != 'error':
                self.update(phase='finished', saved=saved)
        self.close()
