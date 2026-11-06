"""Device-specific PCsensor controls for the Ubuntu/X11 collector."""
from __future__ import annotations

import queue
import select
import threading
import time

PEDAL_BINDINGS = {'a': 'record', 'b': 'save', 'c': 'discard'}
_PHASE = {'record': 'ready', 'save': 'recording', 'discard': 'recording'}
_DEVICE = 'PCsensor FootSwitch Keyboard'


class PedalPressFilter:
    """Ignore repeat, held-key retriggers and mechanical switch bounce."""

    def __init__(self):
        self.held = set()
        self._last = {}

    def press(self, code, name, *, repeat=False, now=None):
        now = time.monotonic() if now is None else now
        if repeat or code in self.held:
            return None
        self.held.add(code)
        command = PEDAL_BINDINGS.get((name or '').lower())
        if command is None or now - self._last.get(code, -float('inf')) < .25:
            return None
        self._last[code] = now
        return command

    def release(self, code):
        self.held.discard(code)


class PedalCommands:
    """Capture intent on the input thread; dispatch on Tk's main thread.

    A press is valid only in the same phase, trial and session when consumed.
    Keys pressed while homing/saving never become commands for the next trial.
    """

    def __init__(self, collector):
        self.collector = collector
        self._pending = queue.SimpleQueue()

    def receive(self, command):
        if command not in _PHASE:
            return
        state = self.collector.status()
        if not state['active'] or state['phase'] != _PHASE.get(command):
            return
        self._pending.put((command, self.collector.control_dir, state.get('attempt'), time.monotonic()))

    def dispatch_pending(self):
        while not self._pending.empty():
            command, session, attempt, created = self._pending.get_nowait()
            state = self.collector.status()
            if (time.monotonic() - created > 1 or not state['active']
                    or session != self.collector.control_dir or attempt != state.get('attempt')
                    or state['phase'] != _PHASE[command]):
                continue
            try:
                self.collector.command(command)
                self.collector.logs.append('페달: ' + {'record': '녹화 시작', 'save': '성공 저장',
                                                      'discard': '실패 · 재시도'}[command])
            except ValueError as exc:
                self.collector.logs.append(f'페달 입력 무시: {exc}')


class PedalListener:
    """Grab only the identified pedal, even while another app has focus.

    Uses the user's X11 session, with no root/input-group privileges. X closes
    the device grab automatically if this app exits. Ordinary keyboards are
    never selected or grabbed. Re-query USB devices so event/id renumbering is
    harmless. Xlib is optional: the mouse-operated collector still works.
    """

    def __init__(self, on_command):
        self.on_command = on_command
        self.status = '페달 확인 중'
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, name='collector-pedals', daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self):
        connection = None
        try:
            from Xlib import X, XK, display
            from Xlib.ext import xinput

            connection = display.Display()
            root = connection.screen().root
            device = None
            keys = PedalPressFilter()
            next_probe = 0
            while not self._stop.is_set():
                if time.monotonic() >= next_probe:
                    devices = connection.xinput_query_device(xinput.AllDevices).devices
                    matching = [d.deviceid for d in devices if d.name == _DEVICE]
                    wanted = matching[0] if len(matching) == 1 else None
                    if wanted != device:
                        if device in {d.deviceid for d in devices}:
                            connection.xinput_ungrab_device(device, X.CurrentTime)
                            connection.sync()
                        device = None
                        keys = PedalPressFilter()
                        if wanted is not None:
                            result = root.xinput_grab_device(
                                wanted, X.CurrentTime, xinput.GrabModeAsync, xinput.GrabModeAsync,
                                False, xinput.KeyPressMask | xinput.KeyReleaseMask)
                            if result.status == X.GrabSuccess:
                                device = wanted
                                self.status = '페달 연결됨 · 왼쪽 시작 / 가운데 저장 / 오른쪽 재시도'
                            else:
                                self.status = '페달을 다른 앱이 사용 중'
                    if wanted is None:
                        self.status = ('같은 페달이 여러 대 연결됨' if len(matching) > 1
                                       else '페달 미연결 · 마우스 버튼 사용 가능')
                    next_probe = time.monotonic() + 1
                if not connection.pending_events() and not select.select(
                        [connection.fileno()], [], [], .1)[0]:
                    continue
                event = connection.next_event()
                # Python-Xlib's X constants do not include GenericEvent (35).
                if event.type != 35 or event.evtype not in (xinput.KeyPress, xinput.KeyRelease):
                    continue
                info = event.data
                if info.sourceid != device:
                    continue
                code = int(info.detail)
                if event.evtype == xinput.KeyRelease:
                    keys.release(code)
                else:
                    name = XK.keysym_to_string(connection.keycode_to_keysym(code, 0))
                    command = keys.press(code, name, repeat=bool(info.flags & xinput.KeyRepeat))
                    if command:
                        self.on_command(command)
        except ImportError:
            self.status = '페달 기능: python3-xlib 설치 필요'
        except Exception as exc:
            self.status = f'페달 연결 확인 필요 (X11): {exc}'
        finally:
            if connection is not None:
                connection.close()  # Releases this client's pedal grab.
