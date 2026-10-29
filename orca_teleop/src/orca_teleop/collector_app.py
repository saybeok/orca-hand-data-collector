"""Ubuntu desktop collector. Start through scripts/collector_ui.sh."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

from orca_teleop.cameras import (
    CameraManager, _capture_source, _linux_video_devices, parse_camera_spec,
)
from orca_teleop.collection_control import atomic_json, send_command

CAMERAS = {'bottom': '바닥', 'wrist': '손목', 'realsense': '리얼센스'}
PHASES = {
    'idle': '연결 전', 'preview': '카메라 확인 중', 'preparing': '연결 확인 중',
    'connecting': '텔레옵 연결 중', 'warming_up': '손 트래킹·센서 대기',
    'ready': '텔레옵 가능 · 녹화 대기', 'starting': '녹화 시작 중',
    'recording': '녹화 중', 'finalizing': '저장·정리 중', 'stopping': '종료 중',
    'finished': '세션 종료', 'error': '확인 필요',
}


def camera_sources() -> list[str]:
    devices = _linux_video_devices()
    result = []
    for info in devices.values():
        if info.get('stream_index') != '0' or not info.get('usb_path'):
            continue
        source = 'usb=' + info['usb_path']
        try:
            _capture_source(source)
        except ValueError:
            continue
        serial = info.get('serial')
        same_serial_ports = {d.get('usb_path') for d in devices.values() if d.get('serial') == serial}
        if serial and len(same_serial_ports) == 1:
            source = 'serial=' + serial
        if source not in result:
            result.append(source)
    return result


def camera_spec(name, source):
    source = str(source).strip()
    cfg = parse_camera_spec(f'{name}:{source}:640x480@30')
    node = _capture_source(cfg.index)
    index = int(str(node).removeprefix('/dev/video'))
    info = _linux_video_devices().get(index, {})
    fmt = '' if 'realsense' in info.get('name', '').lower() else ':MJPG'
    return f'{name}:{source}:640x480{fmt}@30', node


def camera_specs(sources: dict[str, str]) -> list[str]:
    resolved = [camera_spec(name, sources[name]) for name in CAMERAS]
    specs, nodes = zip(*resolved)
    if len(set(nodes)) != len(nodes):
        raise ValueError('서로 다른 카메라를 바닥·손목·리얼센스에 각각 지정해 주세요.')
    return list(specs)


class Collector:
    """Process/camera ownership shared by the UI and test harnesses."""
    def __init__(self, root: Path, demo: bool = False):
        self.root, self.demo = root, demo
        self.config_dir = root / 'data' / 'collector_ui'
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.settings = {
            'task': 'Flip over the hwatu card.', 'target': 50,
            'velocity': 15, 'acceleration': 45,
            'data_root': str(root / 'data' / ('collector_demo' if demo else 'lerobot')),
            'sources': {name: parse_camera_spec(os.environ.get(
                f'ORCA_CAMERA_{name.upper()}', f'{name}:0')).index for name in CAMERAS},
        }
        path = self.config_dir / ('demo_settings.json' if demo else 'settings.json')
        if path.exists():
            self.settings.update(json.loads(path.read_text()))
        self.settings_path = path
        self.process = None
        self.control_dir = None
        self.dataset_root = None
        self.preview = None
        self.phase = 'idle'
        self.error = ''
        self.logs = deque(maxlen=150)
        self._lock = threading.RLock()
        self._cancel = threading.Event()

    def active(self):
        return self.phase == 'preparing' or (self.process is not None and self.process.poll() is None)

    def save_settings(self, settings):
        if self.active():
            raise ValueError('연결된 세션을 종료한 뒤 설정을 바꿔 주세요.')
        settings = dict(settings)
        settings['target'] = int(settings['target'])
        if not 1 <= settings['target'] <= 10000:
            raise ValueError('목표 횟수는 1~10000 사이로 입력해 주세요.')
        for key in ('velocity', 'acceleration'):
            settings[key] = float(settings[key])
            if not math.isfinite(settings[key]) or settings[key] <= 0:
                raise ValueError('속도·가속도는 0보다 큰 숫자로 입력해 주세요.')
        if not settings['task'].strip():
            raise ValueError('작업 지시를 입력해 주세요.')
        settings['data_root'] = str(Path(settings['data_root']).expanduser().resolve())
        self.settings = settings
        atomic_json(self.settings_path, settings)

    def diagnostics(self):
        import yaml
        lines = []
        for label, path in (('손', self.root / 'configs/orcahand-right-v1-feetech/config.yaml'),
                            ('팔', self.root / 'configs/am-arm200-3dof/config.yaml')):
            port = Path(yaml.safe_load(path.read_text())['port'])
            lines.append(f'{label}: {"연결됨" if port.exists() else "USB 연결 필요"}')
        if self.demo:
            return 'UI 시험 모드 — 로봇·카메라를 연결하거나 움직이지 않습니다.'
        try:
            result = subprocess.run(['adb', 'devices', '-l'], capture_output=True, text=True, timeout=5)
            lines.append('Quest ADB:\n' + result.stdout.strip() + result.stderr.strip())
        except (OSError, subprocess.TimeoutExpired) as exc:
            lines.append(f'Quest ADB: {exc}')
        for name, source in self.settings['sources'].items():
            try:
                node = _capture_source(source)
                lines.append(f'{CAMERAS[name]}: {node}')
            except (ValueError, OSError) as exc:
                lines.append(f'{CAMERAS[name]}: {exc}')
        return '\n'.join(lines)

    def close_preview(self):
        with self._lock:
            if self.preview:
                self.preview.close()
                self.preview = None

    def open_preview(self):
        if self.active():
            raise ValueError('텔레옵 연결 중에는 녹화 프로세스의 미리보기를 사용합니다.')
        self.close_preview()
        self.process = None
        self.control_dir = None
        if not self.demo:
            specs, nodes = [], set()
            for name, source in self.settings['sources'].items():
                try:
                    spec, node = camera_spec(name, source)
                    if node in nodes:
                        raise ValueError('같은 입력이 다른 카메라에도 지정되어 있습니다.')
                    nodes.add(node)
                    specs.append(spec)
                except ValueError as exc:
                    self.logs.append(f'{CAMERAS[name]} 미리보기 제외: {exc}')
            if not specs:
                raise ValueError('미리볼 카메라가 없습니다. USB 연결과 카메라 선택을 확인해 주세요.')
            manager = CameraManager([parse_camera_spec(s) for s in specs])
            try:
                manager.open()
                manager.ensure_live()
            except BaseException:
                manager.close()
                raise
            with self._lock:
                self.preview = manager
        self.phase = 'preview'

    def frame(self, name):
        with self._lock:
            if self.preview:
                return self.preview.capture().get(name), 0
        if self.control_dir:
            path = self.control_dir / f'{name}.jpg'
            if path.exists():
                import cv2
                bgr = cv2.imread(str(path))
                if bgr is not None:
                    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), time.time() - path.stat().st_mtime
        return None, None

    def connect(self):
        if self.active():
            raise ValueError('이미 연결 중입니다.')
        self.phase = 'preparing'
        self.error = ''
        self._cancel.clear()
        self.process = None
        self.control_dir = None
        self.close_preview()
        try:
            specs = []
            if not self.demo:
                import yaml
                ports = []
                for path in (self.root / 'configs/orcahand-right-v1-feetech/config.yaml',
                             self.root / 'configs/am-arm200-3dof/config.yaml'):
                    port = Path(yaml.safe_load(path.read_text())['port'])
                    if not port.exists():
                        raise ValueError(f'로봇 USB를 연결해 주세요: {port}')
                    ports.append(port.resolve())
                if ports[0] == ports[1]:
                    raise ValueError('팔과 손의 USB 어댑터가 같은 포트로 지정돼 있습니다.')
                specs = camera_specs(self.settings['sources'])
                manager = CameraManager([parse_camera_spec(s) for s in specs])
                try:
                    manager.open()
                    manager.ensure_live()
                finally:
                    manager.close()
                adb = subprocess.run(['adb', 'devices'], capture_output=True, text=True, timeout=8)
                devices = [line.split()[0] for line in adb.stdout.splitlines()
                           if len(line.split()) == 2 and line.split()[1] == 'device']
                if len(devices) != 1:
                    raise ValueError('Quest ADB 연결·USB 디버깅 허용을 확인해 주세요.\n' + adb.stdout)
                result = subprocess.run(['adb', '-s', devices[0], 'reverse', 'tcp:8765', 'tcp:8765'],
                                        capture_output=True, text=True, timeout=8)
                if result.returncode:
                    raise ValueError(result.stderr.strip())
            if self._cancel.is_set():
                raise ValueError('연결을 취소했습니다.')
            session = 'hwatu_' + datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:6]
            self.control_dir = self.config_dir / 'sessions' / session
            self.control_dir.mkdir(parents=True, mode=0o700)
            self.dataset_root = Path(self.settings['data_root']) / session
            self.dataset_root.parent.mkdir(parents=True, exist_ok=True)
            command = self.build_command(specs)
            atomic_json(self.control_dir / 'launch.json',
                        {'argv': command, 'demo': self.demo, 'parent_pid': os.getpid()})
            env = dict(os.environ, PYTHONUNBUFFERED='1')
            env.pop('PYTHONPATH', None)
            env.pop('LD_LIBRARY_PATH', None)
            self.process = subprocess.Popen(command, cwd=self.root, env=env,
                                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, start_new_session=True,
                                            text=True, bufsize=1)
            self.phase = 'connecting'
            if self._cancel.is_set():
                send_command(self.control_dir, 'stop')
            threading.Thread(target=self._read_output, args=(self.process, self.control_dir),
                             daemon=True).start()
        except BaseException as exc:
            self.phase, self.error = 'error', str(exc)
            raise

    def build_command(self, specs):
        s = self.settings
        command = [str(self.root / 'orca_teleop/.venv/bin/python'),
                   str(self.root / 'orca_teleop/scripts/record_dataset.py'),
                   '--backend', 'hardware' if self.demo else 'am200_hw',
                   '--hand-config', str(self.root / 'configs/orcahand-right-v1-feetech/config.yaml'),
                   '--model-path', str(self.root / 'configs/orcahand-right-v1-feetech/config.yaml'),
                   '--urdf-path', str(self.root / 'orcahand_description/v1/models/urdf/orcahand_right.urdf'),
                   '--retargeter', 'adaptive_analytical',
                   '--retarget-config', str(self.root / 'configs/adaptive_analytical_orca_v1_pinch_calibrated.yaml'),
                   '--clutch', '--arm-scale', '0.35',
                   '--arm-max-velocity-deg-s', str(s['velocity']),
                   '--arm-max-acceleration-deg-s2', str(s['acceleration']),
                   '--fps', '30', '--episode-end', 'space', '--no-auto-home', '--rest-seconds', '0',
                   '--num-episodes', str(s['target']), '--task', s['task'],
                   '--repo-id', f'local/{self.dataset_root.name}', '--root', str(self.dataset_root),
                   '--control-dir', str(self.control_dir)]
        command += ['--stub'] if self.demo else ['--local', '--source', 'metaquest', '--hand', 'right']
        for spec in specs:
            command += ['--camera', spec]
        return command

    def _read_output(self, process, directory):
        with (directory / 'recorder.log').open('w', encoding='utf-8') as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                self.logs.append(line.rstrip())

    def status(self):
        state = {'phase': self.phase, 'error': self.error, 'saved': 0, 'frames': 0, 'seconds': 0}
        if self.control_dir:
            try:
                state.update(json.loads((self.control_dir / 'status.json').read_text()))
            except (OSError, ValueError):
                pass
        if self.process is not None and self.process.poll() is not None:
            if self.process.returncode:
                state.update(phase='error', error=self.error or '\n'.join(list(self.logs)[-8:]))
            elif state['phase'] not in ('finished', 'error'):
                state['phase'] = 'finished'
        if self.error:
            state.update(phase='error', error=self.error)
        state['active'] = self.active()
        return state

    def command(self, command):
        if command == 'stop' and self.phase == 'preparing':
            self._cancel.set()
            return
        state = self.status()
        if not state['active'] or not self.control_dir:
            raise ValueError('먼저 텔레옵을 연결해 주세요.')
        allowed = {'record': ('ready',), 'save': ('recording',), 'discard': ('recording',)}
        if command != 'stop' and state['phase'] not in allowed.get(command, ()):
            raise ValueError('현재 단계에서는 이 버튼을 사용할 수 없습니다.')
        send_command(self.control_dir, command)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true', help='UI workflow test only: no hardware, no cameras.')
    args = parser.parse_args(argv)
    from orca_teleop.collector_desktop import launch
    root = Path(os.environ.get('ORCA_HAND_ROOT', Path(__file__).resolve().parents[3]))
    launch(Collector(root, demo=args.demo))


if __name__ == '__main__':
    main()
