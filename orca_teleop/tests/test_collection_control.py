"""Desktop collection semantics: never save a failed or unfinished attempt."""
import json
import threading
import time

import numpy as np
import pytest

from orca_teleop.collection_control import CollectionControl, send_command
from orca_teleop.collector_app import Collector, camera_specs


def test_buttons_are_phase_gated_and_duplicate_save_is_rejected(tmp_path):
    stop, advance = threading.Event(), threading.Event()
    control = CollectionControl(tmp_path, stop, advance)
    try:
        assert not control.apply_command('save')
        control.update(phase='ready')
        assert control.apply_command('record')
        assert control.start_event.is_set()
        assert not control.apply_command('record')
        control.update(phase='recording')
        assert control.apply_command('save')
        assert advance.is_set() and not control.discard_event.is_set()
        assert not control.apply_command('save')
    finally:
        control.close()


@pytest.mark.parametrize('command', ['discard', 'stop'])
def test_discard_and_stop_do_not_mark_trial_successful(tmp_path, command):
    stop, advance = threading.Event(), threading.Event()
    control = CollectionControl(tmp_path, stop, advance)
    try:
        control.update(phase='recording')
        assert control.apply_command(command)
        assert control.discard_event.is_set() and advance.is_set()
        assert stop.is_set() == (command == 'stop')
        assert json.loads((tmp_path / 'status.json').read_text())['saved'] == 0
    finally:
        control.close()


def test_file_commands_are_consumed_and_error_survives_finish(tmp_path):
    control = CollectionControl(tmp_path, threading.Event(), threading.Event())
    try:
        control.update(phase='ready')
        send_command(tmp_path, 'record')
        assert control.start_event.wait(2)
        control.update(phase='error', error='disk full')
        control.finish(0)
        status = json.loads((tmp_path / 'status.json').read_text())
        assert status['phase'] == 'error' and status['error'] == 'disk full'
    finally:
        control.close()


def test_disappearing_ui_owner_stops_and_discards_the_unfinished_trial(tmp_path, monkeypatch):
    from orca_teleop import collection_control as module

    (tmp_path / 'launch.json').write_text(json.dumps({'parent_pid': 123456}))
    def missing_owner(pid, signal):
        raise ProcessLookupError()
    monkeypatch.setattr(module.os, 'kill', missing_owner)
    stop = threading.Event()
    control = CollectionControl(tmp_path, stop, threading.Event())
    try:
        assert stop.wait(2)
        assert control.discard_event.is_set()
    finally:
        control.close()


def test_preview_does_not_write_outside_control_directory(tmp_path):
    cv2 = pytest.importorskip('cv2')
    control = CollectionControl(tmp_path, threading.Event(), threading.Event())
    try:
        rgb = np.full((480, 640, 3), (240, 0, 0), dtype=np.uint8)
        observation = type('Observation', (), {'images': {'bottom': rgb, '../escape': rgb}})()
        control.observe(observation)
        deadline = time.monotonic() + 2
        while not (tmp_path / 'bottom.jpg').exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        bgr = cv2.imread(str(tmp_path / 'bottom.jpg'))
        assert bgr is not None and bgr[0, 0, 2] > 200 and bgr[0, 0, 0] < 20
        assert bgr.shape == rgb.shape, 'The desktop preview must retain the full recording resolution.'
        assert not (tmp_path.parent / 'escape.jpg').exists()
    finally:
        control.close()


def test_different_selectors_for_same_physical_camera_are_rejected(monkeypatch):
    import orca_teleop.collector_app as module
    monkeypatch.setattr(module, '_capture_source', lambda selector: '/dev/video8')
    monkeypatch.setattr(module, '_linux_video_devices', lambda: {})
    with pytest.raises(ValueError, match='서로 다른'):
        camera_specs({'bottom': 'usb=2-2', 'wrist': 'serial=rs', 'realsense': '8'})


def test_real_and_demo_commands_keep_settings_and_never_overwrite_data(tmp_path):
    real = Collector(tmp_path)
    real.dataset_root = tmp_path / 'dataset'
    real.control_dir = tmp_path / 'controls'
    cmd = real.build_command(['bottom:usb=1-1:640x480:MJPG@30'])
    assert cmd[cmd.index('--backend') + 1] == 'am200_hw'
    assert '--no-auto-home' in cmd and '--overwrite' not in cmd and '--push-to-hub' not in cmd
    assert cmd[cmd.index('--arm-max-velocity-deg-s') + 1] == '15'
    assert '--local' in cmd and '--control-dir' in cmd
    demo = Collector(tmp_path, demo=True)
    demo.dataset_root, demo.control_dir = real.dataset_root, real.control_dir
    cmd = demo.build_command([])
    assert '--stub' in cmd and '--local' not in cmd and '--camera' not in cmd
    assert 'collector_demo' in demo.settings['data_root']


def test_settings_survive_project_relocation_without_serial_device_changes(tmp_path):
    collector = Collector(tmp_path)
    settings = dict(collector.settings, velocity=10, target=5)
    collector.save_settings(settings)
    assert Collector(tmp_path).settings['velocity'] == 10
    with pytest.raises(ValueError):
        collector.save_settings(dict(settings, acceleration=float('nan')))


@pytest.mark.parametrize('demo', [False, True])
@pytest.mark.parametrize('task', ['Flip over the hwatu card.', 'my custom task'])
def test_task_default_migration_preserves_custom_instructions_and_camera_roles(tmp_path, demo, task):
    collector = Collector(tmp_path, demo=demo)
    assert collector.settings['task'] == 'move the hwatu card.'
    sources = {'bottom': 'usb=1-1.3', 'wrist': 'usb=1-1.4', 'realsense': 'serial=rs'}
    collector.save_settings(dict(collector.settings, task=task, sources=sources))
    reloaded = Collector(tmp_path, demo=demo)
    expected = 'move the hwatu card.' if task == 'Flip over the hwatu card.' else task
    assert reloaded.settings['task'] == expected
    persisted = json.loads(reloaded.settings_path.read_text())
    assert persisted['task'] == expected
    assert persisted['sources'] == sources


def test_frame_write_failure_discards_episode_instead_of_counting_success():
    import importlib.util
    import queue
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / 'scripts/record_dataset.py'
    spec = importlib.util.spec_from_file_location('collector_recorder_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Dataset:
        saved = 0
        cleared = 0
        def add_frame(self, frame):
            raise OSError('disk full')
        def save_episode(self):
            self.saved += 1
        def clear_episode_buffer(self):
            self.cleared += 1

    dataset = Dataset()
    frames, finalized = queue.Queue(), queue.Queue()
    failed = threading.Event()
    for frame in ({'action': [1]}, {'action': [2]}, module._SAVE_EPISODE, None):
        frames.put(frame)
    module._recorder_loop(dataset, frames, finalized, failed)
    assert failed.is_set()
    assert finalized.get_nowait() is False
    assert dataset.saved == 0 and dataset.cleared == 1
