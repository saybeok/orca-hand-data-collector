"""All-input camera preview: frame identity, failure isolation, cleanup."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

spec = importlib.util.spec_from_file_location(
    'camera_preview_cli', Path(__file__).resolve().parents[1]/'scripts/preview_cameras.py')
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)


def test_grid_preserves_distinct_views_and_labels():
    red = np.full((60, 80, 3), [255, 0, 0], dtype=np.uint8)
    green = np.full((60, 80, 3), [0, 255, 0], dtype=np.uint8)
    grid = preview._preview_grid(['cam4', 'cam8'], {'cam4': red, 'cam8': green},
                                 {'cam4': 'video4 USB A', 'cam8': 'video8 USB B'}, {}, 160, 120)
    assert grid.shape == (120, 320, 3)
    np.testing.assert_array_equal(grid[100, 80], [0, 0, 255])
    np.testing.assert_array_equal(grid[100, 240], [0, 255, 0])
    assert np.count_nonzero(grid[:50, :160]) > 0
    assert np.count_nonzero(grid[:50, 160:]) > 0


@pytest.mark.parametrize('failure', [RuntimeError, KeyboardInterrupt])
def test_open_failure_or_interrupt_releases_cameras(monkeypatch, failure):
    created = []
    class Camera:
        def __init__(self, config):
            self.name = config.name; self.closed = False; created.append(self)
        def open(self):
            if self.name == 'bad': raise failure('failed')
        def close(self): self.closed = True
    monkeypatch.setattr(preview, 'OpenCVCamera', Camera)
    configs = [preview.OpenCVCameraConfig(name=n) for n in ('good', 'bad')]
    with pytest.raises(failure):
        preview._open_previews(configs, tolerate_errors=failure is KeyboardInterrupt)
    assert all(c.closed for c in created)


def test_all_mode_displays_other_inputs_when_one_is_busy(monkeypatch):
    created, shown = [], []
    class Camera:
        def __init__(self, config):
            self.name = config.name; self.closed = False; created.append(self)
        def open(self):
            if self.name == 'cam2': raise RuntimeError('busy')
        def read(self): return np.full((240, 320, 3), 50, dtype=np.uint8)
        def close(self): self.closed = True
    monkeypatch.setattr(preview, 'OpenCVCamera', Camera)
    monkeypatch.setattr(preview, 'list_available_cameras', lambda: [(2, 640, 480), (8, 640, 480)])
    monkeypatch.setattr(preview, '_linux_video_devices', lambda: {})
    monkeypatch.setattr(preview, '_capture_source', lambda i: i)
    monkeypatch.setattr(cv2, 'imshow', lambda title, frame: shown.append(frame))
    monkeypatch.setattr(cv2, 'waitKey', lambda _: ord('q'))
    monkeypatch.setattr(cv2, 'destroyAllWindows', lambda: None)
    assert preview.main(['--all']) == 0
    assert len(shown) == 1 and shown[0].shape == (300, 800, 3)
    assert all(c.closed for c in created)


def test_all_mode_exits_without_opening_windows_when_no_inputs_exist(monkeypatch):
    monkeypatch.setattr(preview, 'list_available_cameras', lambda: [])
    monkeypatch.setattr(cv2, 'imshow', lambda *a: pytest.fail('No camera, no window'))
    assert preview.main(['--all']) == 1


def test_focus_inspection_only_reads_v4l2_controls(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(preview.shutil, 'which', lambda _: '/usr/bin/v4l2-ctl')
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout='focus_automatic_continuous (bool): value=0', stderr='')
    monkeypatch.setattr(preview.subprocess, 'run', run)
    preview._print_camera_controls('wrist', '/dev/video8')
    args, kwargs = calls[0]
    assert args == ['/usr/bin/v4l2-ctl', '--device', '/dev/video8', '--list-ctrls']
    assert kwargs['timeout'] == 5
    assert 'focus_automatic_continuous' in capsys.readouterr().out


def test_control_inspection_handles_missing_tool_and_unavailable_device(monkeypatch, capsys):
    monkeypatch.setattr(preview.shutil, 'which', lambda _: None)
    preview._print_camera_controls('bottom', '/dev/video3')
    assert 'not installed' in capsys.readouterr().out
    monkeypatch.setattr(preview.shutil, 'which', lambda _: '/usr/bin/v4l2-ctl')
    monkeypatch.setattr(preview.subprocess, 'run', lambda *a, **kw:
                        SimpleNamespace(stdout='', stderr='Cannot open device'))
    preview._print_camera_controls('bottom', '/dev/video3')
    assert 'Cannot open device' in capsys.readouterr().out
