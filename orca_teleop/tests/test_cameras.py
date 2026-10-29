"""Tests for orca_teleop.cameras.

Spec parsing runs anywhere; the OpenCVCamera tests fake ``cv2.VideoCapture`` so
they exercise the threading/frame-processing plumbing without real cameras.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from orca_teleop.cameras import (
    CameraManager,
    OpenCVCamera,
    OpenCVCameraConfig,
    list_available_cameras,
    parse_camera_spec,
    with_recording_defaults,
)

cv2 = pytest.importorskip("cv2")


def test_parse_spec_name_only():
    cfg = parse_camera_spec("front")
    assert cfg == OpenCVCameraConfig(name="front", index=0)


def test_parse_spec_name_index():
    cfg = parse_camera_spec("iphone:1")
    assert cfg.name == "iphone"
    assert cfg.index == 1
    assert cfg.width is None and cfg.height is None and cfg.fps is None


def test_parse_spec_with_resolution():
    cfg = parse_camera_spec("iphone:1:1280x720")
    assert (cfg.index, cfg.width, cfg.height) == (1, 1280, 720)
    assert cfg.fps is None


def test_parse_spec_with_resolution_and_fps():
    cfg = parse_camera_spec("iphone:2:1920x1080@30")
    assert (cfg.index, cfg.width, cfg.height, cfg.fps) == (2, 1920, 1080, 30.0)


def test_parse_spec_index_and_fps_no_resolution():
    cfg = parse_camera_spec("front:0@60")
    assert (cfg.index, cfg.fps) == (0, 60.0)
    assert cfg.width is None and cfg.height is None


def test_parse_spec_missing_name_raises():
    with pytest.raises(ValueError, match="missing a name"):
        parse_camera_spec(":0")


def test_parse_physical_usb_camera_spec():
    cfg = parse_camera_spec('front:usb=1-1.1:640x480@30')
    assert cfg.index == 'usb=1-1.1'
    assert (cfg.width, cfg.height, cfg.fps) == (640, 480, 30)
    assert with_recording_defaults(cfg).index == 'usb=1-1.1'
    with pytest.raises(ValueError, match='USB path'):
        parse_camera_spec('front:usb=')


def test_parse_compressed_usb_camera_spec_preserves_capture_format():
    cfg = parse_camera_spec('bottom:usb=1-1.1:640x480:MJPG@30')
    assert cfg.fourcc == 'MJPG'
    assert with_recording_defaults(cfg).fourcc == 'MJPG'
    with pytest.raises(ValueError, match='FOURCC'):
        parse_camera_spec('bottom:0:640x480:JPEGX@30')


def test_serial_selector_parse_and_resolution_survives_usb_port_change(monkeypatch):
    import orca_teleop.cameras as cameras

    cfg = parse_camera_spec('rs:serial=261643069030:640x480@30')
    assert with_recording_defaults(cfg).index == 'serial=261643069030'
    with pytest.raises(ValueError, match='serial'):
        parse_camera_spec('rs:serial=')
    monkeypatch.setattr(cameras, '_IS_MACOS', False)
    monkeypatch.setattr(cameras.platform, 'system', lambda: 'Linux')
    nodes = {
        0: dict(serial='261643069030', usb_path='2-1.2', interface='Depth', stream_index='0'),
        4: dict(serial='261643069030', usb_path='2-1.2', interface='RGB', stream_index='0'),
        5: dict(serial='261643069030', usb_path='2-1.2', interface='RGB', stream_index='1'),
    }
    monkeypatch.setattr(cameras, '_linux_video_devices', lambda: nodes)
    assert cameras._capture_source(cfg.index) == '/dev/video4'
    nodes = {i + 4: dict(info, usb_path='2-2') for i, info in nodes.items()}
    assert cameras._capture_source(cfg.index) == '/dev/video8'
    nodes[10] = dict(nodes[8], usb_path='1-1')
    with pytest.raises(ValueError, match='multiple cameras'):
        cameras._capture_source(cfg.index)
    nodes.clear()
    with pytest.raises(ValueError, match='found 0'):
        cameras._capture_source(cfg.index)


def test_serial_mapping_change_during_open_releases_device(monkeypatch):
    import orca_teleop.cameras as cameras

    monkeypatch.setattr(cameras, '_IS_MACOS', False)
    monkeypatch.setattr(cameras.platform, 'system', lambda: 'Linux')
    nodes = {8: dict(serial='rs', usb_path='2-2', interface='RGB', stream_index='0')}
    monkeypatch.setattr(cameras, '_linux_video_devices', lambda: nodes)
    created = []
    def factory(source, *args):
        cap = _FakeCapture(source)
        created.append(cap)
        nodes[8] = dict(nodes[8], serial='different-camera')
        return cap
    monkeypatch.setattr(cv2, 'VideoCapture', factory)
    cam = OpenCVCamera(OpenCVCameraConfig(name='rs', index='serial=rs'))
    with pytest.raises(RuntimeError, match='mapping changed'):
        cam.open()
    assert created[0].released


def test_parse_spec_bad_resolution_raises():
    with pytest.raises(ValueError, match="invalid resolution"):
        parse_camera_spec("cam:0:1280")


def test_with_recording_defaults_fills_missing_resolution_and_fps():
    cfg = with_recording_defaults(parse_camera_spec("iphone:1"))
    assert (cfg.width, cfg.height, cfg.fps) == (640, 480, 30.0)


def test_with_recording_defaults_preserves_explicit_values():
    cfg = with_recording_defaults(parse_camera_spec("iphone:1:1280x720@60"))
    assert (cfg.width, cfg.height, cfg.fps) == (1280, 720, 60.0)


class _FakeCapture:
    """Minimal stand-in for cv2.VideoCapture yielding solid-color BGR frames."""

    def __init__(self, index, *_args, width=640, height=480, opened=True, frames_ok=True):
        self.index = index
        self._width = width
        self._height = height
        self._opened = opened
        self._frames_ok = frames_ok
        self.released = False

    def isOpened(self):
        return self._opened

    def set(self, _prop, _value):
        return True

    def read(self):
        if not self._frames_ok:
            return False, None
        # Distinct per-channel values so BGR->RGB conversion is observable.
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        frame[..., 0] = 10  # B
        frame[..., 1] = 20  # G
        frame[..., 2] = 30  # R
        return True, frame

    def release(self):
        self.released = True


def _patch_capture(monkeypatch, **kwargs):
    created: list[_FakeCapture] = []

    def factory(index, *args, **_kw):
        cap = _FakeCapture(index, *args, **kwargs)
        created.append(cap)
        return cap

    monkeypatch.setattr(cv2, "VideoCapture", factory)
    return created


@pytest.mark.parametrize('accepted', [True, False])
def test_explicit_mjpeg_capture_verifies_negotiation_and_closes_on_fallback(monkeypatch, accepted):
    calls, created = [], []
    class FormatCapture(_FakeCapture):
        def set(self, prop, value):
            calls.append((prop, value))
            return True
        def get(self, prop):
            assert prop == cv2.CAP_PROP_FOURCC
            return cv2.VideoWriter_fourcc(*('MJPG' if accepted else 'YUYV'))
    def factory(source, *args):
        capture = FormatCapture(source); created.append(capture); return capture
    monkeypatch.setattr(cv2, 'VideoCapture', factory)
    cfg = parse_camera_spec('bottom:0:640x480:MJPG@30')
    cam = OpenCVCamera(cfg)
    try:
        if accepted:
            assert cam.open() == (480, 640, 3)
            assert cam.read().dtype == np.uint8
        else:
            with pytest.raises(RuntimeError, match='did not accept pixel format'):
                cam.open()
    finally:
        cam.close()
    assert calls[0] == (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    assert created[0].released


def test_camera_open_read_close_converts_to_rgb(monkeypatch):
    _patch_capture(monkeypatch, width=640, height=480)
    cam = OpenCVCamera(OpenCVCameraConfig(name="cam", index=0))
    try:
        shape = cam.open()
        assert shape == (480, 640, 3)
        frame = cam.read()
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8
        # BGR (10,20,30) -> RGB (30,20,10)
        assert tuple(int(x) for x in frame[0, 0]) == (30, 20, 10)
    finally:
        cam.close()
    assert cam._thread is None


def test_camera_resizes_to_configured_resolution(monkeypatch):
    _patch_capture(monkeypatch, width=1920, height=1080)
    cam = OpenCVCamera(OpenCVCameraConfig(name="iphone", index=1, width=1280, height=720))
    try:
        shape = cam.open()
        assert shape == (720, 1280, 3)
        assert cam.read().shape == (720, 1280, 3)
    finally:
        cam.close()


def test_camera_rotate_swaps_dims(monkeypatch):
    _patch_capture(monkeypatch, width=640, height=480)
    cam = OpenCVCamera(OpenCVCameraConfig(name="cam", index=0, rotate=90))
    try:
        shape = cam.open()
        assert shape == (640, 480, 3)
    finally:
        cam.close()


def test_camera_open_failure_when_not_opened(monkeypatch):
    _patch_capture(monkeypatch, opened=False)
    cam = OpenCVCamera(OpenCVCameraConfig(name="cam", index=9))
    with pytest.raises(RuntimeError, match="Failed to open camera"):
        cam.open()


def test_camera_open_failure_when_no_frames(monkeypatch):
    _patch_capture(monkeypatch, frames_ok=False)
    cam = OpenCVCamera(OpenCVCameraConfig(name="cam", index=0, warmup_frames=2))
    with pytest.raises(RuntimeError, match="returned no"):
        cam.open()


def test_camera_close_is_idempotent(monkeypatch):
    _patch_capture(monkeypatch)
    cam = OpenCVCamera(OpenCVCameraConfig(name="cam", index=0))
    cam.open()
    cam.close()
    cam.close()  # must not raise


def test_camera_does_not_leak_reader_thread(monkeypatch):
    before = {t.ident for t in threading.enumerate()}
    _patch_capture(monkeypatch)
    cam = OpenCVCamera(OpenCVCameraConfig(name="cam", index=0))
    cam.open()
    cam.close()
    leaked = [t for t in threading.enumerate() if t.ident not in before and t.is_alive()]
    for t in leaked:
        t.join(timeout=1.0)
    assert not [t for t in leaked if t.is_alive()]


def test_list_available_cameras(monkeypatch):
    def factory(index, *_args, **_kw):
        # Only indices 0 and 2 "exist".
        return _FakeCapture(index, opened=int(str(index).removeprefix('/dev/video')) in (0, 2))

    monkeypatch.setattr(cv2, "VideoCapture", factory)
    found = list_available_cameras(max_index=4)
    assert [idx for idx, _w, _h in found] == [0, 2]
    for _idx, w, h in found:
        assert (w, h) == (640, 480)


def test_linux_discovery_reaches_camera_eight_after_missing_inputs(monkeypatch):
    import orca_teleop.cameras as cameras

    attempted = []
    def factory(index, *_args, **_kw):
        attempted.append(index)
        return _FakeCapture(index, opened=index == '/dev/video8')
    monkeypatch.setattr(cameras, '_IS_MACOS', False)
    monkeypatch.setattr(cameras.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(cameras, '_linux_video_devices', lambda: {i: {} for i in (0, 1, 8, 9)})
    monkeypatch.setattr(cv2, 'VideoCapture', factory)
    assert [i for i, _, _ in list_available_cameras()] == [8]
    assert attempted == ['/dev/video0', '/dev/video1', '/dev/video8', '/dev/video9']


def test_linux_listing_identifies_interfaces_and_distinct_usb_paths(monkeypatch, capsys):
    import orca_teleop.cameras as cameras

    devices = {
        4: dict(interface='RealSense RGB', usb_path='2-1.2'),
        5: dict(interface='RealSense RGB', usb_path='2-1.2'),
        6: dict(interface='HD Camera', usb_path='1-1.1'),
        8: dict(interface='HD Camera', usb_path='1-1.3'),
    }
    monkeypatch.setattr(cameras, '_IS_MACOS', False)
    monkeypatch.setattr(cameras, '_linux_video_devices', lambda: devices)
    monkeypatch.setattr(cv2, 'VideoCapture', lambda index, *a: _FakeCapture(
        index, opened=int(str(index).removeprefix('/dev/video')) != 5))
    cameras.print_available_cameras()
    output = capsys.readouterr().out
    assert '3 usable video input(s)' in output
    assert 'index 8:' in output
    assert 'RealSense RGB [USB 2-1.2]' in output
    assert 'HD Camera [USB 1-1.1]' in output
    assert 'HD Camera [USB 1-1.3]' in output


def test_linux_sysfs_metadata_groups_interfaces_by_physical_usb_device(monkeypatch, tmp_path):
    import orca_teleop.cameras as cameras

    video_class = tmp_path/'class'
    for number, usb, interface in ((0, '2-1.2', 'Depth'), (4, '2-1.2', 'RGB'),
                                   (6, '1-1.1', 'HD Camera'), (8, '1-1.3', 'HD Camera')):
        parent = tmp_path/'devices'/usb
        parent.mkdir(parents=True, exist_ok=True)
        (parent/'idVendor').write_text('8086')
        (parent/'serial').write_text('test-serial')
        device = parent/f'{usb}:{interface}'
        device.mkdir()
        (device/'interface').write_text(interface)
        node = video_class/f'video{number}'
        node.mkdir(parents=True)
        (node/'name').write_text(interface)
        (node/'index').write_text('0')
        (node/'device').symlink_to(device, target_is_directory=True)
    monkeypatch.setattr(cameras.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(cameras, 'Path', lambda _: video_class)
    devices = cameras._linux_video_devices()
    assert devices[0]['usb_path'] == devices[4]['usb_path']
    assert devices[6]['usb_path'] != devices[8]['usb_path']
    assert devices[4]['interface'] == 'RGB'
    assert devices[4]['stream_index'] == '0'
    assert devices[4]['serial'] == 'test-serial'


def test_usb_selection_uses_rgb_and_distinct_ports_despite_identical_names(monkeypatch):
    import orca_teleop.cameras as cameras

    nodes = {
        0: dict(interface='RealSense Depth', usb_path='2-1.2', stream_index='0'),
        4: dict(interface='RealSense RGB', usb_path='2-1.2', stream_index='0'),
        5: dict(interface='RealSense RGB', usb_path='2-1.2', stream_index='1'),
        6: dict(interface='HD Camera', usb_path='1-1.1', stream_index='0'),
        7: dict(interface='HD Camera', usb_path='1-1.1', stream_index='1'),
        8: dict(interface='HD Camera', usb_path='1-1.3', stream_index='0'),
    }
    monkeypatch.setattr(cameras, '_IS_MACOS', False)
    monkeypatch.setattr(cameras.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(cameras, '_linux_video_devices', lambda: nodes)
    assert cameras._capture_source('usb=2-1.2') == '/dev/video4'
    assert cameras._capture_source('usb=1-1.1') == '/dev/video6'
    assert cameras._capture_source('usb=1-1.3') == '/dev/video8'
    nodes = {i+10: v for i, v in nodes.items()}
    assert cameras._capture_source('usb=2-1.2') == '/dev/video14'
    assert cameras._capture_source('usb=1-1.3') == '/dev/video18'
    nodes = {i: v for i, v in nodes.items() if 'RGB' not in v['interface']}
    with pytest.raises(ValueError, match='RGB/color'):
        cameras._capture_source('usb=2-1.2')


def test_linux_open_uses_exact_v4l2_path_with_no_backend_fallback(monkeypatch):
    import orca_teleop.cameras as cameras

    monkeypatch.setattr(cameras, '_IS_MACOS', False)
    monkeypatch.setattr(cameras.platform, 'system', lambda: 'Linux')
    calls = []
    def factory(source, backend):
        calls.append((source, backend))
        return _FakeCapture(source)
    monkeypatch.setattr(cv2, 'VideoCapture', factory)
    cam = OpenCVCamera(OpenCVCameraConfig(name='exact', index=7))
    try:
        cam.open()
    finally:
        cam.close()
    assert calls == [('/dev/video7', cv2.CAP_V4L2)]


def test_usb_mapping_change_during_open_closes_device(monkeypatch):
    import orca_teleop.cameras as cameras

    monkeypatch.setattr(cameras, '_IS_MACOS', False)
    monkeypatch.setattr(cameras.platform, 'system', lambda: 'Linux')
    snapshots = iter([
        {4: dict(interface='RGB', usb_path='2-1.2', stream_index='0')},
        {4: dict(interface='HD Camera', usb_path='1-1.1', stream_index='0')},
    ])
    monkeypatch.setattr(cameras, '_linux_video_devices', lambda: next(snapshots))
    created = _patch_capture(monkeypatch)
    cam = OpenCVCamera(OpenCVCameraConfig(name='rs', index='usb=2-1.2'))
    with pytest.raises(RuntimeError, match='mapping changed'):
        cam.open()
    assert created[0].released and cam._thread is None


def test_macos_still_uses_native_numeric_indices(monkeypatch):
    import orca_teleop.cameras as cameras

    monkeypatch.setattr(cameras, '_IS_MACOS', True)
    assert cameras._capture_source(3) == 3
    assert cameras._open_backend() == cv2.CAP_AVFOUNDATION
    with pytest.raises(ValueError, match='require Linux'):
        cameras._capture_source('usb=1-1.1')


def test_three_usb_camera_views_retain_their_own_frames(monkeypatch):
    import orca_teleop.cameras as cameras

    nodes = {
        4: dict(interface='RGB', usb_path='2-1.2', stream_index='0'),
        6: dict(interface='HD Camera', usb_path='1-1.1', stream_index='0'),
        8: dict(interface='HD Camera', usb_path='1-1.3', stream_index='0'),
    }
    monkeypatch.setattr(cameras, '_IS_MACOS', False)
    monkeypatch.setattr(cameras.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(cameras, '_linux_video_devices', lambda: nodes)
    class DistinctCapture(_FakeCapture):
        def read(self):
            ok, frame = super().read()
            frame[..., 0] = int(self.index.removeprefix('/dev/video'))
            return ok, frame
    monkeypatch.setattr(cv2, 'VideoCapture', DistinctCapture)
    specs = ('rs:usb=2-1.2', 'a:usb=1-1.1', 'b:usb=1-1.3')
    manager = CameraManager([parse_camera_spec(s) for s in specs])
    try:
        manager.open()
        images = manager.capture()
        assert [int(images[n][0, 0, 2]) for n in ('rs', 'a', 'b')] == [4, 6, 8]
    finally:
        manager.close()


def test_camera_manager_open_capture_close(monkeypatch):
    _patch_capture(monkeypatch, width=640, height=480)
    manager = CameraManager(
        [OpenCVCameraConfig(name="front", index=0), OpenCVCameraConfig(name="side", index=1)]
    )
    try:
        shapes = manager.open()
        assert shapes == {"front": (480, 640, 3), "side": (480, 640, 3)}
        assert manager.shapes == shapes
        assert len(manager) == 2
        assert manager.names == ["front", "side"]
        frames = manager.capture()
        assert set(frames) == {"front", "side"}
        assert frames["front"].shape == (480, 640, 3)
    finally:
        manager.close()
    assert len(manager) == 0
    assert manager.shapes == {}


def test_camera_manager_empty():
    manager = CameraManager([])
    assert manager.open() == {}
    assert manager.capture() == {}
    manager.ensure_live()
    manager.close()


def test_camera_manager_ensure_live_requires_fresh_frames(monkeypatch):
    _patch_capture(monkeypatch, width=640, height=480)
    manager = CameraManager([OpenCVCameraConfig(name="front", index=0)])
    try:
        manager.open()
        manager.ensure_live(samples=3, interval_s=0.0)
    finally:
        manager.close()


def test_camera_manager_ensure_live_raises_when_stale(monkeypatch):
    _patch_capture(monkeypatch, width=640, height=480)
    manager = CameraManager([OpenCVCameraConfig(name="front", index=0)])
    try:
        manager.open()
        # Freeze the freshness clock so the next read looks stale.
        cam = manager._cameras[0]
        cam._last_ok_ts = 0.0
        with pytest.raises(RuntimeError, match="no fresh frame"):
            manager.ensure_live(samples=1)
    finally:
        manager.close()


def test_camera_manager_rejects_duplicate_names():
    with pytest.raises(ValueError, match="Duplicate camera name"):
        CameraManager(
            [OpenCVCameraConfig(name="cam", index=0), OpenCVCameraConfig(name="cam", index=1)]
        )


def test_camera_manager_rejects_two_names_for_the_same_usb_camera():
    with pytest.raises(ValueError, match='Duplicate physical USB'):
        CameraManager([OpenCVCameraConfig(name='first', index='usb=1-1.1'),
                       OpenCVCameraConfig(name='second', index='usb=1-1.1')])


def test_camera_manager_open_failure_releases_opened(monkeypatch):
    # Index 0 opens fine; index 1 fails to open -> manager must release index 0.
    created: list[_FakeCapture] = []

    def factory(index, *_args, **_kw):
        cap = _FakeCapture(index, opened=(int(str(index).removeprefix('/dev/video')) == 0))
        created.append(cap)
        return cap

    monkeypatch.setattr(cv2, "VideoCapture", factory)
    manager = CameraManager(
        [OpenCVCameraConfig(name="ok", index=0), OpenCVCameraConfig(name="bad", index=1)]
    )
    with pytest.raises(RuntimeError, match="Failed to open camera"):
        manager.open()
    assert len(manager) == 0
    # The successfully-opened capture must have been released during rollback.
    assert created[0].released is True
