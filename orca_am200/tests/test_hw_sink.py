"""Hardware sink compatibility helpers avoid per-motor command flooding."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from orca_teleop.am200.hw_sink import (
    Am200HardwareSink,
    _check_bus_ports,
    _enable_sync_hand_writes,
)


class _FakeMap:
    dof = 4
    arm_slice = slice(0, 2)
    hand_slice = slice(2, 4)


def test_enable_sync_hand_writes_replaces_legacy_individual_writer():
    calls = []

    class Client:
        def write_desired_pos(self, *_args, **_kwargs):
            raise AssertionError("legacy individual writer must not be used")

        def write_positions_sync(self, motor_ids, positions, **kwargs):
            calls.append((list(motor_ids), np.asarray(positions), kwargs))

    client = Client()
    _enable_sync_hand_writes(SimpleNamespace(_motor_client=client))
    client.write_desired_pos([1, 17], np.array([0.1, 0.2]), speed=60, torque=500)

    assert calls[0][0] == [1, 17]
    np.testing.assert_allclose(calls[0][1], [0.1, 0.2])
    assert calls[0][2] == {"speed": 60, "torque": 500}


def test_enable_sync_hand_writes_rejects_unsupported_core():
    hand = SimpleNamespace(_motor_client=SimpleNamespace())
    with pytest.raises(RuntimeError, match="lacks Feetech sync-write"):
        _enable_sync_hand_writes(hand)


def _feedback_sink(enabled, read_ok=True):
    sink = Am200HardwareSink(enable_arm=False, hand_feedback_debug=enabled)
    writes, reads = [], []
    sink._target = np.array([45.0])
    sink._map = SimpleNamespace(hand_slice=slice(0, 1),
                                hand_to_physical=lambda q: {'thumb_dip': float(q[0])})

    def read():
        reads.append(True)
        return SimpleNamespace(data={'thumb_dip': 23.0})

    sink._hand = SimpleNamespace(
        set_joint_positions=lambda target, **kw: writes.append((dict(target), kw)),
        get_joint_position=read,
        config=SimpleNamespace(joint_to_motor_map={'thumb_dip': 15}),
        _motor_client=SimpleNamespace(last_read_ok=read_ok),
    )
    return sink, writes, reads


def test_feedback_debug_preserves_writes_and_throttles_reads(monkeypatch, caplog):
    sink, writes, reads = _feedback_sink(True)
    with caplog.at_level('INFO', logger='orca_teleop.am200.hw_sink'):
        for now in (0.0, .1, 1.1):
            monkeypatch.setattr('orca_teleop.am200.hw_sink.time.monotonic', lambda: now)
            sink._apply()
    assert len(writes) == 3
    assert len(reads) == 2
    assert all(w[0] == {'thumb_dip': 45.0} for w in writes)
    assert 'cmd_deg=45.00' in caplog.text
    assert 'encoder_joint_deg=23.00' in caplog.text
    assert 'error_deg=22.00' in caplog.text


def test_feedback_disabled_does_not_add_bus_reads():
    sink, writes, reads = _feedback_sink(False)
    sink._apply()
    assert len(writes) == 1
    assert not reads


def test_failed_feedback_does_not_report_cached_angles(caplog):
    sink, writes, reads = _feedback_sink(True, read_ok=False)
    with caplog.at_level('INFO', logger='orca_teleop.am200.hw_sink'):
        sink._apply()
    assert len(writes) == 1
    assert len(reads) == 1
    assert 'cached values rejected' in caplog.text
    assert 'encoder_joint_deg=' not in caplog.text


def _full_feedback_sink():
    sink, writes, reads = _feedback_sink(True)
    mapping = {
        'thumb_mcp': 13, 'thumb_abd': 14, 'thumb_pip': 16, 'thumb_dip': 15,
        'index_abd': -3, 'index_mcp': 2, 'index_pip': 1,
        'middle_abd': -4, 'middle_mcp': 9, 'middle_pip': 8,
        'ring_abd': 12, 'ring_mcp': 10, 'ring_pip': 11,
        'pinky_abd': -5, 'pinky_mcp': 7, 'pinky_pip': 6,
    }
    command = {joint: 20.0 for joint in mapping}
    measured = {joint: 17.0 for joint in mapping}
    sink._map.hand_to_physical = lambda _: dict(command)
    sink._hand.config.joint_to_motor_map = mapping

    def read():
        reads.append(True)
        return SimpleNamespace(data=measured)

    sink._hand.get_joint_position = read
    return sink, writes, reads, command, measured


def test_all_finger_feedback_reuses_one_read_and_preserves_commands(monkeypatch, caplog):
    sink, writes, reads, command, _ = _full_feedback_sink()
    with caplog.at_level('INFO', logger='orca_teleop.am200.hw_sink'):
        for now in (0.0, .1, 1.1):
            monkeypatch.setattr('orca_teleop.am200.hw_sink.time.monotonic', lambda: now)
            sink._apply()
    assert len(reads) == 2
    assert len(writes) == 3
    assert all(target == command for target, _ in writes)
    for finger in ('thumb', 'index', 'middle', 'ring', 'pinky'):
        records = [r.message for r in caplog.records if f'finger={finger} |' in r.message]
        assert len(records) == 2
        assert 'encoder_joint_deg=[17.0' in records[0]
        assert 'error_deg=[3.0' in records[0]
    assert 'motor_ids=[12, 10, 11]' in caplog.text
    assert 'motor_ids=[13, 14, 16, 15]' in caplog.text
    assert 'motor_ids=[-5, 7, 6]' in caplog.text


@pytest.mark.parametrize('bad_angle', [None, float('nan')])
def test_bad_finger_feedback_does_not_fabricate_angles_or_stop_writes(caplog, bad_angle):
    sink, writes, reads, _, measured = _full_feedback_sink()
    measured['ring_mcp'] = bad_angle
    with caplog.at_level('INFO', logger='orca_teleop.am200.hw_sink'):
        sink._apply()
    assert len(writes) == len(reads) == 1
    assert 'ring has missing/nonfinite angles' in caplog.text
    assert 'finger=ring |' not in caplog.text
    assert 'finger=pinky |' in caplog.text


def test_go_home_glides_instead_of_snapping(monkeypatch):
    monkeypatch.setattr("orca_teleop.am200.hw_sink.time.sleep", lambda _s: None)
    sink = Am200HardwareSink.__new__(Am200HardwareSink)
    sink._map = _FakeMap()
    sink._hand = None
    sink._target = np.array([0.6, -0.6, 0.0, 0.0])
    sent = []
    sink._apply = lambda: sent.append(sink._target.copy())

    sink.go_home(duration_s=1.0, hz=30.0)

    arm = np.stack(sent)[:, :2]
    assert len(sent) == 30
    np.testing.assert_allclose(arm[-1], [0.0, 0.0], atol=1e-12)
    assert np.max(np.abs(np.diff(np.vstack([[0.6, -0.6], arm]), axis=0))) < 0.05


def test_bus_ports_must_be_distinct_devices(tmp_path):
    device = tmp_path / "ttyACM1"
    device.touch()
    alias = tmp_path / "by-id-hand"
    alias.symlink_to(device)
    with pytest.raises(RuntimeError, match="both resolve to"):
        _check_bus_ports(str(alias), str(device))


def test_missing_bus_port_is_refused_before_auto_detect(tmp_path):
    present = tmp_path / "ttyACM1"
    present.touch()
    with pytest.raises(RuntimeError, match="arm serial port .* is not connected"):
        _check_bus_ports(str(present), str(tmp_path / "ttyACM0"))
    _check_bus_ports(str(present), None)
