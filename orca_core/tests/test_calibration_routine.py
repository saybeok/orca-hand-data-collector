"""Failure-path tests for calibration routine internals: guarded motor reads,
offset-calibration failures, and the atomic YAML persistence helper."""

import os
import shutil
import threading

import numpy as np
import pytest
import yaml

from orca_core import MockOrcaHand
from orca_core.maintenance import calibration_routine
from orca_core.utils.utils import read_yaml, write_yaml_atomic

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO_ROOT, "orca_core", "models", "v2", "orcahand-right")


@pytest.fixture
def calib_dir(tmp_path):
    shutil.copy(os.path.join(MODEL_DIR, "config.yaml"), tmp_path / "config.yaml")
    return tmp_path


@pytest.fixture
def connected_hand(calib_dir):
    hand = MockOrcaHand(config_path=str(calib_dir / "config.yaml"))
    success, msg = hand.connect()
    assert success, msg
    try:
        yield hand
    finally:
        hand.disconnect()


# ---------------------------------------------------------------------------
# _read_motor_pos_checked
# ---------------------------------------------------------------------------


def test_partial_calibration_mode_change_only_targets_selected_motors(
    connected_hand, monkeypatch
):
    calls = []
    original = connected_hand.set_control_mode

    def record_mode(mode, motor_ids=None):
        calls.append(motor_ids)
        return original(mode, motor_ids=motor_ids)

    monkeypatch.setattr(connected_hand, "set_control_mode", record_mode)
    # Stop before driving: only exercise preparation and the first stop check.
    calibration_routine.run_calibration(
        connected_hand, joints=["ring_mcp", "pinky_mcp"],
        should_stop=lambda: True, persist=False,
    )
    expected = sorted(connected_hand.config.joint_to_motor_map[j]
                      for j in ["ring_mcp", "pinky_mcp"])
    assert calls == [expected]


@pytest.fixture
def v1_hand(tmp_path):
    """Use this unit's real joint map with an entirely simulated motor bus."""
    from pathlib import Path
    from orca_core import OrcaJointPositions
    root = Path(REPO_ROOT).parent
    shutil.copy(root / "configs/orcahand-right-v1-feetech/config.yaml", tmp_path / "config.yaml")
    ordering = OrcaJointPositions._default_joint_ids
    hand = MockOrcaHand(config_path=str(tmp_path / "config.yaml"))
    assert hand.connect()[0]
    try:
        yield hand
    finally:
        hand.disconnect()
        OrcaJointPositions._default_joint_ids = ordering


@pytest.mark.parametrize("include_abd", [False, True])
def test_v1_middle_flexion_never_drives_abd_even_after_abd_calibration(v1_hand, monkeypatch, include_abd):
    hand = v1_hand
    monkeypatch.setattr(calibration_routine.time, "sleep", lambda _: None)
    monkeypatch.setattr(type(hand.motor_client), "requires_offset_calibration", property(lambda _: False))
    stage = {}; writes = []
    original = hand.motor_client.write_desired_pos
    def track(motors, positions):
        if stage.get("joint") in ("middle_mcp", "middle_pip"):
            assert not hand.motor_client._torque_enabled[4], "ABD is powered during flexion"
            assert set(motors) <= {stage["motor"]}, "Unrelated motor receives a goal"
            writes.append((stage["joint"], tuple(motors)))
        return original(motors, positions)
    def progress(event):
        if event['event'] == 'step_started':
            joint = next(iter(event['joints']))
            stage.update(joint=joint, motor=hand.config.joint_to_motor_map[joint])
    monkeypatch.setattr(hand.motor_client, 'write_desired_pos', track)
    joints = ['middle_mcp', 'middle_pip']
    if include_abd:
        joints.insert(0, 'middle_abd')
    result = calibration_routine.run_calibration(hand, joints=joints, progress_callback=progress, persist=False)
    assert result is not None
    assert {joint for joint, _ in writes} == {'middle_mcp', 'middle_pip'}


@pytest.mark.parametrize('fail_on', [1, 2, 3])
def test_unreleased_middle_abd_aborts_before_flexion_drive(v1_hand, monkeypatch, fail_on):
    hand = v1_hand
    hand.enable_torque([4])  # Prior target is still powered when the release fails.
    original = hand.disable_torque
    count = 0
    def failed_release(motor_ids=None):
        nonlocal count
        count += 1
        ids = hand.config.motor_ids if motor_ids is None else motor_ids
        if count == fail_on:
            original([mid for mid in ids if mid != 4])
            return [4]
        return original(ids)
    monkeypatch.setattr(hand, 'disable_torque', failed_release)
    monkeypatch.setattr(hand.motor_client, 'write_desired_pos', lambda *a, **k: pytest.fail('must not drive'))
    events = []
    with pytest.raises(RuntimeError, match='4'):
        calibration_routine.run_calibration(hand, joints=['middle_mcp'], persist=False, progress_callback=events.append)
    assert any(e['event'] == 'calibration_isolation_failed' and e['motor_ids'] == [4] for e in events)


class _StubReadHand:
    """Hand stand-in whose bulk read reports stale for the first N attempts."""

    def __init__(self, failing_reads):
        self.reads = 0
        self._motor_lock = threading.RLock()
        stub = self

        class _Client:
            @property
            def last_read_ok(self):
                return stub.reads > failing_reads

        self.motor_client = _Client()

    def get_motor_pos(self):
        self.reads += 1
        return np.array([0.1, 0.2])


def test_read_motor_pos_checked_retries_until_fresh():
    hand = _StubReadHand(failing_reads=2)
    pos = calibration_routine._read_motor_pos_checked(hand, retries=5)
    assert hand.reads == 3
    np.testing.assert_array_equal(pos, np.array([0.1, 0.2]))


def test_read_motor_pos_checked_raises_on_persistent_stale_reads():
    hand = _StubReadHand(failing_reads=99)
    with pytest.raises(RuntimeError, match="no status packets"):
        calibration_routine._read_motor_pos_checked(hand, retries=3)
    assert hand.reads == 3


# ---------------------------------------------------------------------------
# calibrate_offset failure handling
# ---------------------------------------------------------------------------


def _force_offset_calibration(hand, monkeypatch, offset_stub):
    monkeypatch.setattr(
        type(hand.motor_client),
        "requires_offset_calibration",
        property(lambda self: True),
    )
    monkeypatch.setattr(hand.motor_client, "calibrate_offset", offset_stub)


def test_initial_offset_failure_skips_joint_and_keeps_limits(
    connected_hand, calib_dir, monkeypatch
):
    """A failed pre-drive offset calibration must not drive the joint nor
    persist limits for it."""
    import dataclasses as dc

    hand = connected_hand
    pre = {mid: [-0.7, 0.7] for mid in hand.config.motor_ids}
    ratios = {mid: 0.02 for mid in hand.config.motor_ids}
    hand.calibration = dc.replace(
        hand.calibration, motor_limits_dict=pre, joint_to_motor_ratios_dict=ratios
    )

    _force_offset_calibration(hand, monkeypatch, lambda motor_id, upper=True: False)

    target = "thumb_cmc"
    target_motor = hand.config.joint_to_motor_map[target]
    events = []
    hand.calibrate(joints=[target], progress_callback=events.append, persist=True)

    assert hand.motor_limits_dict[target_motor] == pre[target_motor]
    raw = read_yaml(str(calib_dir / "calibration.yaml"))
    assert raw["motor_limits"][target_motor] == pre[target_motor]
    failures = [e for e in events if e["event"] == "offset_calibration_failed"]
    assert failures and all(e["motor"] == target_motor for e in failures)
    assert not any(e["event"] == "limit_recorded" for e in events)
    assert not any(e["event"] == "joint_calibrated" for e in events)


def test_wrist_offset_failure_keeps_wrist_uncalibrated_and_recoverable(
    connected_hand, calib_dir, monkeypatch
):
    """A wrist step skipped by a failed offset calibration must not persist
    wrist_calibrated, and a later run with working offsets must calibrate it."""
    import dataclasses as dc

    hand = connected_hand
    wrist_motor = hand.config.joint_to_motor_map["wrist"]
    limits = {mid: list(l) for mid, l in hand.calibration.motor_limits_dict.items()}
    limits[wrist_motor] = [None, None]
    hand.calibration = dc.replace(
        hand.calibration, motor_limits_dict=limits, calibrated=False
    )
    assert not hand.calibration.wrist_calibrated

    _force_offset_calibration(
        hand, monkeypatch, lambda motor_id, upper=True: motor_id != wrist_motor
    )

    events = []
    hand.calibrate(joints=["wrist"], progress_callback=events.append, persist=True)

    assert not hand.calibration.wrist_calibrated
    raw = read_yaml(str(calib_dir / "calibration.yaml"))
    assert raw["wrist_calibrated"] is False
    assert raw["calibrated"] is False
    assert any(
        e["event"] == "offset_calibration_failed" and e["motor"] == wrist_motor
        for e in events
    )

    monkeypatch.setattr(
        hand.motor_client, "calibrate_offset", lambda motor_id, upper=True: True
    )
    events = []
    hand.calibrate(joints=["wrist"], progress_callback=events.append, persist=True)

    assert not any(e["event"] == "wrist_skipped" for e in events)
    assert hand.calibration.wrist_calibrated
    assert all(l is not None for l in hand.motor_limits_dict[wrist_motor])
    assert read_yaml(str(calib_dir / "calibration.yaml"))["wrist_calibrated"] is True


def test_stale_wrist_flag_without_limits_recalibrates(connected_hand, calib_dir):
    """A persisted wrist_calibrated flag with no stored wrist limits must not
    skip the wrist; the run recalibrates it and leaves a consistent state."""
    import dataclasses as dc

    hand = connected_hand
    wrist_motor = hand.config.joint_to_motor_map["wrist"]
    limits = {mid: list(l) for mid, l in hand.calibration.motor_limits_dict.items()}
    limits[wrist_motor] = [None, None]
    hand.calibration = dc.replace(
        hand.calibration,
        motor_limits_dict=limits,
        wrist_calibrated=True,
        calibrated=False,
    )

    events = []
    hand.calibrate(joints=["wrist"], progress_callback=events.append, persist=True)

    assert not any(e["event"] == "wrist_skipped" for e in events)
    assert hand.calibration.wrist_calibrated
    assert all(l is not None for l in hand.motor_limits_dict[wrist_motor])
    assert read_yaml(str(calib_dir / "calibration.yaml"))["wrist_calibrated"] is True


def test_final_offset_failure_records_no_limit(connected_hand, monkeypatch):
    """A failed post-release offset calibration must not record a limit taken
    in the un-shifted motor frame."""
    hand = connected_hand
    calls = {}

    def offset_stub(motor_id, upper=True):
        calls[motor_id] = calls.get(motor_id, 0) + 1
        return calls[motor_id] == 1

    _force_offset_calibration(hand, monkeypatch, offset_stub)

    target = "thumb_cmc"
    target_motor = hand.config.joint_to_motor_map[target]
    pre = list(hand.motor_limits_dict[target_motor])
    events = []
    hand.calibrate(joints=[target], progress_callback=events.append)

    assert hand.motor_limits_dict[target_motor] == pre
    assert any(e["event"] == "offset_calibration_failed" for e in events)
    assert not any(e["event"] == "limit_recorded" for e in events)
    assert not any(e["event"] == "joint_calibrated" for e in events)


# ---------------------------------------------------------------------------
# Torque-release failure handling
# ---------------------------------------------------------------------------


def test_failed_torque_release_skips_limit_capture(
    connected_hand, calib_dir, monkeypatch
):
    """A pre-capture torque release the bus rejected must not record a limit
    taken under tendon tension; the prior limit stays in place."""
    import dataclasses as dc

    hand = connected_hand
    pre = {mid: [-0.7, 0.7] for mid in hand.config.motor_ids}
    ratios = {mid: 0.02 for mid in hand.config.motor_ids}
    hand.calibration = dc.replace(
        hand.calibration, motor_limits_dict=pre, joint_to_motor_ratios_dict=ratios
    )

    target = "thumb_cmc"
    target_motor = hand.config.joint_to_motor_map[target]
    orig_disable = hand.disable_torque

    def failing_disable(motor_ids=None):
        result = orig_disable(motor_ids)
        if motor_ids == [target_motor]:
            return [target_motor]
        return result

    monkeypatch.setattr(hand, "disable_torque", failing_disable)

    events = []
    hand.calibrate(joints=[target], progress_callback=events.append, persist=True)

    assert hand.motor_limits_dict[target_motor] == pre[target_motor]
    raw = read_yaml(str(calib_dir / "calibration.yaml"))
    assert raw["motor_limits"][target_motor] == pre[target_motor]
    failures = [e for e in events if e["event"] == "torque_release_failed"]
    assert failures and all(e["motor"] == target_motor for e in failures)
    assert not any(e["event"] == "limit_recorded" for e in events)
    assert not any(e["event"] == "joint_calibrated" for e in events)


# ---------------------------------------------------------------------------
# write_yaml_atomic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("span", [0.0, 0.00306797, -0.2, np.nan])
def test_invalid_motor_span_keeps_prior_calibration_and_fails(
    connected_hand, calib_dir, monkeypatch, span,
):
    import dataclasses as dc

    hand = connected_hand
    target = "thumb_cmc"
    motor = hand.config.joint_to_motor_map[target]
    prior = {mid: [-.7,.7] for mid in hand.config.motor_ids}
    hand.calibration = dc.replace(
        hand.calibration, motor_limits_dict=prior,
        joint_to_motor_ratios_dict={mid: .02 for mid in hand.config.motor_ids},
    )
    monkeypatch.setattr(type(hand.motor_client), "requires_offset_calibration", property(lambda _: False))
    lower, upper = 6.08683586, 6.08683586 + span
    captures = iter((lower, upper) if hand.config.joint_inversion_dict[target] else (upper, lower))

    def capture(_hand):
        values = hand.get_motor_pos().copy()
        values[hand.config.motor_id_to_idx_dict[motor]] = next(captures)
        return values

    monkeypatch.setattr(calibration_routine, "_read_motor_pos_checked", capture)
    events = []
    with pytest.raises(ValueError, match="motor span"):
        hand.calibrate(joints=[target], progress_callback=events.append, persist=True)
    assert hand.motor_limits_dict[motor] == prior[motor]
    assert hand.joint_to_motor_ratios_dict[motor] == .02
    saved = read_yaml(str(calib_dir / "calibration.yaml"))
    assert saved["motor_limits"][motor] == prior[motor]
    assert saved["joint_to_motor_ratios"][motor] == .02
    assert any(e["event"] == "calibration_span_invalid" for e in events)
    assert not any(e["event"] in ("joint_calibrated", "calibration_done") for e in events)
    assert not any(hand.motor_client._torque_enabled.values())


def test_invalid_command_frame_is_rejected_before_calibration_motion(connected_hand, monkeypatch):
    hand = connected_hand
    target = "thumb_cmc"
    motor = hand.config.joint_to_motor_map[target]
    validated = []

    def reject(ids, positions):
        validated.append(list(ids))
        raise ValueError("outside command frame")

    monkeypatch.setattr(hand.motor_client, "validate_position_frame", reject, raising=False)
    monkeypatch.setattr(hand, "enable_torque", lambda *a, **kw: pytest.fail("must not enable torque"))
    monkeypatch.setattr(hand, "_set_motor_pos", lambda *a, **kw: pytest.fail("must not move"))
    with pytest.raises(ValueError, match="command frame"):
        hand.calibrate(joints=[target], persist=False)
    assert validated == [[motor]]


def test_write_yaml_atomic_round_trips_and_replaces(tmp_path):
    path = tmp_path / "calibration.yaml"
    write_yaml_atomic(str(path), {"calibrated": False})
    write_yaml_atomic(
        str(path), {"calibrated": True, "motor_limits": {1: [-1.0, 1.0]}}
    )
    assert read_yaml(str(path)) == {
        "calibrated": True,
        "motor_limits": {1: [-1.0, 1.0]},
    }
    assert [p.name for p in tmp_path.iterdir()] == ["calibration.yaml"]


def test_write_yaml_atomic_failed_write_keeps_original(tmp_path, monkeypatch):
    path = tmp_path / "calibration.yaml"
    write_yaml_atomic(str(path), {"calibrated": False})

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(yaml, "dump", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        write_yaml_atomic(str(path), {"calibrated": True})

    assert read_yaml(str(path)) == {"calibrated": False}
    assert [p.name for p in tmp_path.iterdir()] == ["calibration.yaml"]


def test_write_yaml_atomic_preserves_permissions(tmp_path):
    path = tmp_path / "calibration.yaml"
    path.write_text("calibrated: false\n")
    os.chmod(path, 0o664)
    write_yaml_atomic(str(path), {"calibrated": True})
    assert os.stat(path).st_mode & 0o777 == 0o664


def test_write_yaml_atomic_preserves_symlink(tmp_path):
    target = tmp_path / "real_calibration.yaml"
    target.write_text("calibrated: false\n")
    link = tmp_path / "calibration.yaml"
    link.symlink_to(target)

    write_yaml_atomic(str(link), {"calibrated": True})

    assert link.is_symlink(), "symlink was replaced by a regular file"
    assert os.readlink(str(link)) == str(target), "symlink was repointed"
    assert read_yaml(str(target)) == {"calibrated": True}
    assert read_yaml(str(link)) == {"calibrated": True}
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "calibration.yaml",
        "real_calibration.yaml",
    ]


def test_write_yaml_atomic_converts_numpy_arrays(tmp_path):
    path = tmp_path / "data.yaml"
    write_yaml_atomic(
        str(path),
        {"limits": np.array([1.0, 2.0]), "per_motor": {1: np.array([0.5, 0.6])}},
    )
    assert read_yaml(str(path)) == {
        "limits": [1.0, 2.0],
        "per_motor": {1: [0.5, 0.6]},
    }


@pytest.mark.parametrize("span", [.9219, 1.0354371070861816, 2.3807384967803955])
def test_configured_motor_travel_floor_rejects_short_sweeps_before_commit(
        connected_hand, calib_dir, monkeypatch, span):
    import dataclasses as dc
    hand = connected_hand
    joint = "thumb_cmc"
    motor = hand.config.joint_to_motor_map[joint]
    hand.config = dc.replace(hand.config, calibration_min_motor_span_rad={joint: 1.8})
    prior = {mid: [-5., -2.6] for mid in hand.config.motor_ids}
    hand.calibration = dc.replace(
        hand.calibration, motor_limits_dict=prior,
        joint_to_motor_ratios_dict={mid: .02 for mid in hand.config.motor_ids})
    monkeypatch.setattr(type(hand.motor_client), "requires_offset_calibration", property(lambda _: False))
    lower, upper = -4.542117118835449, -4.542117118835449 + span
    values = iter((lower, upper) if hand.config.joint_inversion_dict[joint] else (upper, lower))
    def capture(_):
        positions = hand.get_motor_pos().copy()
        positions[hand.config.motor_id_to_idx_dict[motor]] = next(values)
        return positions
    monkeypatch.setattr(calibration_routine, "_read_motor_pos_checked", capture)
    events = []
    if span < 1.8:
        with pytest.raises(ValueError, match="must exceed 1.800000"):
            hand.calibrate(joints=[joint], progress_callback=events.append, persist=True)
        assert hand.motor_limits_dict[motor] == prior[motor]
        assert read_yaml(str(calib_dir / "calibration.yaml"))["motor_limits"][motor] == prior[motor]
        assert not any(e["event"] in ("joint_calibrated", "calibration_done") for e in events)
        assert not any(hand.motor_client._torque_enabled.values())
    else:
        hand.calibrate(joints=[joint], progress_callback=events.append, persist=True)
        assert hand.motor_limits_dict[motor] == [lower, upper]
        assert any(e["event"] == "calibration_done" for e in events)


def test_loaded_short_span_is_not_considered_calibrated(connected_hand, capsys):
    import dataclasses as dc
    hand = connected_hand
    joint = "thumb_cmc"
    motor = hand.config.joint_to_motor_map[joint]
    hand.config = dc.replace(hand.config, calibration_min_motor_span_rad={joint: 1.8})
    limits = {mid: [-5., -2.6] for mid in hand.config.motor_ids}
    hand.calibration = dc.replace(
        hand.calibration, motor_limits_dict=limits,
        joint_to_motor_ratios_dict={mid: .02 for mid in hand.config.motor_ids})
    assert hand.is_calibrated(verbose=False)
    limits[motor] = [-4.5421, -3.5067]
    hand.calibration = dc.replace(hand.calibration, motor_limits_dict=limits)
    assert not hand.is_calibrated(verbose=False)
    assert not hand.is_calibrated(verbose=True)
    assert "configured minimum 1.8000" in capsys.readouterr().out
