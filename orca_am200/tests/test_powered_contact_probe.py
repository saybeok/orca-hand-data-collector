"""Powered pinch diagnosis must isolate one pair and preserve calibration."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from orca_core import OrcaJointPositions
from test_pinch_retarget import AdaptiveAnalyticalRetargeter, HAND_CONFIG, ROOT, URDF

spec = importlib.util.spec_from_file_location('check_pinch_contact', ROOT/'scripts/check_pinch_contact.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.fixture
def pair(monkeypatch):
    ordering = OrcaJointPositions._default_joint_ids
    before = [HAND_CONFIG.read_bytes(), HAND_CONFIG.with_name('calibration.yaml').read_bytes()]
    hand = probe.make_probe(str(HAND_CONFIG), 'index')
    assert hand.config.motor_ids == [13, 14, 16, 15, 3, 2, 1]
    assert hand.config.joint_ids == ['thumb_mcp','thumb_abd','thumb_pip','thumb_dip','index_abd','index_mcp','index_pip']
    pose = dict(hand.config.neutral_position)
    state = SimpleNamespace(pose=pose.copy(), writes=[], enabled=[], fresh=True)
    monkeypatch.setattr(hand, '_motor_client', SimpleNamespace(last_read_ok=True))
    def read():
        hand.motor_client.last_read_ok = state.fresh
        return SimpleNamespace(data=state.pose.copy())
    monkeypatch.setattr(hand, 'get_joint_position', read)
    def write(values):
        assert set(values) == set(pose)
        state.writes.append(dict(values)); state.pose = dict(values)
    monkeypatch.setattr(hand, 'set_joint_positions', write)
    monkeypatch.setattr(hand, 'set_max_current', lambda value: None)
    monkeypatch.setattr(hand, 'enable_torque', lambda: state.enabled.append(True) or [])
    monkeypatch.setattr(probe.time, 'sleep', lambda seconds: None)
    try:
        yield hand, state
    finally:
        hand._motor_client = None
        OrcaJointPositions._default_joint_ids = ordering
        assert before == [HAND_CONFIG.read_bytes(), HAND_CONFIG.with_name('calibration.yaml').read_bytes()]


def test_hold_preloads_current_pose_and_jog_holds_all_parent_joints(pair):
    hand, state = pair
    command = probe.begin_hold(hand)
    assert state.writes == [command] and state.enabled == [True]
    result = probe.jog(hand, command, 'thumb_dip', 2)
    assert result['thumb_dip'] == command['thumb_dip']+2
    previous = command['thumb_dip']
    for row in state.writes[1:]:
        assert abs(row['thumb_dip']-previous) <= 5*.05+1e-8
        previous = row['thumb_dip']
        for j in command:
            if j != 'thumb_dip': assert row[j] == command[j]


def test_reference_motion_is_synchronized_bounded_and_never_moves_unselected_motors(pair):
    hand, state = pair
    command = probe.begin_hold(hand)
    goal = dict(command, thumb_dip=command['thumb_dip']+4, index_pip=command['index_pip']+8)
    result = probe.move_reference(hand, command, goal)
    assert result == goal
    rows = np.array([[s[j] for j in command] for s in state.writes])
    assert np.max(abs(np.diff(rows, axis=0))) <= .25+1e-8
    state.fresh = False
    count = len(state.writes)
    with pytest.raises(RuntimeError): probe.move_reference(hand, goal, command)
    assert len(state.writes) == count


@pytest.mark.parametrize('delta', [0, 6, float('nan'), float('inf')])
def test_invalid_delta_cannot_move(pair, delta):
    hand, state = pair
    with pytest.raises(ValueError): probe.jog(hand, state.pose, 'thumb_dip', delta)
    assert not state.writes and not state.enabled


def test_stale_or_out_of_rom_feedback_aborts_before_torque_and_motion(pair):
    hand, state = pair
    state.fresh = False
    with pytest.raises(RuntimeError): probe.begin_hold(hand)
    assert not state.writes and not state.enabled
    state.fresh = True; state.pose['thumb_dip'] = 150
    with pytest.raises(RuntimeError): probe.begin_hold(hand)
    assert not state.writes and not state.enabled


def test_one_encoder_count_at_endpoint_is_preserved_but_command_is_clamped(pair):
    hand, state = pair
    upper = hand.config.joint_roms_dict['thumb_abd'][1]
    tick = probe.encoder_tick_deg(hand, 'thumb_abd')
    state.pose['thumb_abd'] = upper + .9*tick
    assert probe.fresh_angles(hand)['thumb_abd'] > upper
    command = probe.begin_hold(hand)
    assert command['thumb_abd'] == upper
    state.pose['thumb_abd'] = upper + 2*tick
    with pytest.raises(RuntimeError): probe.fresh_angles(hand)


def test_model_inspection_explains_hull_points_without_claiming_pad_contact(pair, tmp_path):
    hand, state = pair
    r = AdaptiveAnalyticalRetargeter(hand_config_path=str(HAND_CONFIG), urdf_path=str(URDF),
        config_path=str(ROOT/'configs/adaptive_analytical_orca_v1_pinch_surface.yaml'))
    geometry = r._surface_contacts
    command = probe.begin_hold(hand)
    path = tmp_path/'powered.json'
    row = probe.save_snapshot(path, r, geometry, hand, command, 'index', True)
    assert row['operator_confirmed_pad_contact'] is True
    assert row['encoder_is_independent_joint_sensor'] is False
    info = row['encoder_model']
    assert info['desired_pad_contact_verified'] is False
    assert np.shape(info['nearest_distal_local_mm']) == (2, 3)
    d, _ = geometry.distances([0], with_jacobian=False)
    assert info['signed_hull_clearance_mm'] == pytest.approx(d[0]*1000)
    assert info['configured_closed_target_mm'] == pytest.approx(-7.1337506263)
    assert json.loads(path.read_text())['method'] == 'powered_contact_check'
    probe.save_snapshot(path, r, geometry, hand, command, 'index', False)
    assert len(json.loads(path.read_text())['samples']) == 2
    assert len(state.writes) == 1  # Saving and FK never move the motors.


def test_stale_dataset_cannot_be_overwritten(pair, tmp_path):
    hand, state = pair
    r = AdaptiveAnalyticalRetargeter(hand_config_path=str(HAND_CONFIG), urdf_path=str(URDF),
        config_path=str(ROOT/'configs/adaptive_analytical_orca_v1_pinch_calibrated.yaml'))
    path = tmp_path/'powered.json'; path.write_text('{"method":"passive"}')
    before = path.read_bytes()
    with pytest.raises(ValueError, match='새 --samples'):
        probe.save_snapshot(path, r, r._surface_contacts, hand, state.pose, 'index', True)
    assert path.read_bytes() == before
