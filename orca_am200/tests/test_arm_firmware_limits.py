"""Firmware stops must constrain IK rather than leave a fictitious moving arm."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from orca_core import OrcaJointPositions

from orca_teleop.am200 import Am200ActionComposer, ArmMappingConfig
from orca_teleop.am200.arm_client import AmArmFeetech
from orca_teleop.hand_modes import FoldThreeFingersComposer
from orca_teleop.ingress.server import HandLandmarks
from orca_teleop.pipeline import _configure_arm_limits

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT/'orca_am200/robot_model/am_arm200_orca_right.xml'
HAND = ROOT/'configs/orcahand-right-v1-feetech/config.yaml'
# Read-only 2026-11-05 home capture, reproduced here without opening hardware.
RAW_LIMITS = np.array([[1335, 2693], [691, 2677], [1151, 3032]])
HOME = np.array([2108, 691, 2802])*(-2*np.pi/4096)
MOTOR_LIMITS = np.sort(RAW_LIMITS*(-2*np.pi/4096), axis=1)


def arm(signs):
    value = AmArmFeetech.__new__(AmArmFeetech)
    value._motor_home = HOME.copy()
    value._motor_limits = MOTOR_LIMITS.copy()
    value._signs = np.asarray(signs, dtype=float)
    value._max_step = None
    return value


@pytest.mark.parametrize('signs', [[1, 1, 1], [-1, 1, 1]])
def test_firmware_travel_is_converted_with_each_motor_polarity(signs):
    value = arm(signs)
    lower, upper = value.model_limits
    assert upper[1] == pytest.approx(0)
    assert np.rad2deg(lower[1]) == pytest.approx(-174.55078125)
    for n in range(11):
        q = lower+(upper-lower)*n/10
        motor = HOME+value._signs*q
        assert np.all(motor >= MOTOR_LIMITS[:, 0]-1e-8)
        assert np.all(motor <= MOTOR_LIMITS[:, 1]+1e-8)


def test_motor_writer_rejects_unsupported_goal_instead_of_silently_clipping_driver_raw():
    value = arm([1, 1, 1])
    writes = []
    value._client = SimpleNamespace(write_positions_sync=lambda *a, **kw: writes.append(a))
    value._motor_ids = [1, 2, 3]
    value._speed, value._max_torque = 100, 700
    with pytest.raises(ValueError, match='firmware'):
        value.write_model_rad(np.deg2rad([0, 45, 0]))
    assert not writes
    value.write_model_rad(np.deg2rad([0, -10, 5]))
    assert len(writes) == 1


@pytest.mark.parametrize('folded', [False, True])
def test_ik_never_accumulates_impossible_lift_and_can_move_forward_after_backwards(folded):
    c = Am200ActionComposer.from_model_path(MODEL, hand_config_path=HAND,
        config=ArmMappingConfig(scale=1, max_velocity_rad_s=np.deg2rad(30),
                                max_acceleration_rad_s2=np.deg2rad(90)))
    outer = FoldThreeFingersComposer(HAND, c) if folded else c
    lower, upper = arm([1, 1, 1]).model_limits
    _configure_arm_limits(SimpleNamespace(arm_joint_limits=(lower, upper)), outer)
    np.testing.assert_array_equal(c.joint_map.upper[:3], upper)
    hand = OrcaJointPositions(dict(yaml.safe_load(HAND.read_text())['neutral_position']))
    frames = []
    for n, forward in enumerate([0]+[-.02]*90+[.02]*120+[0]*60):
        wrist = np.eye(4)
        wrist[0, 3] = forward
        frame = HandLandmarks(np.zeros((21, 3)), 'right', 1_000_000_000+n*33_333_333,
                             wrist_transform=wrist)
        command = outer(frame, hand)
        frames.append(command.position[:3])
    positions = np.asarray(frames)
    assert np.max(positions[:, 1]) <= 1e-9
    assert np.min(positions[91:211, 1]) < np.deg2rad(-10)
    motors = HOME+positions
    assert np.all(motors >= MOTOR_LIMITS[:, 0]-1e-8)
    assert np.all(motors <= MOTOR_LIMITS[:, 1]+1e-8)
    assert np.max(abs(np.diff(positions, axis=0))) <= np.deg2rad(30)/30+1e-7


def test_failed_ik_bound_configuration_preserves_arm_hold_without_return_motion():
    closed = []
    sink = SimpleNamespace(arm_joint_limits=(np.ones(3), -np.ones(3)),
                           close=lambda return_home=True: closed.append(return_home))
    c = Am200ActionComposer.from_model_path(MODEL)
    with pytest.raises(ValueError):
        _configure_arm_limits(sink, c)
    assert closed == [False]
