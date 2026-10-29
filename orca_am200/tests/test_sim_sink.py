"""Contract tests for the 20-DoF simulation sink.

These drive the real MuJoCo model rather than a fake, because the properties
worth pinning are exactly the ones a fake would paper over: that the canonical
ordering reaches the intended actuator, that the hand converts between the
physical and model angle conventions, and that a hand-only command leaves the
arm where it was.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from orca_core import OrcaJointPositions

from orca_teleop.am200 import Am200SimSink, RobotJointCommand

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "robot_model" / "am_arm200_orca_right.xml"
HAND_CONFIG_PATH = ROOT.parent / "configs" / "orcahand-right-v1-feetech" / "config.yaml"

ARM_DOF = 3
TOTAL_DOF = 20
# Position servos under gravity do not land exactly on target; the hand joints
# settle within about 1.5 deg of command at the home pose.
SERVO_TOLERANCE_DEG = 3.0


@pytest.fixture(scope="module")
def sink():
    sink = Am200SimSink(
        model_path=MODEL_PATH,
        hand_config_path=HAND_CONFIG_PATH,
        render_mode="rgb_array",
    )
    sink.connect()
    yield sink
    sink.close()


def settle(sink: Am200SimSink, seconds: float = 1.5) -> np.ndarray:
    for _ in range(int(seconds / sink._dt)):
        sink._step()
    return sink.get_observation().joint_state


def test_joint_ids_are_the_canonical_twenty(sink):
    assert len(sink.joint_ids) == TOTAL_DOF
    assert sink.joint_ids[:ARM_DOF] == [
        "arm.shoulder_pan",
        "arm.shoulder_lift",
        "arm.elbow_flex",
    ]
    assert all(role.startswith("hand.") for role in sink.joint_ids[ARM_DOF:])


def test_observation_is_one_vector_plus_the_render(sink):
    obs = sink.get_observation()
    assert obs.joint_state.shape == (TOTAL_DOF,)
    assert set(obs.images) == {"frontal"}
    assert obs.images["frontal"].shape == (240, 320, 3)


def test_home_pose_reproduces_the_configured_neutral_hand(sink):
    """A physical-degree command must read back as the same physical degrees.

    This is the round trip that the `ref` sign controls: get it wrong and the
    thumb and abduction joints come back offset by up to 42 degrees, which is an
    order of magnitude beyond the servo tolerance below.
    """
    sink.go_home()
    state = settle(sink)

    measured = sink._map.hand_to_physical(state[ARM_DOF:])
    commanded = sink.home_position().as_dict()
    for joint_id, target in commanded.items():
        if joint_id == "wrist":  # covered by the droop test below
            continue
        assert measured[joint_id] == pytest.approx(target, abs=SERVO_TOLERANCE_DEG), (
            f"{joint_id} commanded {target} deg, settled at {measured[joint_id]:.1f} deg"
        )


def test_wrist_droop_is_the_upstream_gain_and_not_a_conversion_error(sink):
    """The wrist sags ~5 deg holding the hand, and it is a gain, not an offset.

    ``orca_sim``'s v1 hand gives the wrist actuator kp=2 N m/rad while the hand
    distal to that joint applies 0.18 N m of gravity torque, so the steady-state
    error is torque/kp by construction. Pinning it against that prediction keeps
    the test sensitive to a convention mistake, which would show up as a constant
    offset unrelated to load.
    """
    model = sink._env.model
    actuator = sink.joint_ids.index("hand.wrist")
    kp = float(model.actuator_gainprm[sink._map.actuator_index[actuator]][0])
    assert kp == pytest.approx(2.0), "upstream changed the wrist gain; re-derive the expectation"

    sink.go_home()
    state = settle(sink)
    droop_deg = abs(sink._map.hand_to_physical(state[ARM_DOF:])["wrist"])
    assert 3.0 < droop_deg < 8.0, (
        f"wrist settled {droop_deg:.1f} deg from target; expected the ~5 deg "
        "predicted by 0.18 N m / kp=2"
    )


def test_a_hand_only_command_leaves_the_arm_target_alone(sink):
    """The upstream retargeter emits hand joints only; the arm must not drift."""
    arm_target = np.array([0.4, -0.6, 0.9])
    sink.set_target(
        RobotJointCommand.now(
            tuple(sink.joint_ids),
            np.concatenate([arm_target, sink._target[ARM_DOF:]]),
        )
    )
    settle(sink)

    fist = OrcaJointPositions(
        {
            **sink.home_position().as_dict(),
            "index_mcp": 60.0,
            "middle_mcp": 60.0,
        }
    )
    sink.set_target(fist)
    assert sink._target[:ARM_DOF] == pytest.approx(arm_target, abs=1e-9)

    state = settle(sink)
    assert np.rad2deg(state[:ARM_DOF]) == pytest.approx(
        np.rad2deg(arm_target), abs=SERVO_TOLERANCE_DEG
    )
    measured = sink._map.hand_to_physical(state[ARM_DOF:])
    assert measured["index_mcp"] == pytest.approx(60.0, abs=SERVO_TOLERANCE_DEG)


def test_targets_are_clamped_to_the_joint_limits(sink):
    huge = np.full(TOTAL_DOF, 100.0)
    sink.set_target(RobotJointCommand.now(tuple(sink.joint_ids), huge))
    assert np.all(sink._target <= sink._map.upper + 1e-9)

    sink.set_target(RobotJointCommand.now(tuple(sink.joint_ids), -huge))
    assert np.all(sink._target >= sink._map.lower - 1e-9)


def test_optional_arm_envelope_matches_hardware_client(sink, monkeypatch):
    from orca_teleop.am200.arm_client import AmArmFeetech, default_arm_config_path

    motor_writes = []

    class Client:
        def write_positions_sync(self, ids, motor, **kwargs):
            motor_writes.append(motor.copy())

    # Config and command conversion only; no serial port is opened.
    arm = AmArmFeetech(default_arm_config_path(), max_step_rad=0.3)
    arm._client = Client()
    arm._motor_home = np.zeros(3)
    monkeypatch.setattr(sink, "_arm_max_step_rad", 0.3)
    for q in ([0.8, -0.7, 0.1], [-0.8, 0.6, -0.2]):
        arm.write_model_rad(np.array(q))
        target = sink._home_vector()
        hand = target[3:].copy()
        target[:3] = q
        sink.set_target(RobotJointCommand.now(tuple(sink.joint_ids), target))
        np.testing.assert_allclose(sink._target[:3], motor_writes[-1] / arm._signs, atol=1e-7)
        np.testing.assert_array_equal(sink._target[3:], hand)


def test_a_mismatched_command_is_rejected_rather_than_misapplied(sink):
    with pytest.raises(ValueError):
        sink.set_target(RobotJointCommand.now(("arm.shoulder_pan",), np.zeros(1)))

    with pytest.raises(TypeError):
        sink.set_target(np.zeros(TOTAL_DOF))


def test_canonical_order_drives_the_intended_actuator(sink):
    """One joint at a time, so a permutation bug cannot hide behind symmetry."""
    sink.go_home()
    baseline = settle(sink)

    index = sink.joint_ids.index("hand.index_pip")
    target = sink._target.copy()
    target[index] = sink._map.upper[index] - 0.05
    sink.set_target(RobotJointCommand.now(tuple(sink.joint_ids), target))
    moved = settle(sink)

    delta_deg = np.rad2deg(np.abs(moved - baseline))
    assert delta_deg[index] > 10.0, "the addressed joint barely moved"

    others = np.delete(delta_deg, index)
    other_names = [name for position, name in enumerate(sink.joint_ids) if position != index]
    worst = int(np.argmax(others))
    assert others.max() < 10.0, (
        f"commanding index_pip also moved {other_names[worst]} by {others[worst]:.1f} deg"
    )
