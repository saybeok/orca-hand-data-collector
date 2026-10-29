"""Arm command continuity under jitter, frame-rate changes and tracking loss."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from orca_core import OrcaJointPositions

from orca_teleop.am200 import Am200ActionComposer, ArmMappingConfig
from orca_teleop.ingress.server import HandLandmarks

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "orca_am200/robot_model/am_arm200_orca_right.xml"


def composer(limited=True):
    config = ArmMappingConfig(
        max_velocity_rad_s=np.deg2rad(120) if limited else None,
        max_acceleration_rad_s2=np.deg2rad(720) if limited else None,
    )
    return Am200ActionComposer.from_model_path(MODEL, config=config)


def frame(position, timestamp):
    wrist = np.eye(4)
    wrist[:3, 3] = position
    return HandLandmarks(np.zeros((21, 3)), "right", int(timestamp), wrist_transform=wrist)


def action(c):
    return OrcaJointPositions({j: 0.0 for j in c._map.hand_joint_ids})


@pytest.mark.parametrize("hz", [15, 30, 60])
@pytest.mark.parametrize("speed,accel", [(120, 720), (15, 45)])
def test_joint_commands_are_bounded_through_target_steps_and_reversals(hz, speed, accel):
    c = Am200ActionComposer.from_model_path(MODEL, config=ArmMappingConfig(
        max_velocity_rad_s=np.deg2rad(speed),
        max_acceleration_rad_s2=np.deg2rad(accel),
    ))
    hand = action(c)
    positions = [np.zeros(3)] + [np.array([0.0, 0.0, 0.15])] * hz
    positions += [np.array([0.0, 0.0, -0.05])] * hz
    commands, velocities = [], []
    for i, wrist in enumerate(positions):
        commands.append(c(frame(wrist, 1_000_000_000 + round(i * 1e9 / hz)), hand).position[:3])
        velocities.append(c._arm_velocity.copy())
    dq = np.diff(commands, axis=0)
    assert np.max(np.abs(dq)) <= np.deg2rad(speed) / hz + 1e-7
    assert np.max(np.abs(np.diff(velocities, axis=0))) <= np.deg2rad(accel) / hz + 1e-7
    assert np.max(np.abs(np.diff(dq, axis=0))) <= np.deg2rad(accel) / hz**2 + 1e-7
    assert np.max(np.abs(commands)) > 0.05, "a frozen arm trivially satisfies bounds"


def test_a_stationary_noisy_wrist_produces_less_command_jitter():
    output = []
    for limited in (False, True):
        c = composer(limited)
        hand = action(c)
        values = []
        for i in range(250):
            z = 0.0 if i == 0 else 0.10 + (0.003 * (-1)**i if i >= 150 else 0)
            values.append(c(frame([0, 0, z], 1_000_000_000 + i * 33_333_333), hand).position[:3])
        output.append(np.sqrt(np.mean(np.diff(values[150:], axis=0)**2)))
    assert output[0] > 1e-5
    assert output[1] < 0.4 * output[0]


def test_duplicate_frame_and_tracking_gap_do_not_move_the_arm():
    c = composer()
    hand = action(c)
    c(frame([0, 0, 0], 1_000_000_000), hand)
    c(frame([0, 0, 0.1], 1_033_333_333), hand)
    held = c.arm_position
    for timestamp in (1_033_333_333, 1_020_000_000, 2_000_000_000):
        command = c(frame([0.1, 0, 0.1], timestamp), hand)
        np.testing.assert_allclose(command.position[:3], held, atol=1e-12)
    # Reacquired wrist now anchors to the held pose; no delayed catch-up.
    command = c(frame([0.1, 0, 0.1], 2_033_333_333), hand)
    np.testing.assert_allclose(command.position[:3], held, atol=1e-10)


def test_clutch_release_holds_and_reengagement_reanchors():
    c = composer()
    hand = action(c)
    c(frame([0, 0, 0], 1_000_000_000), hand)
    c(frame([0, 0, 0.1], 1_033_333_333), hand)
    held = c.arm_position
    released = replace(frame([0.2, 0, 0.1], 1_066_666_666), clutch_supported=True)
    np.testing.assert_allclose(c(released, hand).position[:3], held)
    engaged = replace(released, timestamp_ns=1_100_000_000, clutch_engaged=True)
    np.testing.assert_allclose(c(engaged, hand).position[:3], held, atol=1e-10)


def test_reference_converges_to_a_fixed_joint_target_without_drift():
    c = composer()
    target = np.array([0.2, -0.3, 0.4])
    for _ in range(150):
        c._advance_arm(target, 1 / 30)
    np.testing.assert_allclose(c.arm_position, target, atol=1e-6)


@pytest.mark.parametrize("speed,accel", [(0, 1), (1, -1), (np.nan, 1), (1, None)])
def test_invalid_motion_settings_are_rejected(speed, accel):
    with pytest.raises(ValueError):
        ArmMappingConfig(max_velocity_rad_s=speed, max_acceleration_rad_s2=accel)


def test_webxr_left_hand_open_releases_arm_after_wire_decode():
    import queue
    import threading
    from orca_teleop.ingress.metaquest.publisher import MetaQuestPublisher
    from orca_teleop.ingress.metaquest.bridge import WebXRHandSample
    from orca_teleop.ingress.hand_stream_pb2 import HandFrame
    from orca_teleop.ingress.server import _HandStreamServicer

    publisher = MetaQuestPublisher(clutch_enabled=True)
    c = Am200ActionComposer.from_model_path(MODEL, config=ArmMappingConfig(scale=0.35))
    hand = action(c)
    received = queue.Queue()
    service = _HandStreamServicer(received, threading.Event())

    class Context:
        def peer(self):
            return "test-no-socket"

    def send(z, left_gap, n):
        hands = {}
        if left_gap is not None:
            landmarks = np.zeros((25, 3))
            landmarks[1:, 2] = 0.13
            landmarks[9, 0] = left_gap  # WebXR index tip; thumb tip is 4.
            hands["left"] = {"landmarks": landmarks.tolist()}
        publisher._bridge.state.update({"hands": hands})
        pose = np.eye(4)
        pose[2, 3] = z
        proto = publisher._sample_to_proto(WebXRHandSample(n, 1_000_000_000 + n * 33_333_333,
                                                          np.zeros((25, 3)), pose))
        decoded = HandFrame.FromString(proto.SerializeToString())
        service.StreamHandFrames(iter([decoded]), Context())
        return c(received.get_nowait(), hand)

    send(0, 0.01, 0)
    moved = send(0.1, 0.01, 1).position[:3].copy()
    assert np.linalg.norm(moved) > 0.01
    for n in range(2, 8):
        np.testing.assert_array_equal(send(0.1 + n * 0.02, 0.10, n).position[:3], moved)
    assert not c._engaged
    np.testing.assert_array_equal(send(0.3, None, 8).position[:3], moved)
    # Re-pinching anchors at the held robot target rather than catching up.
    np.testing.assert_allclose(send(0.3, 0.01, 9).position[:3], moved, atol=1e-10)
