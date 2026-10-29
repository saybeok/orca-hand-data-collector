"""The arm path: wrist pose over the wire, IK, and the composed 20-DoF command.

The properties worth pinning are the ones that fail silently. A dropped proto
field looks like an arm that simply does not move. A wrong frame alignment looks
like an arm that moves, just sideways. Both are checked here against the real
model rather than a stub.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from orca_core import OrcaJointPositions

from orca_teleop.am200 import (
    Am200ActionComposer,
    ArmIk,
    ArmMappingConfig,
    CanonicalJointMap,
    RobotJointCommand,
)
from orca_teleop.ingress import hand_stream_pb2
from orca_teleop.ingress.server import HandLandmarks

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "robot_model" / "am_arm200_orca_right.xml"
HAND_CONFIG_PATH = ROOT.parent / "configs" / "orcahand-right-v1-feetech" / "config.yaml"
# The adaptive backend needs the URDF and the alias table to build its joint
# mapping, so the hook tests below cannot fall back to packaged defaults.
URDF_PATH = Path(
    os.environ.get(
        "ORCA_URDF_PATH",
        ROOT.parent / "orcahand_description" / "v1" / "models" / "urdf" / "orcahand_right.urdf",
    )
)
RETARGET_CONFIG_PATH = Path(
    os.environ.get(
        "ORCA_RETARGET_CONFIG", ROOT.parent / "configs" / "adaptive_analytical_orca_v1.yaml"
    )
)

ARM_DOF = 3
TOTAL_DOF = 20

# Borrow upstream's own idea of a pose the retargeter will accept, so this test
# follows along if that definition changes rather than silently going degenerate.
sys.path.insert(0, str(ROOT.parent / "orca_teleop" / "tests"))
from conftest import plausible_hand_keypoints  # noqa: E402


@pytest.fixture(scope="module")
def arm_ik() -> ArmIk:
    return ArmIk(MODEL_PATH)


@pytest.fixture
def composer() -> Am200ActionComposer:
    return Am200ActionComposer.from_model_path(MODEL_PATH)


@pytest.fixture(scope="module")
def hand_action() -> OrcaJointPositions:
    import yaml

    config = yaml.safe_load(HAND_CONFIG_PATH.read_text(encoding="utf-8"))
    positions = dict(config["neutral_position"])
    positions["wrist"] = 0.0
    return OrcaJointPositions(positions)


def wrist_frame(position) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = position
    return transform


def frame_at(position) -> HandLandmarks:
    return HandLandmarks(
        keypoints=np.zeros((21, 3), dtype=np.float32),
        handedness="right",
        timestamp_ns=0,
        wrist_transform=wrist_frame(position),
    )


# --- the wire ---------------------------------------------------------------


def test_wrist_pose_survives_the_proto():
    """The field the arm depends on; before this it was dropped at the boundary."""
    transform = np.arange(16, dtype=np.float64).reshape(4, 4)
    frame = hand_stream_pb2.HandFrame(
        keypoints=[0.0] * 63,
        handedness="right",
        timestamp_ns=7,
        wrist_transform=transform.ravel().tolist(),
    )
    decoded = hand_stream_pb2.HandFrame.FromString(frame.SerializeToString())
    np.testing.assert_allclose(
        np.array(decoded.wrist_transform).reshape(4, 4), transform, atol=1e-6
    )


def test_a_publisher_without_a_wrist_pose_yields_none():
    """Empty must mean 'absent', not 'identity' — identity would be a real pose."""
    frame = hand_stream_pb2.HandFrame(keypoints=[0.0] * 63, handedness="right")
    assert len(frame.wrist_transform) == 0


def test_the_ingress_server_decodes_the_wrist_pose():
    """Covers the receiving half: over a real gRPC hop, not just serialisation."""
    import queue
    import threading

    import grpc

    from orca_teleop.ingress import hand_stream_pb2_grpc
    from orca_teleop.ingress.server import IngressServer

    transform = wrist_frame([0.11, -0.22, 0.33])
    landmarks_q: queue.Queue = queue.Queue(maxsize=8)
    server = IngressServer(landmarks_q, threading.Event(), port=0)
    port = server.start()
    channel = grpc.insecure_channel(f"localhost:{port}")

    try:
        stub = hand_stream_pb2_grpc.HandStreamStub(channel)
        stub.StreamHandFrames(
            iter(
                [
                    hand_stream_pb2.HandFrame(
                        keypoints=np.zeros(63, dtype=np.float32).tolist(),
                        handedness="right",
                        timestamp_ns=1,
                        wrist_transform=transform.ravel().tolist(),
                    )
                ]
            )
        )
        received = landmarks_q.get(timeout=5.0)
    finally:
        channel.close()
        server.stop()

    assert received.wrist_transform is not None, "the wrist pose was dropped on receipt"
    np.testing.assert_allclose(received.wrist_transform, transform, atol=1e-6)


def test_a_malformed_wrist_pose_is_dropped_rather_than_reshaped():
    """A wrong length must not become a garbage pose that silently drives the arm."""
    import queue
    import threading

    import grpc

    from orca_teleop.ingress import hand_stream_pb2_grpc
    from orca_teleop.ingress.server import IngressServer

    landmarks_q: queue.Queue = queue.Queue(maxsize=8)
    server = IngressServer(landmarks_q, threading.Event(), port=0)
    port = server.start()
    channel = grpc.insecure_channel(f"localhost:{port}")

    try:
        stub = hand_stream_pb2_grpc.HandStreamStub(channel)
        stub.StreamHandFrames(
            iter(
                [
                    hand_stream_pb2.HandFrame(
                        keypoints=np.zeros(63, dtype=np.float32).tolist(),
                        handedness="right",
                        timestamp_ns=1,
                        wrist_transform=[1.0, 2.0, 3.0],
                    )
                ]
            )
        )
        received = landmarks_q.get(timeout=5.0)
    finally:
        channel.close()
        server.stop()

    assert received.wrist_transform is None


# --- the pipeline hook ------------------------------------------------------


def run_worker_once(composer) -> tuple[object, list]:
    """Drive `retargeter_worker` over a single frame and return what it emitted.

    Calls the worker directly on a pre-filled queue, so the assertions are about
    the hook's wiring rather than about gRPC or a running sim.
    """
    import queue
    import threading

    from orca_teleop.pipeline import _SHUTDOWN, TeleopQueues, retargeter_worker
    from orca_teleop.retargeting.adaptive_analytical import CALIBRATION_FRAMES

    seen: list = []

    def recording_composer(frame, action):
        seen.append((frame, action))
        return composer(frame, action)

    # The adaptive backend emits nothing until auto-scale calibration completes,
    # so the hook cannot fire on a single frame.
    n_frames = CALIBRATION_FRAMES + 2
    queues = TeleopQueues(
        landmarks_q=queue.Queue(maxsize=n_frames + 2),
        actions_q=queue.Queue(maxsize=n_frames + 2),
    )
    for index in range(n_frames):
        queues.landmarks_q.put(
            HandLandmarks(
                keypoints=plausible_hand_keypoints(),
                handedness="right",
                timestamp_ns=index + 1,
                wrist_transform=wrist_frame([0.0, 0.0, 0.1]),
            )
        )
    queues.landmarks_q.put(_SHUTDOWN)

    retargeter_worker(
        queues,
        threading.Event(),
        model_path=str(HAND_CONFIG_PATH),
        urdf_path=str(URDF_PATH),
        retargeter_config_path=str(RETARGET_CONFIG_PATH),
        landmark_source="webxr",
        action_composer=recording_composer,
    )
    return queues.actions_q.get_nowait(), seen


def test_the_retargeter_worker_emits_the_composed_command():
    """A frame in, a composed command out, with the hand action handed over intact."""
    emitted, seen = run_worker_once(
        lambda frame, action: RobotJointCommand.now(("arm.shoulder_pan",), np.array([0.5]))
    )

    assert seen, "the hook never fired"
    frame, hand_only = seen[0]
    assert frame.wrist_transform is not None, "the hook must see the wrist pose"
    assert isinstance(hand_only, OrcaJointPositions)
    assert isinstance(emitted, RobotJointCommand), "the composer's output must reach the sink"


def test_a_failing_composer_falls_back_to_the_hand():
    """A bug in the arm path must not take the hand down with it."""

    def broken(frame, action):
        raise RuntimeError("composer blew up")

    emitted, _ = run_worker_once(broken)
    assert isinstance(emitted, OrcaJointPositions)


# --- inverse kinematics -----------------------------------------------------


def test_ik_recovers_a_pose_it_generated(arm_ik):
    """Forward then inverse: the only self-consistency check that needs no truth."""
    truth = np.array([0.3, -0.4, 0.5])
    target = arm_ik.flange_position(truth)
    result = arm_ik.solve(target, seed=truth * 0.5, rate_limit=False)
    assert result.converged, f"residual {result.residual_m * 1e3:.2f} mm"
    np.testing.assert_allclose(result.achieved, target, atol=1e-3)


def test_ik_respects_the_joint_limits(arm_ik):
    lower, upper = arm_ik.limits
    result = arm_ik.solve(np.array([5.0, 5.0, 5.0]), seed=np.zeros(ARM_DOF), rate_limit=False)
    assert np.all(result.position >= lower - 1e-9)
    assert np.all(result.position <= upper + 1e-9)


def test_an_unreachable_target_reports_a_residual_instead_of_diverging(arm_ik):
    """Reach is ~0.49 m; 3 m away must degrade gracefully and say so."""
    result = arm_ik.solve(np.array([3.0, 0.0, 0.0]), seed=np.zeros(ARM_DOF), rate_limit=False)
    assert not result.converged
    assert result.residual_m > 1.0
    assert np.all(np.isfinite(result.position))


def test_the_rate_limit_bounds_a_single_step(arm_ik):
    """A teleport in the target must not become a teleport in the joints."""
    far = arm_ik.flange_position(np.array([1.5, 0.8, -0.9]))
    result = arm_ik.solve(far, seed=np.zeros(ARM_DOF), rate_limit=True)
    assert result.clamped_by_rate
    assert np.abs(result.position).max() <= arm_ik._max_joint_step_rad + 1e-9


# --- the composed command ---------------------------------------------------


def test_the_first_frame_anchors_rather_than_jumping(composer, hand_action):
    """Absolute Quest coordinates are meaningless, so frame one must not move the arm."""
    action = composer(frame_at([1.7, -0.3, 2.2]), hand_action)
    assert isinstance(action, RobotJointCommand)
    np.testing.assert_allclose(action.position[:ARM_DOF], np.zeros(ARM_DOF), atol=1e-6)


def test_the_arm_follows_the_operator_and_stops_when_they_do(composer, hand_action):
    origin = np.array([1.7, -0.3, 2.2])
    composer(frame_at(origin), hand_action)

    for _ in range(80):  # let the rate limiter converge
        moved = composer(frame_at(origin + np.array([0.0, 0.0, 0.20])), hand_action)

    assert np.abs(moved.position[:ARM_DOF]).max() > 0.05, "the arm did not follow"

    held = composer(frame_at(origin + np.array([0.0, 0.0, 0.20])), hand_action)
    np.testing.assert_allclose(held.position[:ARM_DOF], moved.position[:ARM_DOF], atol=1e-6)


def test_operator_up_moves_the_flange_up(composer, hand_action):
    """Fixes the frame alignment. Both frames are z-up, so this axis is unambiguous.

    Without it a transposed or mis-signed alignment still produces plausible
    motion, just in the wrong direction, which no other test here would catch.
    """
    origin = np.zeros(3)
    composer(frame_at(origin), hand_action)
    start = composer._ik.flange_position(composer.arm_position)

    for _ in range(120):
        composer(frame_at(origin + np.array([0.0, 0.0, 0.15])), hand_action)
    end = composer._ik.flange_position(composer.arm_position)

    rise = end[2] - start[2]
    assert rise > 0.03, f"operator moved up 150 mm, flange moved {rise * 1e3:.0f} mm in z"
    horizontal = np.linalg.norm(end[:2] - start[:2])
    assert horizontal < abs(rise), (
        f"a purely vertical wrist motion moved the flange {horizontal * 1e3:.0f} mm sideways "
        f"against {rise * 1e3:.0f} mm up, so the frame alignment is wrong"
    )


def test_scale_sets_how_far_the_flange_travels(hand_action):
    displacement = np.array([0.0, 0.0, 0.2])

    travel = {}
    for scale in (0.25, 0.75):
        composer = Am200ActionComposer.from_model_path(
            MODEL_PATH, config=ArmMappingConfig(scale=scale)
        )
        composer(frame_at(np.zeros(3)), hand_action)
        start = composer._ik.flange_position(composer.arm_position)
        for _ in range(200):
            composer(frame_at(displacement), hand_action)
        travel[scale] = np.linalg.norm(composer._ik.flange_position(composer.arm_position) - start)

    assert travel[0.75] > travel[0.25] * 2.0, (
        f"tripling the scale barely changed travel: {travel[0.25]:.3f} -> {travel[0.75]:.3f} m"
    )
    assert travel[0.25] == pytest.approx(0.25 * 0.2, abs=0.02)


def test_a_tracking_jump_re_anchors_instead_of_lunging(composer, hand_action):
    origin = np.zeros(3)
    composer(frame_at(origin), hand_action)
    for _ in range(60):
        composer(frame_at(origin + np.array([0.0, 0.0, 0.1])), hand_action)
    before = composer.arm_position.copy()

    # Hand tracking drops and re-acquires a metre away.
    after = composer(frame_at(origin + np.array([1.0, 0.0, 0.1])), hand_action)
    np.testing.assert_allclose(after.position[:ARM_DOF], before, atol=1e-6)


def test_a_frame_without_a_wrist_pose_passes_the_hand_through(composer, hand_action):
    """Keeps the webcam publisher working through the same composer."""
    frame = HandLandmarks(
        keypoints=np.zeros((21, 3), dtype=np.float32),
        handedness="right",
        timestamp_ns=0,
        wrist_transform=None,
    )
    assert composer(frame, hand_action) is hand_action


def test_the_hand_half_of_the_command_still_round_trips(composer, hand_action):
    """The arm must not disturb the physical/model conversion for the hand.

    The tolerance admits one known clamp: the v1 config's ``pinky_abd`` neutral
    sits 1.045e-4 deg outside the model's own limit for that joint, so clipping
    shaves exactly that much. It stays far below the scale a convention error
    would produce, which is whole degrees to tens of degrees.
    """
    action = composer(frame_at(np.zeros(3)), hand_action)
    assert action.position.shape == (TOTAL_DOF,)

    recovered = composer._map.hand_to_physical(action.position[ARM_DOF:])
    for joint_id, commanded in hand_action.as_dict().items():
        assert recovered[joint_id] == pytest.approx(commanded, abs=1e-3)


def test_the_command_is_in_the_canonical_order(composer, hand_action):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    expected = CanonicalJointMap.load(MODEL_PATH.parent / "joint_map.json", model)
    action = composer(frame_at(np.zeros(3)), hand_action)
    assert action.joint_ids == expected.roles


def test_hold_arm_composer_emits_full_command_with_arm_at_home(hand_action):
    import mujoco

    from orca_teleop.am200 import HoldArmActionComposer

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    joint_map = CanonicalJointMap.load(MODEL_PATH.parent / "joint_map.json", model)
    command = HoldArmActionComposer(joint_map)(frame_at([0.3, 0.1, 0.2]), hand_action)

    assert isinstance(command, RobotJointCommand)
    assert len(command.position) == TOTAL_DOF
    np.testing.assert_allclose(command.position[joint_map.arm_slice], np.zeros(ARM_DOF))
    np.testing.assert_allclose(
        command.position[joint_map.hand_slice],
        joint_map.clip(np.r_[np.zeros(ARM_DOF), joint_map.hand_from_physical(hand_action)])[
            joint_map.hand_slice
        ],
    )
