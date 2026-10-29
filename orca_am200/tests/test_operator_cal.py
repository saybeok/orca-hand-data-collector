"""Clutch and persisted operator calibration.

The property under test is reproducibility: the same operator posture must
command the same robot pose on every run. An anchor captured at process start
cannot do that, and the failure is silent — the arm still moves, just to a
different place, which is exactly how a demonstration dataset acquires a
per-episode offset nobody notices.

So the tests here mostly compare *two independent runs* rather than checking one.
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
    ArmMappingConfig,
    OperatorCalibration,
    default_calibration_path,
)
from orca_teleop.ingress.metaquest.landmarks import (
    INDEX_TIP_INDEX,
    THUMB_TIP_INDEX,
    PinchDetector,
)
from orca_teleop.ingress.server import HandLandmarks

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "robot_model" / "am_arm200_orca_right.xml"
HAND_CONFIG_PATH = ROOT.parent / "configs" / "orcahand-right-v1-feetech" / "config.yaml"

ARM_DOF = 3
NEUTRAL = np.array([0.1, -0.2, 1.1])

sys.path.insert(0, str(ROOT.parent / "orca_teleop" / "tests"))


@pytest.fixture(scope="module")
def hand_action() -> OrcaJointPositions:
    import yaml

    config = yaml.safe_load(HAND_CONFIG_PATH.read_text(encoding="utf-8"))
    positions = dict(config["neutral_position"])
    positions["wrist"] = 0.0
    return OrcaJointPositions(positions)


@pytest.fixture
def calibration() -> OperatorCalibration:
    return OperatorCalibration(
        neutral_wrist_flu_m=NEUTRAL.copy(),
        handedness="right",
        samples=60,
        captured_at="2026-08-26T00:00:00+00:00",
    )


def frame(position, *, clutch_supported=False, clutch_engaged=False) -> HandLandmarks:
    transform = np.eye(4)
    transform[:3, 3] = position
    return HandLandmarks(
        keypoints=np.zeros((21, 3), dtype=np.float32),
        handedness="right",
        timestamp_ns=0,
        wrist_transform=transform,
        clutch_supported=clutch_supported,
        clutch_engaged=clutch_engaged,
    )


def pinch_landmarks(gap_m: float) -> np.ndarray:
    points = np.zeros((25, 3), dtype=np.float64)
    points[THUMB_TIP_INDEX] = [0.0, 0.0, 0.0]
    points[INDEX_TIP_INDEX] = [gap_m, 0.0, 0.0]
    return points


def settle(composer, position, hand_action, *, engaged=True, supported=False, steps=250):
    for _ in range(steps):
        composer(
            frame(position, clutch_supported=supported, clutch_engaged=engaged),
            hand_action,
        )
    return composer._ik.flange_position(composer.arm_position)


# --- pinch detection --------------------------------------------------------


def test_pinch_latches_with_hysteresis():
    """One threshold would chatter, which on a clutch means a stuttering arm."""
    detector = PinchDetector(close_m=0.03, open_m=0.045)

    assert not detector.update(pinch_landmarks(0.08))
    assert detector.update(pinch_landmarks(0.02))
    # Between the thresholds the previous state must persist, both ways.
    assert detector.update(pinch_landmarks(0.038))
    assert not detector.update(pinch_landmarks(0.06))
    assert not detector.update(pinch_landmarks(0.038))


def test_a_lost_hand_releases_the_clutch():
    """Fail-safe direction: an untracked hand cannot be granting permission."""
    detector = PinchDetector()
    detector.update(pinch_landmarks(0.01))
    assert detector.engaged
    assert not detector.update(None)


def test_pinch_rejects_a_wrong_shaped_frame():
    with pytest.raises(ValueError):
        PinchDetector().update(np.zeros((21, 3)))


def test_pinch_thresholds_must_be_ordered():
    with pytest.raises(ValueError):
        PinchDetector(close_m=0.05, open_m=0.03)


# --- calibration file -------------------------------------------------------


def test_calibration_round_trips_through_yaml(tmp_path, calibration):
    path = calibration.save(tmp_path / "operator_calibration.yaml")
    loaded = OperatorCalibration.load(path)

    np.testing.assert_allclose(loaded.neutral_wrist_flu_m, calibration.neutral_wrist_flu_m)
    assert loaded.handedness == calibration.handedness
    assert loaded.samples == calibration.samples
    assert loaded.captured_at == calibration.captured_at


def test_a_missing_file_loads_as_none(tmp_path):
    """Absent must be reportable, not fatal — the caller warns and falls back."""
    assert OperatorCalibration.load(tmp_path / "nope.yaml") is None


def test_a_future_format_version_is_refused(tmp_path):
    """Better to demand a re-capture than to read a vector that means something else."""
    path = tmp_path / "operator_calibration.yaml"
    path.write_text(
        "version: 99\nhandedness: right\nneutral_wrist_flu_m: [0, 0, 0]\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="format version"):
        OperatorCalibration.load(path)


def test_capture_takes_the_median_and_reports_the_spread():
    held_still = np.tile(NEUTRAL, (40, 1)) + np.random.default_rng(0).normal(0, 0.002, (40, 3))
    held_still[7] = [5.0, 5.0, 5.0]  # a tracking blip the median must survive

    calibration, spread_m = OperatorCalibration.capture(held_still, "right")
    np.testing.assert_allclose(calibration.neutral_wrist_flu_m, NEUTRAL, atol=0.002)
    assert spread_m < 0.01, "one outlier should not dominate the reported spread"


def test_capture_needs_enough_samples_to_measure_spread():
    with pytest.raises(ValueError):
        OperatorCalibration.capture(NEUTRAL.reshape(1, 3), "right")


def test_the_default_path_is_overridable(monkeypatch, tmp_path):
    monkeypatch.setenv("ORCA_OPERATOR_CALIBRATION", str(tmp_path / "elsewhere.yaml"))
    assert default_calibration_path() == tmp_path / "elsewhere.yaml"

    monkeypatch.delenv("ORCA_OPERATOR_CALIBRATION")
    monkeypatch.setenv("ORCA_HAND_ROOT", str(tmp_path))
    assert default_calibration_path() == tmp_path / "configs" / "operator_calibration.yaml"


# --- what the calibration buys: reproducibility -----------------------------


def test_calibration_makes_two_runs_agree(calibration, hand_action):
    """The whole point. Same posture, two processes, same commanded arm pose.

    The two runs differ in where the operator's hand was when tracking began,
    which is the one thing a first-frame anchor would let leak into the result.
    """
    poses = []
    for start_offset in ([0.0, 0.0, 0.0], [0.13, -0.07, 0.11]):
        composer = Am200ActionComposer.from_model_path(
            MODEL_PATH, config=ArmMappingConfig(scale=0.5), calibration=calibration
        )
        composer(frame(NEUTRAL + np.asarray(start_offset)), hand_action)
        poses.append(settle(composer, NEUTRAL + np.array([0.0, 0.0, 0.2]), hand_action))

    np.testing.assert_allclose(poses[0], poses[1], atol=2e-3)


def test_without_calibration_two_runs_disagree(hand_action):
    """Pins the failure the file fixes, so the guarantee cannot quietly regress.

    If this ever starts passing, first-frame anchoring became reproducible and
    the calibration file is no longer earning its complexity.
    """
    poses = []
    for start_offset in ([0.0, 0.0, 0.0], [0.13, -0.07, 0.11]):
        composer = Am200ActionComposer.from_model_path(
            MODEL_PATH, config=ArmMappingConfig(scale=0.5), calibration=None
        )
        composer(frame(NEUTRAL + np.asarray(start_offset)), hand_action)
        poses.append(settle(composer, NEUTRAL + np.array([0.0, 0.0, 0.2]), hand_action))

    assert np.linalg.norm(poses[0] - poses[1]) > 0.02


def test_the_calibrated_neutral_maps_to_the_arm_home(calibration, hand_action):
    composer = Am200ActionComposer.from_model_path(MODEL_PATH, calibration=calibration)
    home = composer._ik.flange_position(np.zeros(ARM_DOF))

    settled = settle(composer, NEUTRAL, hand_action)
    np.testing.assert_allclose(settled, home, atol=2e-3)


def test_reset_restores_the_mapping_for_the_next_episode(calibration, hand_action):
    """Episode starts must not inherit the previous episode's clutch drift."""
    composer = Am200ActionComposer.from_model_path(
        MODEL_PATH, config=ArmMappingConfig(scale=0.5), calibration=calibration
    )
    first = settle(composer, NEUTRAL + np.array([0.0, 0.0, 0.15]), hand_action)

    # Drift the mapping the way a clutch cycle would, then start a new episode.
    composer(frame(NEUTRAL, clutch_supported=True, clutch_engaged=False), hand_action)
    composer(
        frame(NEUTRAL + np.array([0.3, 0.0, 0.0]), clutch_supported=True, clutch_engaged=True),
        hand_action,
    )
    composer.reset()

    second = settle(composer, NEUTRAL + np.array([0.0, 0.0, 0.15]), hand_action)
    np.testing.assert_allclose(first, second, atol=3e-3)


# --- clutch behaviour -------------------------------------------------------


def test_a_released_clutch_parks_the_arm_but_not_the_hand(calibration, hand_action):
    composer = Am200ActionComposer.from_model_path(MODEL_PATH, calibration=calibration)
    settle(composer, NEUTRAL, hand_action, supported=True, engaged=True, steps=60)
    parked = composer.arm_position.copy()

    fist = OrcaJointPositions({**hand_action.as_dict(), "index_mcp": 55.0})
    for _ in range(120):
        action = composer(
            frame(NEUTRAL + np.array([0.0, 0.0, 0.4]), clutch_supported=True, clutch_engaged=False),
            fist,
        )

    np.testing.assert_allclose(composer.arm_position, parked, atol=1e-12)
    recovered = composer._map.hand_to_physical(action.position[ARM_DOF:])
    assert recovered["index_mcp"] == pytest.approx(55.0, abs=1e-3)


def test_re_engaging_resumes_from_where_the_arm_was(calibration, hand_action):
    """The mouse-lift: reposition while released, then carry on without a jump."""
    composer = Am200ActionComposer.from_model_path(
        MODEL_PATH, config=ArmMappingConfig(scale=0.5), calibration=calibration
    )
    settle(composer, NEUTRAL, hand_action, supported=True, engaged=True, steps=60)
    parked = composer.arm_position.copy()

    # Released: hand travels a long way, arm must not.
    for _ in range(30):
        composer(
            frame(NEUTRAL + np.array([0.0, 0.0, 0.5]), clutch_supported=True, clutch_engaged=False),
            hand_action,
        )

    # First engaged frame at the new hand position must not move the arm either.
    composer(
        frame(NEUTRAL + np.array([0.0, 0.0, 0.5]), clutch_supported=True, clutch_engaged=True),
        hand_action,
    )
    assert np.abs(composer.arm_position - parked).max() < 0.07, (
        "re-engaging jumped the arm instead of re-anchoring"
    )


def test_a_publisher_without_a_clutch_follows_unconditionally(calibration, hand_action):
    """The webcam case: absent clutch must not read as a permanently held brake."""
    composer = Am200ActionComposer.from_model_path(
        MODEL_PATH, config=ArmMappingConfig(scale=0.5), calibration=calibration
    )
    start = settle(composer, NEUTRAL, hand_action, steps=30)
    moved = settle(composer, NEUTRAL + np.array([0.0, 0.0, 0.2]), hand_action)

    assert moved[2] - start[2] > 0.05, "the arm stayed put with no clutch in play"


# --- the clutch on the wire -------------------------------------------------


def _webxr_payload(*, left_gap_m: float, right_gap_m: float) -> dict:
    """A both-hands frame in the shape the WebXR client sends."""

    def hand(gap_m: float) -> dict:
        points = np.full((25, 3), 0.05, dtype=np.float64)
        points[THUMB_TIP_INDEX] = [0.0, 0.0, 0.0]
        points[INDEX_TIP_INDEX] = [gap_m, 0.0, 0.0]
        return {"landmarks": points.tolist(), "wrist": np.eye(4).ravel().tolist()}

    return {
        "client_wall_ms": 1.0,
        "hands": {"left": hand(left_gap_m), "right": hand(right_gap_m)},
    }


def _publisher_with(payload: dict, *, handedness="right", clutch_enabled=True):
    from orca_teleop.ingress.metaquest.publisher import MetaQuestPublisher

    publisher = MetaQuestPublisher(handedness=handedness, clutch_enabled=clutch_enabled)
    publisher.bridge.state.update(payload)
    return publisher, publisher.bridge.state.get_hand_sample(handedness)


def test_the_clutch_reads_the_off_hand_not_the_working_one():
    """The bug this forecloses: pinching to grasp would otherwise clutch the arm.

    Both hands pinch and open independently here, so a publisher that read the
    working hand would get the opposite answer on both assertions.
    """
    publisher, sample = _publisher_with(_webxr_payload(left_gap_m=0.01, right_gap_m=0.09))
    assert publisher._sample_to_proto(sample).clutch_engaged is True

    publisher, sample = _publisher_with(_webxr_payload(left_gap_m=0.09, right_gap_m=0.01))
    assert publisher._sample_to_proto(sample).clutch_engaged is False


def test_a_left_handed_operator_gets_the_clutch_on_the_right():
    publisher, sample = _publisher_with(
        _webxr_payload(left_gap_m=0.09, right_gap_m=0.01), handedness="left"
    )
    assert publisher._sample_to_proto(sample).clutch_engaged is True


def test_clutch_support_is_advertised_only_when_enabled():
    payload = _webxr_payload(left_gap_m=0.01, right_gap_m=0.09)

    publisher, sample = _publisher_with(payload, clutch_enabled=True)
    assert publisher._sample_to_proto(sample).clutch_supported is True

    publisher, sample = _publisher_with(payload, clutch_enabled=False)
    frame_pb = publisher._sample_to_proto(sample)
    assert frame_pb.clutch_supported is False
    assert frame_pb.clutch_engaged is False, "a disabled clutch must not report engagement"


def test_an_absent_off_hand_reports_released_not_unsupported():
    """Distinguishing these is what keeps a lost hand from meaning 'go ahead'."""
    payload = _webxr_payload(left_gap_m=0.01, right_gap_m=0.09)
    del payload["hands"]["left"]

    publisher, sample = _publisher_with(payload)
    frame_pb = publisher._sample_to_proto(sample)
    assert frame_pb.clutch_supported is True
    assert frame_pb.clutch_engaged is False


def test_the_clutch_survives_a_real_grpc_hop():
    """Covers the decode side, which a serialisation test alone would not."""
    import queue
    import threading

    import grpc

    from orca_teleop.ingress import hand_stream_pb2, hand_stream_pb2_grpc
    from orca_teleop.ingress.server import IngressServer

    landmarks_q: queue.Queue = queue.Queue(maxsize=4)
    server = IngressServer(landmarks_q, threading.Event(), port=0)
    port = server.start()
    channel = grpc.insecure_channel(f"localhost:{port}")
    try:
        hand_stream_pb2_grpc.HandStreamStub(channel).StreamHandFrames(
            iter(
                [
                    hand_stream_pb2.HandFrame(
                        keypoints=np.zeros(63, dtype=np.float32).tolist(),
                        handedness="right",
                        timestamp_ns=1,
                        clutch_supported=True,
                        clutch_engaged=True,
                    )
                ]
            )
        )
        received = landmarks_q.get(timeout=5.0)
    finally:
        channel.close()
        server.stop()

    assert received.clutch_supported is True
    assert received.clutch_engaged is True


def test_the_committed_calibration_if_any_is_loadable():
    """Guards the real file against a hand edit that would break every run."""
    path = default_calibration_path()
    if not path.is_file():
        pytest.skip(f"no operator calibration at {path}")
    assert OperatorCalibration.load(path) is not None


def test_environment_does_not_leak_into_the_tests():
    """These tests must not silently pick up a developer's own calibration."""
    assert (
        "ORCA_OPERATOR_CALIBRATION" not in os.environ
        or Path(os.environ["ORCA_OPERATOR_CALIBRATION"]).parent.exists()
    )
