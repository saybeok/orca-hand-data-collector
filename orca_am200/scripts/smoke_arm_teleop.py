"""End-to-end check that a wrist pose on the wire moves the arm in the sim.

The unit tests cover each stage in isolation. This one walks a synthetic
operator through the whole path — gRPC publisher, ingress decode, retargeter,
composer, sim sink — and reports how far the flange actually travelled, which
is the thing a stage-by-stage test cannot confirm.

Kept as a script rather than a test because it stands up a real server and a
real MuJoCo model, and its value is the number it prints.

    python orca_am200/scripts/smoke_arm_teleop.py
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

import grpc
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "orca_teleop" / "tests"))

from conftest import plausible_hand_keypoints  # noqa: E402

from orca_teleop.am200 import (  # noqa: E402
    Am200ActionComposer,
    Am200SimSink,
    ArmMappingConfig,
    OperatorCalibration,
    RobotJointCommand,
)
from orca_teleop.ingress import hand_stream_pb2, hand_stream_pb2_grpc  # noqa: E402
from orca_teleop.ingress.server import IngressServer  # noqa: E402
from orca_teleop.pipeline import _SHUTDOWN, TeleopQueues, retargeter_worker  # noqa: E402
from orca_teleop.retargeting.adaptive_analytical import CALIBRATION_FRAMES  # noqa: E402

MODEL_PATH = ROOT / "robot_model" / "am_arm200_orca_right.xml"
HAND_CONFIG = ROOT.parent / "configs" / "orcahand-right-v1-feetech" / "config.yaml"
URDF_PATH = ROOT.parent / "orcahand_description" / "v1" / "models" / "urdf" / "orcahand_right.urdf"
RETARGET_CONFIG = ROOT.parent / "configs" / "adaptive_analytical_orca_v1.yaml"

LIFT_M = 0.25  # how far the synthetic operator raises their wrist
# The synthetic operator's neutral posture, which the calibration below declares.
# Being an explicit calibration rather than a first-frame anchor, the expected
# flange rise is exactly scale * LIFT_M and does not depend on when tracking
# started or how long the retargeter's own calibration took.
NEUTRAL_WRIST = np.zeros(3)
# Held still first anyway, because the retargeter emits nothing until its
# auto-scale calibration completes. Real operators do the same thing.
STILL_FRAMES = CALIBRATION_FRAMES + 20
RAMP_FRAMES = 150
SETTLE_FRAMES = 250
N_FRAMES = STILL_FRAMES + RAMP_FRAMES + SETTLE_FRAMES


def wrist_transform(height: float) -> list[float]:
    transform = np.eye(4)
    transform[:3, 3] = [0.0, 0.0, height]
    return transform.ravel().tolist()


def stream_frames(port: int) -> None:
    """A synthetic operator raising their hand, published over real gRPC."""
    channel = grpc.insecure_channel(f"localhost:{port}")
    stub = hand_stream_pb2_grpc.HandStreamStub(channel)
    keypoints = plausible_hand_keypoints().ravel().tolist()

    def frames():
        for index in range(N_FRAMES):
            # Still, then ramp, then hold: the last phase lets the rate limiter
            # and the servos settle before anything is measured.
            ramped = (index - STILL_FRAMES) / RAMP_FRAMES
            height = NEUTRAL_WRIST[2] + LIFT_M * float(np.clip(ramped, 0.0, 1.0))
            yield hand_stream_pb2.HandFrame(
                keypoints=keypoints,
                handedness="right",
                timestamp_ns=time.time_ns(),
                wrist_transform=wrist_transform(height),
            )
            time.sleep(0.002)

    try:
        stub.StreamHandFrames(frames())
    finally:
        channel.close()


def main() -> int:
    sink = Am200SimSink(
        model_path=MODEL_PATH,
        hand_config_path=HAND_CONFIG,
        render_mode="rgb_array",
    )
    sink.connect()

    composer = Am200ActionComposer.from_model_path(
        MODEL_PATH,
        config=ArmMappingConfig(scale=0.5),
        calibration=OperatorCalibration(
            neutral_wrist_flu_m=NEUTRAL_WRIST.copy(),
            handedness="right",
            samples=0,
            captured_at="synthetic",
        ),
    )
    start_flange = composer._ik.flange_position(np.zeros(3))

    queues = TeleopQueues(
        landmarks_q=queue.Queue(maxsize=8),
        actions_q=queue.Queue(maxsize=8),
    )
    stop_event = threading.Event()
    server = IngressServer(queues.landmarks_q, stop_event, port=0)
    port = server.start()

    worker = threading.Thread(
        target=retargeter_worker,
        kwargs=dict(
            queues=queues,
            stop_event=stop_event,
            model_path=str(HAND_CONFIG),
            urdf_path=str(URDF_PATH),
            retargeter_config_path=str(RETARGET_CONFIG),
            landmark_source="webxr",
            action_composer=composer,
        ),
        name="retargeter",
        daemon=True,
    )
    worker.start()

    publisher = threading.Thread(target=stream_frames, args=(port,), daemon=True)
    publisher.start()

    commands = 0
    deadline = time.monotonic() + 60.0
    try:
        while time.monotonic() < deadline:
            try:
                action = queues.actions_q.get(timeout=0.5)
            except queue.Empty:
                if not publisher.is_alive():
                    break
                continue
            if action is _SHUTDOWN:
                break
            if isinstance(action, RobotJointCommand):
                commands += 1
            sink.dispatch_action(action)
    finally:
        stop_event.set()
        server.stop()
        publisher.join(timeout=5.0)
        worker.join(timeout=5.0)

    measured_q = np.asarray(sink.get_observation().joint_state[:3], dtype=np.float64)
    commanded_q = composer.arm_position
    sink.close()

    # Separating these two matters: a shortfall in the commanded pose means the
    # mapping is wrong, whereas a shortfall only in the measured one is the
    # position servo sagging under the hand's weight.
    commanded_flange = composer._ik.flange_position(commanded_q)
    measured_flange = composer._ik.flange_position(measured_q)
    expected = LIFT_M * 0.5

    print(f"20-DoF commands consumed : {commands}")
    print(f"operator raised wrist    : {LIFT_M * 100:.1f} cm")
    print(f"expected flange rise     : {expected * 100:.1f} cm  (scale 0.5)")
    print()
    lag_deg = np.round(np.rad2deg(measured_q - commanded_q), 1).tolist()
    print(f"arm commanded (deg)      : {np.round(np.rad2deg(commanded_q), 1).tolist()}")
    print(f"arm measured  (deg)      : {np.round(np.rad2deg(measured_q), 1).tolist()}")
    print(f"servo lag     (deg)      : {lag_deg}")
    print()
    print(f"flange commanded rise    : {(commanded_flange - start_flange)[2] * 100:.1f} cm")
    print(f"flange measured  rise    : {(measured_flange - start_flange)[2] * 100:.1f} cm")
    print(
        f"sideways drift           : "
        f"{np.linalg.norm((measured_flange - start_flange)[:2]) * 100:.1f} cm"
    )

    commanded_rise = (commanded_flange - start_flange)[2]
    ok = (
        commands > 0
        and abs(commanded_rise - expected) < 0.005
        and np.linalg.norm((measured_flange - start_flange)[:2]) < 0.02
    )
    print("\nRESULT:", "arm tracked the operator" if ok else "ARM DID NOT TRACK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
