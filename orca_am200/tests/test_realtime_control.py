"""A stalled finger optimizer must not queue wrist motion or lose clutch edges."""
from dataclasses import replace
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from orca_core import OrcaJointPositions

from orca_teleop.am200 import Am200ActionComposer, ArmMappingConfig
from orca_teleop.ingress.server import HandLandmarks
from orca_teleop.latest import LatestQueue
from orca_teleop.pipeline import TeleopQueues, _SHUTDOWN, retargeter_worker
from orca_teleop.recording import TeleopActionMirror, teleop_consumer_loop

ROOT = Path(__file__).resolve().parents[2]
HAND = ROOT/'configs/orcahand-right-v1-feetech/config.yaml'
MODEL = ROOT/'orca_am200/robot_model/am_arm200_orca_right.xml'


def frame(n, z=0, engaged=True, reset=False):
    wrist = np.eye(4)
    wrist[2, 3] = z
    return HandLandmarks(np.zeros((21, 3)), 'right', 1_000_000_000+n*33_333_333,
                         wrist_transform=wrist, clutch_supported=True,
                         clutch_engaged=engaged, retarget_recalibrate=reset,
                         received_monotonic_ns=time.monotonic_ns())


def hand_action():
    return OrcaJointPositions(dict(yaml.safe_load(HAND.read_text())['neutral_position']))


def test_latest_input_keeps_newest_pose_reset_and_brief_release():
    q = LatestQueue(_SHUTDOWN)
    q.put(frame(0, engaged=False, reset=True))
    for n in range(1, 100):
        q.put(frame(n, z=n*.001))
    newest = q.get_nowait()
    assert newest.timestamp_ns == frame(99).timestamp_ns
    assert newest.wrist_transform[2, 3] == pytest.approx(.099)
    assert newest.retarget_recalibrate and newest.clutch_released
    assert q.empty()
    q.task_done()
    q.join()  # Replaced values must not leave unfinished task counts behind.
    q.put(_SHUTDOWN)
    q.put(frame(100))
    assert q.get_nowait() is _SHUTDOWN


def test_short_release_between_pose_updates_reanchors_the_real_composer():
    c = Am200ActionComposer.from_model_path(MODEL, hand_config_path=HAND,
        config=ArmMappingConfig(max_velocity_rad_s=np.deg2rad(20),
                                max_acceleration_rad_s2=np.deg2rad(40)))
    hand = hand_action()
    c(frame(0), hand)
    c(frame(1, z=.08), hand)
    held = c.arm_position
    q = LatestQueue(_SHUTDOWN)
    q.put(frame(2, z=.2, engaged=False))
    q.put(frame(3, z=.25))
    np.testing.assert_allclose(c(q.get_nowait(), hand).position[:3], held, atol=1e-12)


@pytest.fixture
def running_controller(monkeypatch):
    from orca_teleop import pipeline
    action = hand_action()
    first = threading.Event()
    blocked = threading.Event()
    release = threading.Event()
    reset = threading.Event()
    calls = []

    def solve(pose):
        calls.append(pose.timestamp_ns)
        if len(calls) == 1:
            first.set()
        else:
            blocked.set()
            assert release.wait(3)
        return action

    monkeypatch.setattr(pipeline.Retargeter, 'from_paths', lambda *a, **kw:
        SimpleNamespace(retarget=solve, reset=reset.set))
    composer = Am200ActionComposer.from_model_path(MODEL, hand_config_path=HAND,
        config=ArmMappingConfig(max_velocity_rad_s=np.deg2rad(20),
                                max_acceleration_rad_s2=np.deg2rad(40)))
    queues = TeleopQueues(LatestQueue(_SHUTDOWN), LatestQueue(_SHUTDOWN))
    stop = threading.Event()
    thread = threading.Thread(target=retargeter_worker, args=(queues, stop),
                              kwargs={'action_composer': composer}, daemon=True)
    thread.start()
    try:
        queues.landmarks_q.put(frame(0))
        assert first.wait(2)
        # The first solve has returned, but allow its cache callback to run.
        command = None
        for n in range(1, 8):
            queues.landmarks_q.put(frame(n))
            try:
                command = queues.actions_q.get(timeout=.05)
                break
            except queue.Empty:
                continue
        assert command is not None
        assert blocked.wait(1)
        yield SimpleNamespace(queues=queues, stop=stop, composer=composer,
            release=release, calls=calls, reset=reset, n=n, first_command=command)
    finally:
        release.set()
        stop.set()
        queues.landmarks_q.put(_SHUTDOWN)
        thread.join(3)
        assert not thread.is_alive()


def test_arm_follows_wrist_while_finger_solver_is_blocked(running_controller):
    r = running_controller
    positions = []
    for n in range(r.n+1, r.n+6):
        r.queues.landmarks_q.put(frame(n, z=.08))
        command = r.queues.actions_q.get(timeout=.2)
        positions.append(command.position[:3])
    assert len(r.calls) == 2, 'Finger solve was meant to remain blocked'
    assert np.max(abs(positions[-1]-positions[0])) > 1e-5
    assert np.max(abs(np.diff(positions, axis=0))) <= np.deg2rad(20)/30+1e-7
    # A release is handled without waiting for the finger solve to finish.
    r.queues.landmarks_q.put(frame(r.n+6, z=.3, engaged=False))
    held = r.queues.actions_q.get(timeout=.2).position[:3]
    np.testing.assert_allclose(held, positions[-1], atol=1e-12)


def test_recalibration_cannot_reuse_an_inflight_old_hand_result(running_controller):
    r = running_controller
    r.queues.landmarks_q.put(frame(r.n+1, reset=True))
    with pytest.raises(queue.Empty):
        r.queues.actions_q.get(timeout=.04)
    r.release.set()
    assert r.reset.wait(1)
    # The reset frame emits no finger pose. Old in-flight results stay invalid.
    with pytest.raises(queue.Empty):
        r.queues.actions_q.get(timeout=.04)
    for n in range(r.n+2, r.n+8):
        r.queues.landmarks_q.put(frame(n))
        try:
            command = r.queues.actions_q.get(timeout=.05)
            break
        except queue.Empty:
            pass
    else:
        pytest.fail('No output after a fresh post-reset hand solve')
    assert command.position.shape == (20,)


def test_stale_finger_result_cannot_keep_the_arm_advancing(running_controller, monkeypatch):
    import orca_teleop.am200.realtime as realtime
    r = running_controller
    held = r.composer.arm_position
    monkeypatch.setattr(realtime, 'HAND_STALE_S', 0)
    r.queues.landmarks_q.put(frame(r.n+1, z=.3))
    with pytest.raises(queue.Empty):
        r.queues.actions_q.get(timeout=.04)
    np.testing.assert_array_equal(r.composer.arm_position, held)


def test_recording_uses_only_completed_successful_dispatches():
    q = LatestQueue(_SHUTDOWN)
    stop, enabled, entered, release = [threading.Event() for _ in range(4)]
    mirror = TeleopActionMirror()
    action = hand_action()

    def dispatch(command):
        entered.set()
        assert release.wait(2)
        return command is action

    worker = threading.Thread(target=teleop_consumer_loop, args=(q,), kwargs=dict(
        mirror=mirror, stop_event=stop, dispatch_action=dispatch,
        dispatch_enabled=enabled, shutdown_sentinel=_SHUTDOWN, heartbeat_interval=.01))
    worker.start()
    try:
        q.put(action)
        deadline = time.monotonic()+1
        while mirror.snapshot() is None and time.monotonic() < deadline:
            time.sleep(.001)
        assert mirror.snapshot() is action
        assert mirror.applied_snapshot() is None
        enabled.set()
        q.put(action)
        assert entered.wait(1)
        assert mirror.applied_snapshot() is None, 'A pending bus write is not applied'
        release.set()
        deadline = time.monotonic()+1
        while mirror.applied_snapshot() is None and time.monotonic() < deadline:
            time.sleep(.001)
        assert mirror.applied_snapshot() is action
        rejected = hand_action()
        q.put(rejected)
        deadline = time.monotonic()+1
        while mirror.snapshot() is not rejected and time.monotonic() < deadline:
            time.sleep(.001)
        assert mirror.applied_snapshot() is action
        mirror.reset()
        assert mirror.applied_snapshot() is None
    finally:
        release.set()
        stop.set()
        worker.join(2)


def test_independent_control_keeps_fixed_fingers_and_final_command_diagnostics(monkeypatch):
    from orca_teleop import pipeline
    from orca_teleop.hand_modes import FoldThreeFingersComposer
    c = Am200ActionComposer.from_model_path(MODEL, hand_config_path=HAND)
    mode = FoldThreeFingersComposer(HAND, c)
    raw = hand_action()
    raw = OrcaJointPositions({**raw.data, 'thumb_dip': 130})
    bound = []
    def diagnose(ik, final, stamp):
        assert ik['thumb_dip'] == 130
        assert final['thumb_dip'] == pytest.approx(112)
        for joint, value in mode.pose_deg.items():
            assert final[joint] == pytest.approx(value)
        return {'command_timestamp_ns': stamp, 'final_deg': final}
    monkeypatch.setattr(pipeline.Retargeter, 'from_paths', lambda *a, **kw:
        SimpleNamespace(retarget=lambda pose: raw, command_diagnostics=diagnose,
                        set_fixed_hand_joints=bound.append))
    queues = TeleopQueues(LatestQueue(_SHUTDOWN), LatestQueue(_SHUTDOWN))
    stop = threading.Event()
    worker = threading.Thread(target=retargeter_worker, args=(queues, stop), kwargs=dict(
        action_composer=mode, fixed_hand_pose_deg=mode.pose_deg,
        diagnostic_hand_map=c.joint_map), daemon=True)
    worker.start()
    try:
        for n in range(10):
            queues.landmarks_q.put(frame(n))
            try:
                command = queues.actions_q.get(timeout=.05)
                break
            except queue.Empty:
                continue
        else:
            pytest.fail('Independent control never became ready')
        assert bound == [mode.pose_deg]
        physical = c.joint_map.hand_to_physical(command.position[c.joint_map.hand_slice])
        trace = command.hand_diagnostics
        assert trace is not None
        assert trace['command_timestamp_ns'] == command.timestamp_ns
        assert trace['hand_source_timestamp_ns'] >= 1_000_000_000
        for joint, value in trace['final_deg'].items():
            assert physical[joint] == pytest.approx(value)
    finally:
        stop.set()
        worker.join(2)
