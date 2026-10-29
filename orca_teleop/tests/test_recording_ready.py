"""Tests for recording readiness gating."""

from __future__ import annotations

import queue
import threading
import time

import numpy as np
from orca_core import OrcaJointPositions

from orca_teleop.recording import (
    TeleopActionMirror,
    drain_actions_queue,
    poll_latest_action,
    teleop_consumer_loop,
    wait_for_recording_ready,
    wait_for_teleop_mirror_ready,
)

_SHUTDOWN = object()
_JOINT_IDS = [f"j{i}" for i in range(3)]


def _action(values: list[float] | None = None) -> OrcaJointPositions:
    vals = values if values is not None else [0.0, 0.0, 0.0]
    return OrcaJointPositions.from_dict(dict(zip(_JOINT_IDS, vals, strict=True)))


def test_wait_for_recording_ready_requires_action_and_obs_streaks():
    actions_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    obs_calls = {"n": 0}
    dispatched: list[OrcaJointPositions] = []

    def get_observation():
        obs_calls["n"] += 1
        return {"ok": True}

    for i in range(5):
        actions_q.put(_action([float(i), 0.0, 0.0]))

    ready = wait_for_recording_ready(
        get_observation=get_observation,
        actions_q=actions_q,
        stop_event=stop_event,
        dispatch_action=dispatched.append,
        shutdown_sentinel=_SHUTDOWN,
        min_action_streak=5,
        min_obs_streak=5,
        status_interval_s=100.0,
    )

    assert ready is True
    assert obs_calls["n"] == 5
    assert len(dispatched) == 5


def test_wait_for_recording_ready_resets_when_observation_fails():
    actions_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    obs_calls = {"n": 0}

    def get_observation():
        obs_calls["n"] += 1
        if obs_calls["n"] < 4:
            raise RuntimeError("camera dead")
        return {"ok": True}

    # 3 failing probes, then 5 healthy ones — must not declare ready early.
    for i in range(8):
        actions_q.put(_action([float(i), 0.0, 0.0]))

    ready = wait_for_recording_ready(
        get_observation=get_observation,
        actions_q=actions_q,
        stop_event=stop_event,
        dispatch_action=None,
        shutdown_sentinel=_SHUTDOWN,
        min_action_streak=5,
        min_obs_streak=5,
        status_interval_s=100.0,
    )

    assert ready is True
    assert obs_calls["n"] == 8


def test_wait_for_recording_ready_returns_false_on_stop():
    actions_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    def stopper():
        time.sleep(0.05)
        stop_event.set()

    threading.Thread(target=stopper, daemon=True).start()
    ready = wait_for_recording_ready(
        get_observation=lambda: None,
        actions_q=actions_q,
        stop_event=stop_event,
        dispatch_action=None,
        shutdown_sentinel=_SHUTDOWN,
        heartbeat_interval=0.01,
        min_action_streak=50,
        min_obs_streak=50,
        status_interval_s=100.0,
    )
    assert ready is False


def test_wait_for_recording_ready_returns_false_on_shutdown_sentinel():
    actions_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    actions_q.put(_SHUTDOWN)

    ready = wait_for_recording_ready(
        get_observation=lambda: None,
        actions_q=actions_q,
        stop_event=stop_event,
        dispatch_action=None,
        shutdown_sentinel=_SHUTDOWN,
        min_action_streak=5,
        min_obs_streak=5,
        status_interval_s=100.0,
    )
    assert ready is False
    assert stop_event.is_set()


def test_wait_for_recording_ready_treats_dispatch_failure_as_not_ready():
    actions_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    state = {"n": 0}

    def dispatch(action: OrcaJointPositions) -> None:
        state["n"] += 1
        if state["n"] < 3:
            raise RuntimeError("servo offline")

    for i in range(7):
        actions_q.put(_action([float(i), 0.0, 0.0]))

    ready = wait_for_recording_ready(
        get_observation=lambda: np.zeros(3),
        actions_q=actions_q,
        stop_event=stop_event,
        dispatch_action=dispatch,
        shutdown_sentinel=_SHUTDOWN,
        min_action_streak=5,
        min_obs_streak=5,
        status_interval_s=100.0,
    )
    assert ready is True
    assert state["n"] == 7


def test_drain_actions_queue_discards_stale_commands():
    actions_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    for i in range(4):
        actions_q.put(_action([float(i), 0.0, 0.0]))

    drained = drain_actions_queue(
        actions_q,
        stop_event=stop_event,
        shutdown_sentinel=_SHUTDOWN,
    )

    assert drained == 4
    assert actions_q.empty()


def test_poll_latest_action_returns_newest_command():
    actions_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    actions_q.put(_action([1.0, 0.0, 0.0]))
    actions_q.put(_action([2.0, 0.0, 0.0]))

    latest = poll_latest_action(
        actions_q,
        last_action=None,
        stop_event=stop_event,
        shutdown_sentinel=_SHUTDOWN,
    )

    assert latest is not None
    assert latest.as_array(_JOINT_IDS)[0] == 2.0
    assert actions_q.empty()


def test_teleop_consumer_loop_mirrors_and_respects_dispatch_gate():
    actions_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    dispatch_enabled = threading.Event()
    mirror = TeleopActionMirror()
    dispatched: list[OrcaJointPositions] = []

    thread = threading.Thread(
        target=teleop_consumer_loop,
        args=(actions_q,),
        kwargs={
            "mirror": mirror,
            "stop_event": stop_event,
            "shutdown_sentinel": _SHUTDOWN,
            "dispatch_action": dispatched.append,
            "dispatch_enabled": dispatch_enabled,
            "heartbeat_interval": 0.01,
        },
        daemon=True,
    )
    thread.start()

    actions_q.put(_action([1.0, 0.0, 0.0]))
    time.sleep(0.05)
    assert mirror.snapshot() is not None
    assert dispatched == []

    dispatch_enabled.set()
    actions_q.put(_action([2.0, 0.0, 0.0]))
    time.sleep(0.05)
    assert len(dispatched) == 1
    assert dispatched[0].as_array(_JOINT_IDS)[0] == 2.0

    stop_event.set()
    thread.join(timeout=1.0)


def test_wait_for_teleop_mirror_ready_waits_for_mirror_updates():
    stop_event = threading.Event()
    mirror = TeleopActionMirror()

    def feed_mirror():
        for i in range(5):
            time.sleep(0.02)
            mirror.update(_action([float(i), 0.0, 0.0]))

    threading.Thread(target=feed_mirror, daemon=True).start()
    ready = wait_for_teleop_mirror_ready(
        get_observation=lambda: {"ok": True},
        mirror=mirror,
        stop_event=stop_event,
        heartbeat_interval=0.01,
        min_action_streak=5,
        min_obs_streak=5,
        status_interval_s=100.0,
    )
    assert ready is True


def test_action_to_feature_accepts_robot_joint_command():
    from orca_teleop.am200 import RobotJointCommand
    from orca_teleop.recording import action_to_feature

    roles = ("arm.shoulder_pan", "arm.shoulder_lift", "arm.elbow_flex")
    cmd = RobotJointCommand.now(roles, np.array([0.1, -0.2, 0.3]))
    out = action_to_feature(cmd, list(roles))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [0.1, -0.2, 0.3])


def test_mirror_readiness_does_not_accept_stopped_stream_after_slow_observation(monkeypatch):
    import orca_teleop.recording as recording

    clock = [0.0]
    stop = threading.Event()
    mirror = TeleopActionMirror()
    emitted = [False]

    def sleep(seconds):
        clock[0] += seconds
        if not emitted[0]:
            for _ in range(5):
                mirror.update(_action())
            emitted[0] = True
        elif clock[0] > 3:
            stop.set()

    def observation():
        clock[0] += 1.1  # Stream stops while the sensor call is blocked.

    monkeypatch.setattr(recording.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(recording.time, 'sleep', sleep)
    assert not wait_for_teleop_mirror_ready(
        get_observation=observation, mirror=mirror, stop_event=stop,
        min_action_streak=5, min_obs_streak=1,
    )
