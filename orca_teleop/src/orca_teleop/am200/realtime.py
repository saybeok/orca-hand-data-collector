"""Follow fresh wrist poses without waiting for nonlinear finger retargeting.

    ingress -> arm IK + latest solved fingers -> single motor writer
            -> latest hand frame -> finger optimizer -> finger result cache

The recording queue is separate. No camera frames or dataset rows are dropped.
Only the finger thread accesses the retargeter and its diagnostic FK state.
"""
from dataclasses import dataclass, replace
import logging
import queue
import threading
import time

from orca_teleop.am200.canonical import RobotJointCommand
from orca_teleop.latest import LatestQueue, put_latest
from orca_teleop.ingress.server import HandLandmarks

logger = logging.getLogger(__name__)
HAND_STALE_S = 0.5


@dataclass(frozen=True)
class _HandResult:
    action: object
    diagnostics: dict | None
    epoch: int
    source_monotonic_ns: int
    source_timestamp_ns: int


def independent_arm_worker(queues, stop_event, *, composer, hand_worker_kwargs):
    from orca_teleop.pipeline import TeleopQueues, _SHUTDOWN, retargeter_worker

    hand_queues = TeleopQueues(LatestQueue(_SHUTDOWN), LatestQueue(_SHUTDOWN))
    lock = threading.Lock()
    result = None
    epoch = 0

    def cache_hand(frame, action, diagnostics):
        nonlocal result
        with lock:
            # A solve started before a requested recalibration cannot make the
            # arm ready again. The new calibration must emit a fresh result.
            if frame.control_epoch == epoch:
                result = _HandResult(action, diagnostics, epoch,
                                     frame.received_monotonic_ns or time.monotonic_ns(),
                                     frame.timestamp_ns)

    def solve_fingers():
        try:
            retargeter_worker(
                hand_queues, stop_event, **hand_worker_kwargs,
                independent_arm=False, hand_result_callback=cache_hand,
            )
        finally:
            stop_event.set()

    worker = threading.Thread(target=solve_fingers, name='finger-retargeter', daemon=True)
    worker.start()
    consumed_diagnostics = None
    last_log = time.monotonic()
    ages, durations = [], []
    stale_logged = False
    logger.info('Independent arm control enabled: latest wrist input, separate finger optimizer')
    try:
        while not stop_event.is_set():
            try:
                frame = queues.landmarks_q.get(timeout=.05)
            except queue.Empty:
                continue
            if frame is _SHUTDOWN:
                break
            if not isinstance(frame, HandLandmarks):
                raise ValueError(f'Expected HandLandmarks, got {type(frame)}')
            if frame.retarget_recalibrate:
                with lock:
                    epoch += 1
                    result = None
            hand_queues.landmarks_q.put_nowait(replace(frame, control_epoch=epoch))
            with lock:
                hand = result
            if hand is None:
                continue
            now = time.monotonic_ns()
            hand_age_s = (now-hand.source_monotonic_ns)*1e-9
            if hand_age_s > HAND_STALE_S:
                if not stale_logged:
                    logger.warning('Finger tracking stale (%.0f ms); holding arm until a fresh solve',
                                   hand_age_s*1000)
                    stale_logged = True
                continue
            stale_logged = False
            start = time.perf_counter()
            command = composer(frame, hand.action)
            # Missing/invalid wrist poses cannot produce a full robot command.
            if not isinstance(command, RobotJointCommand):
                continue
            if hand.diagnostics is not None and hand is not consumed_diagnostics:
                # The trace concerns this cached hand pose, not the newer wrist
                # input. Consume it once, so the bus is not polled every tick.
                trace = {**hand.diagnostics,
                         'command_timestamp_ns': command.timestamp_ns,
                         'hand_source_timestamp_ns': hand.source_timestamp_ns,
                         'hand_input_age_ms': hand_age_s*1000}
                command = replace(command, hand_diagnostics=trace)
                consumed_diagnostics = hand
            put_latest(queues.actions_q, command, _SHUTDOWN)
            durations.append((time.perf_counter()-start)*1000)
            if frame.received_monotonic_ns:
                ages.append((time.monotonic_ns()-frame.received_monotonic_ns)*1e-6)
            if time.monotonic()-last_log >= 2:
                logger.info('Arm control | IK+compose %.2f ms | input-to-command %.2f ms | '
                            'hand age %.0f ms | replaced inputs=%d hand_inputs=%d',
                            sum(durations)/len(durations),
                            sum(ages)/len(ages) if ages else 0.0, hand_age_s*1000,
                            getattr(queues.landmarks_q, 'replaced_count', 0),
                            hand_queues.landmarks_q.replaced_count)
                ages.clear()
                durations.clear()
                last_log = time.monotonic()
    finally:
        stop_event.set()
        hand_queues.landmarks_q.put_nowait(_SHUTDOWN)
        worker.join(timeout=2)
        put_latest(queues.actions_q, _SHUTDOWN, _SHUTDOWN)
