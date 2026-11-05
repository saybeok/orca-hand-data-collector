"""Serialize taught-pose returns with live IK composition and motor writes."""
from dataclasses import replace
import threading
import time

from orca_teleop.am200.canonical import RobotJointCommand


class ArmEpisodeReturn:
    def __init__(self, sink, composer, dispatch_enabled):
        self.sink = sink
        self.composer = composer
        self.dispatch_enabled = dispatch_enabled
        self._lock = threading.RLock()
        self._cutoff_ns = 0
        self._await_release = False

    def reset(self):
        with self._lock:
            self.composer.reset()

    def compose(self, frame, hand_action):
        with self._lock:
            if self._await_release:
                # The runner drains old commands and warms up with dispatch
                # disabled. Do not count those frames as a new release, and
                # never compare the headset's clock with the PC clock.
                if (self.dispatch_enabled.is_set()
                        and (not frame.clutch_supported or not frame.clutch_engaged)):
                    self._await_release = False
                frame = replace(frame, clutch_supported=True, clutch_engaged=False)
            if not self.dispatch_enabled.is_set():
                frame = replace(frame, clutch_supported=True, clutch_engaged=False)
            return self.composer(frame, hand_action)

    def dispatch(self, action):
        with self._lock:
            # Recheck inside the lock: a return may have disabled dispatch
            # after the consumer's outer event check.
            if not self.dispatch_enabled.is_set():
                return
            if isinstance(action, RobotJointCommand) and action.timestamp_ns < self._cutoff_ns:
                return
            self.sink.dispatch_action(action)

    def return_home(self):
        self.dispatch_enabled.clear()
        with self._lock:
            # Any already-running write finishes before the slow return starts.
            position = self.sink.return_arm_home()
            self.composer.reset_arm_position(position)
            self._cutoff_ns = time.time_ns()
            self._await_release = True
            return position
