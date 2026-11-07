"""Single-slot mailboxes for live control, independent of recording queues."""
from dataclasses import replace
import queue

from orca_teleop.ingress.server import HandLandmarks


def coalesce_frames(previous, current):
    """Keep the newest pose without losing reset requests or clutch releases."""
    if isinstance(previous, HandLandmarks) and isinstance(current, HandLandmarks):
        return replace(
            current,
            retarget_recalibrate=(previous.retarget_recalibrate or current.retarget_recalibrate),
            clutch_released=(previous.clutch_released or current.clutch_released
                             or (previous.clutch_supported and not previous.clutch_engaged)),
        )
    return current


class LatestQueue(queue.Queue):
    """Nonblocking replacement; a queued shutdown cannot be overwritten.

    Only live poses/commands belong here. Dataset frames use their own lossless
    queue. Queue's mutex also protects replacement and unfinished-task counts.
    """

    def __init__(self, shutdown_sentinel):
        super().__init__(maxsize=1)
        self.shutdown_sentinel = shutdown_sentinel
        self.replaced_count = 0

    def put(self, item, block=True, timeout=None):
        with self.not_full:
            if self._qsize():
                previous = self._get()
                self.unfinished_tasks -= 1
                self.replaced_count += 1
                item = (previous if previous is self.shutdown_sentinel
                        else coalesce_frames(previous, item))
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()


def put_latest(q, item, shutdown_sentinel):
    """Also support callers supplying a standard bounded queue."""
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                previous = q.get_nowait()
            except queue.Empty:
                continue
            item = (previous if previous is shutdown_sentinel
                    else coalesce_frames(previous, item))
