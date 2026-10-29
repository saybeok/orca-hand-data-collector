"""Capture the operator's neutral wrist position, once, to a file.

The counterpart to `orca_core`'s `hand.sh calibrate`, for the human rather than
the hand. It records where your wrist sits when you are in your neutral working
posture and writes it to `configs/operator_calibration.yaml`, which every later
teleop run loads. That is what makes the human-to-robot mapping the same on
Monday as it was on Friday, and the same in episode 40 as in episode 1.

Do it once per operator, and again whenever your seating or the headset's play
space changes.

    python orca_am200/scripts/calibrate_operator.py

Then, in the Quest browser, open the printed URL, start hand tracking, hold your
working hand in a comfortable neutral posture, and keep still until it finishes.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np

from orca_teleop.am200.operator_cal import OperatorCalibration, default_calibration_path
from orca_teleop.ingress.metaquest.bridge import QuestTelemetryBridge

logger = logging.getLogger("calibrate_operator")

# Two seconds at 30 Hz. Long enough for the median to reject tracking blips,
# short enough that holding still is not a chore.
DEFAULT_SAMPLES = 60
# Hand tracking jitters by a couple of millimetres at rest. Ten times that is
# deliberate movement, not jitter, and the capture should be rejected.
MAX_SPREAD_M = 0.02


def collect(bridge: QuestTelemetryBridge, handedness: str, samples: int, fps: int) -> np.ndarray:
    """Gather distinct wrist positions, one per bridge update."""
    positions: list[np.ndarray] = []
    last_sequence: int | None = None
    period = 1.0 / fps
    next_note = time.monotonic() + 5.0

    while len(positions) < samples:
        sample = bridge.state.get_hand_sample(handedness)
        if sample is not None and sample.sequence_id != last_sequence:
            last_sequence = sample.sequence_id
            if sample.wrist_matrix is not None:
                positions.append(np.asarray(sample.wrist_matrix, dtype=np.float64)[:3, 3])
                if len(positions) % 15 == 0:
                    logger.info("  %d / %d samples", len(positions), samples)
        elif time.monotonic() >= next_note:
            logger.warning(
                "Waiting for %s-hand frames. Open the page in Quest Browser, tap Start "
                "hand tracking, and keep the hand visible.",
                handedness,
            )
            next_note = time.monotonic() + 5.0
        time.sleep(period)

    return np.asarray(positions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--output", default=None, help=f"defaults to {default_calibration_path()}")
    parser.add_argument(
        "--force",
        action="store_true",
        help="write even if you moved too much during the capture",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    bridge = QuestTelemetryBridge(host=args.host, port=args.port)
    bridge.start()
    try:
        logger.info("Quest WebXR page: %s", bridge.url)
        logger.info(
            "Over USB: `adb reverse tcp:%d tcp:%d`, then open http://localhost:%d in "
            "Quest Browser.",
            args.port,
            args.port,
            args.port,
        )
        logger.info("")
        logger.info("Hold your %s hand in a comfortable neutral posture and keep still.", args.hand)

        positions = collect(bridge, args.hand, args.samples, args.fps)
    finally:
        bridge.stop()

    calibration, spread_m = OperatorCalibration.capture(positions, args.hand)
    logger.info("")
    logger.info(
        "neutral wrist (FLU, m) : %s", np.round(calibration.neutral_wrist_flu_m, 4).tolist()
    )
    logger.info("spread (95th pct)      : %.1f mm", spread_m * 1000)

    if spread_m > MAX_SPREAD_M and not args.force:
        logger.error(
            "You moved %.0f mm during the capture, over the %.0f mm limit. That would bake "
            "your drift into every future run. Try again holding still, or pass --force.",
            spread_m * 1000,
            MAX_SPREAD_M * 1000,
        )
        return 1

    path = calibration.save(args.output)
    logger.info("written to %s", path)
    logger.info("")
    logger.info("Teleop runs now load this automatically. Re-run it if your seating changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
