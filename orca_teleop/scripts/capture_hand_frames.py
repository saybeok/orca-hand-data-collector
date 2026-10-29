"""Record raw Quest WebXR hand frames for offline retargeting evaluation.

The live teleop path hands these exact keypoints to the retargeter, so a
capture replays bit-for-bit through ``eval_retarget_pinch.py`` without the
headset, the robot, or the simulator.

    adb reverse tcp:8765 tcp:8765
    python scripts/capture_hand_frames.py --name pinch_01

Hold a relaxed open hand for ~2 s first (the retargeter auto-scales on the
first frames), then perform the motions; Ctrl-C saves.
"""

from __future__ import annotations

import argparse
import multiprocessing
import queue
import threading
import time
from pathlib import Path

import numpy as np

from orca_teleop.constants import DEFAULT_PORT
from orca_teleop.ingress.server import HandLandmarks, IngressServer
from orca_teleop.pipeline import _metaquest_publisher

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "hand_captures"
PROMPT = """\
Recording. In the headset:
  1. relaxed open hand, still, ~2 s   (auto-scale)
  2. thumb-index pinch x5             (touch, release)
  3. thumb-middle, thumb-ring, thumb-pinky pinch x3 each
  4. fist / open x3
Ctrl-C to save."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--quest-port", type=int, default=8765)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    out = args.out_dir / f"{args.name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    landmarks_q: queue.Queue = queue.Queue(maxsize=4096)
    stop = threading.Event()
    server = IngressServer(landmarks_q, stop, port=args.port)
    server.start()
    publisher = multiprocessing.get_context("spawn").Process(
        target=_metaquest_publisher,
        args=(args.port, args.hand, "0.0.0.0", args.quest_port, args.fps),
        daemon=True,
    )
    publisher.start()
    print(f"Open http://localhost:{args.quest_port} in Quest Browser and start hand tracking.")
    print(PROMPT)

    frames: list[HandLandmarks] = []
    last_report = time.monotonic()
    try:
        while True:
            try:
                item = landmarks_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if isinstance(item, HandLandmarks) and item.handedness == args.hand:
                frames.append(item)
            if time.monotonic() - last_report > 2.0:
                print(f"  {len(frames)} frames", flush=True)
                last_report = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.stop()
        publisher.terminate()
        publisher.join(timeout=3.0)

    if not frames:
        print("no frames captured; nothing saved")
        return 1
    np.savez_compressed(
        out,
        keypoints=np.stack([f.keypoints for f in frames]).astype(np.float32),
        wrist_angle_degrees=np.array([f.wrist_angle_degrees for f in frames], dtype=np.float32),
        recalibrate=np.array([f.retarget_recalibrate for f in frames], dtype=bool),
        timestamp_ns=np.array([f.timestamp_ns for f in frames], dtype=np.int64),
        handedness=args.hand,
    )
    print(f"saved {len(frames)} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
