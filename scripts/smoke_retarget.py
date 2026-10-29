"""Offline retargeter smoke test: no camera, no hardware.

Builds the adaptive_analytical retargeter from the v1 Feetech config plus the
v1 right-hand URDF and drives it with a synthetic MediaPipe hand that opens and
closes, so the URDF/frame wiring can be checked before plugging anything in.
"""

import os
import sys

import numpy as np

from orca_teleop.retargeting import Retargeter, TargetPose

MODEL_PATH = os.environ["ORCA_MODEL_PATH"]
URDF_PATH = os.environ["ORCA_URDF_PATH"]
RETARGET_CONFIG = os.environ["ORCA_RETARGET_CONFIG"]

# MediaPipe's 21-point layout: wrist, then thumb..pinky as 4 points each.
_FINGER_ROOT_X = [-0.03, -0.015, 0.0, 0.015, 0.03]
_SEGMENT_LEN = 0.035


def synthetic_hand(curl: float) -> np.ndarray:
    """A right hand with every finger curled by ``curl`` in [0, 1]."""
    points = [np.array([0.0, 0.0, 0.0])]
    for root_x in _FINGER_ROOT_X:
        pos = np.array([root_x, 0.02, 0.0])
        angle = 0.0
        for _ in range(4):
            angle += curl * 0.55
            pos = pos + _SEGMENT_LEN * np.array([0.0, np.cos(angle), -np.sin(angle)])
            points.append(pos.copy())
    return np.asarray(points)


def main() -> int:
    retargeter = Retargeter.from_paths(
        model_path=MODEL_PATH,
        urdf_path=URDF_PATH,
        backend="adaptive_analytical",
        config_path=RETARGET_CONFIG,
    )
    print(f"retargeter built: {type(retargeter).__name__}")

    # The retargeter returns None while it auto-scales, so hold a fixed pose.
    calib_frames = 0
    for _ in range(120):
        if retargeter.retarget(TargetPose(synthetic_hand(0.0))) is not None:
            break
        calib_frames += 1
    else:
        print("FAIL: retargeter never finished calibration", file=sys.stderr)
        return 1
    print(f"calibration finished after {calib_frames} frames")

    results = {}
    for label, curl in (("open", 0.0), ("half", 0.5), ("closed", 1.0)):
        out = retargeter.retarget(TargetPose(synthetic_hand(curl)))
        if out is None:
            print(f"FAIL: no output for {label}", file=sys.stderr)
            return 1
        results[label] = dict(out)

    for joint in ("index_mcp", "index_pip", "middle_mcp", "thumb_mcp", "wrist"):
        row = "  ".join(f"{k}={results[k][joint]:7.2f}" for k in results)
        print(f"{joint:12s} {row}")

    spread = max(
        abs(results["closed"][j] - results["open"][j]) for j in results["open"]
    )
    if spread < 1.0:
        print(f"FAIL: joints barely moved (max delta {spread:.3f} deg)", file=sys.stderr)
        return 1
    print(f"OK: max joint delta open->closed = {spread:.1f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
