"""Landmark and wrist adapters for the repository-owned Meta Quest WebXR client."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

WEBXR_HAND_JOINT_NAMES = (
    "wrist",
    "thumb-metacarpal",
    "thumb-phalanx-proximal",
    "thumb-phalanx-distal",
    "thumb-tip",
    "index-finger-metacarpal",
    "index-finger-phalanx-proximal",
    "index-finger-phalanx-intermediate",
    "index-finger-phalanx-distal",
    "index-finger-tip",
    "middle-finger-metacarpal",
    "middle-finger-phalanx-proximal",
    "middle-finger-phalanx-intermediate",
    "middle-finger-phalanx-distal",
    "middle-finger-tip",
    "ring-finger-metacarpal",
    "ring-finger-phalanx-proximal",
    "ring-finger-phalanx-intermediate",
    "ring-finger-phalanx-distal",
    "ring-finger-tip",
    "pinky-finger-metacarpal",
    "pinky-finger-phalanx-proximal",
    "pinky-finger-phalanx-intermediate",
    "pinky-finger-phalanx-distal",
    "pinky-finger-tip",
)

# WebXR has one extra metacarpal joint for every non-thumb finger. The
# retargeter consumes the MediaPipe/MANO-like 21-point layout, whose MCP point
# corresponds to WebXR's phalanx-proximal joint.
WEBXR_TO_RETARGETER_LANDMARK_INDICES = (
    0,
    1,
    2,
    3,
    4,
    6,
    7,
    8,
    9,
    11,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    21,
    22,
    23,
    24,
)

# WebXR: right-handed, X right, Y up, -Z forward.
# Robot/MuJoCo FLU: X forward, Y left, Z up.
XR_TO_FLU_BASIS = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def retargeter_landmarks_from_webxr(points: np.ndarray, handedness: str) -> np.ndarray:
    """Reduce live WebXR's 25 joints to the retargeter's 21-joint layout.

    WebXR already reports a right-handed point cloud, so this adapter only
    removes the four extra non-thumb metacarpal joints. It deliberately does
    not mirror either hand.
    """
    if handedness not in ("left", "right"):
        raise ValueError(f"Unsupported Meta Quest handedness: {handedness!r}")
    landmarks = np.asarray(points, dtype=np.float64)
    if landmarks.shape != (25, 3):
        raise ValueError(f"Expected WebXR landmarks with shape (25, 3), got {landmarks.shape}")
    return landmarks[list(WEBXR_TO_RETARGETER_LANDMARK_INDICES)].copy()


def retargeter_landmarks_from_quest(points: np.ndarray, handedness: str) -> np.ndarray:
    """Adapt legacy recorded HTS/Unity landmarks.

    This remains for replaying the old ``quest-poses`` datasets. Live Quest
    Browser frames must use :func:`retargeter_landmarks_from_webxr` instead.
    """
    if handedness not in ("left", "right"):
        raise ValueError(f"Unsupported Meta Quest handedness: {handedness!r}")
    landmarks = np.asarray(points, dtype=np.float64).copy()
    if landmarks.ndim != 2 or landmarks.shape[1] != 3:
        raise ValueError(f"Meta Quest landmarks must have shape (N, 3); got {landmarks.shape}")
    if handedness == "right":
        landmarks[:, 1] *= -1.0
    return landmarks


def xr_flat_matrix_to_numpy(raw_matrix: Sequence[float]) -> np.ndarray:
    """Decode a column-major WebXR transform matrix."""
    matrix = np.asarray(raw_matrix, dtype=np.float64)
    if matrix.size != 16:
        raise ValueError(f"Expected 16 values for a WebXR pose matrix, got {matrix.size}")
    return matrix.reshape(4, 4).T


def xr_matrix_to_flu_matrix(raw_matrix: Sequence[float]) -> np.ndarray:
    """Convert a WebXR transform to the robot's forward-left-up basis."""
    xr_matrix = xr_flat_matrix_to_numpy(raw_matrix)
    flu_matrix = np.eye(4, dtype=np.float64)
    flu_matrix[:3, :3] = XR_TO_FLU_BASIS @ xr_matrix[:3, :3] @ XR_TO_FLU_BASIS.T
    flu_matrix[:3, 3] = XR_TO_FLU_BASIS @ xr_matrix[:3, 3]
    return flu_matrix


THUMB_TIP_INDEX = WEBXR_HAND_JOINT_NAMES.index("thumb-tip")
INDEX_TIP_INDEX = WEBXR_HAND_JOINT_NAMES.index("index-finger-tip")

# A pinch closes to roughly 20 mm and an open hand sits well past 60 mm. The gap
# between these two is deliberate: a single threshold chatters on and off around
# the boundary, which on a clutch means the arm stuttering between following and
# frozen.
PINCH_CLOSE_M = 0.030
PINCH_OPEN_M = 0.045

FIST_TIP_INDICES = (
    WEBXR_HAND_JOINT_NAMES.index("index-finger-tip"),
    WEBXR_HAND_JOINT_NAMES.index("middle-finger-tip"),
    WEBXR_HAND_JOINT_NAMES.index("ring-finger-tip"),
    WEBXR_HAND_JOINT_NAMES.index("pinky-finger-tip"),
)
WRIST_INDEX = WEBXR_HAND_JOINT_NAMES.index("wrist")
# Mean tip-to-wrist distance: fist ~5 cm, open hand ~12+ cm.
FIST_CLOSE_M = 0.070
FIST_OPEN_M = 0.100


class PinchDetector:
    """Latching thumb-to-index pinch, used as a clutch on the operator's off hand.

    Latching matters more than the exact thresholds. The clutch decides whether
    the robot moves at all, so a detector that flickers is worse than one that is
    slightly slow to release.
    """

    def __init__(self, close_m: float = PINCH_CLOSE_M, open_m: float = PINCH_OPEN_M) -> None:
        if not 0.0 < close_m < open_m:
            raise ValueError(f"need 0 < close_m < open_m, got close={close_m}, open={open_m}")
        self._close_m = float(close_m)
        self._open_m = float(open_m)
        self._engaged = False

    @property
    def engaged(self) -> bool:
        return self._engaged

    def reset(self) -> None:
        self._engaged = False

    def update(self, landmarks: np.ndarray | None) -> bool:
        """Advance the latch. ``None`` (hand not tracked) always releases.

        Releasing on a lost hand is the fail-safe direction: an untracked off
        hand cannot be giving permission to move.
        """
        if landmarks is None:
            self._engaged = False
            return False

        points = np.asarray(landmarks, dtype=np.float64)
        if points.shape != (25, 3):
            raise ValueError(f"Expected WebXR landmarks with shape (25, 3), got {points.shape}")

        gap = float(np.linalg.norm(points[THUMB_TIP_INDEX] - points[INDEX_TIP_INDEX]))
        if self._engaged:
            if gap > self._open_m:
                self._engaged = False
        elif gap < self._close_m:
            self._engaged = True
        return self._engaged


class FistDetector:
    """Latching fist on the off hand — rising edge triggers retarget recalibration.

    Distinct from :class:`PinchDetector` so clutch (thumb–index pinch) and
    hand re-sync (fist) do not fight each other.
    """

    def __init__(self, close_m: float = FIST_CLOSE_M, open_m: float = FIST_OPEN_M) -> None:
        if not 0.0 < close_m < open_m:
            raise ValueError(f"need 0 < close_m < open_m, got close={close_m}, open={open_m}")
        self._close_m = float(close_m)
        self._open_m = float(open_m)
        self._engaged = False

    @property
    def engaged(self) -> bool:
        return self._engaged

    def reset(self) -> None:
        self._engaged = False

    def update(self, landmarks: np.ndarray | None) -> bool:
        """Return whether the hand is currently in a fist. ``None`` releases."""
        if landmarks is None:
            self._engaged = False
            return False

        points = np.asarray(landmarks, dtype=np.float64)
        if points.shape != (25, 3):
            raise ValueError(f"Expected WebXR landmarks with shape (25, 3), got {points.shape}")

        wrist = points[WRIST_INDEX]
        mean_gap = float(
            np.mean([np.linalg.norm(points[i] - wrist) for i in FIST_TIP_INDICES])
        )
        if self._engaged:
            if mean_gap > self._open_m:
                self._engaged = False
        elif mean_gap < self._close_m:
            self._engaged = True
        return self._engaged

    def rising_edge(self, landmarks: np.ndarray | None) -> bool:
        """True only on the frame the fist latch closes."""
        was = self._engaged
        now = self.update(landmarks)
        return now and not was


class WristAngleEstimator:
    """Reproduce the WebXR demonstration pipeline's relative wrist control."""

    def __init__(self) -> None:
        self._zero_pitch_rad: float | None = None

    @staticmethod
    def _hand_x_pitch_rad(wrist_matrix: np.ndarray) -> float:
        matrix = np.asarray(wrist_matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected wrist matrix with shape (4, 4), got {matrix.shape}")
        x_in_world = matrix[:3, 0]
        return float(np.arcsin(np.clip(x_in_world[2], -1.0, 1.0)))

    def reset(self) -> None:
        self._zero_pitch_rad = None

    def update(self, wrist_matrix: np.ndarray | None) -> float:
        if wrist_matrix is None:
            return 0.0
        pitch = self._hand_x_pitch_rad(wrist_matrix)
        if self._zero_pitch_rad is None:
            self._zero_pitch_rad = pitch
            return 0.0
        return float(np.degrees(self._zero_pitch_rad - pitch))
