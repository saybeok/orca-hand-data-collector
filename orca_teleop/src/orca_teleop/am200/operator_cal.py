"""Persisted operator calibration: the reproducible half of the arm mapping.

Modelled on `orca_core`'s `calibration.yaml`: an explicit user-invoked step
writes a YAML file, everything afterwards loads it, and a missing file is a
warning rather than an error. That shape is what makes a calibration
reproducible across episodes and across sessions, which an anchor captured at
process start can never be.

What it stores is one vector: where the operator's wrist sits when they are in
their neutral posture, in the publisher's FLU world frame. Pairing that with the
arm's home pose fixes the whole mapping, so the same operator posture commands
the same robot pose on every run.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)

CALIBRATION_ENV = "ORCA_OPERATOR_CALIBRATION"
FORMAT_VERSION = 1


def default_calibration_path() -> Path:
    """Alongside the hand config, which is where this machine's setup lives."""
    override = os.environ.get(CALIBRATION_ENV)
    if override:
        return Path(override)
    root = Path(os.environ.get("ORCA_HAND_ROOT", Path.home() / "orca_hand"))
    return root / "configs" / "operator_calibration.yaml"


@dataclass(frozen=True)
class OperatorCalibration:
    """The operator's neutral wrist position, and what it was captured with."""

    neutral_wrist_flu_m: np.ndarray
    handedness: str
    samples: int
    captured_at: str

    def __post_init__(self) -> None:
        if self.neutral_wrist_flu_m.shape != (3,):
            raise ValueError(
                f"neutral_wrist_flu_m must have shape (3,), got {self.neutral_wrist_flu_m.shape}"
            )
        if not np.all(np.isfinite(self.neutral_wrist_flu_m)):
            raise ValueError("neutral_wrist_flu_m contains non-finite values")
        if self.handedness not in ("left", "right"):
            raise ValueError(f"handedness must be 'left' or 'right', got {self.handedness!r}")

    @classmethod
    def capture(
        cls, wrist_positions: np.ndarray, handedness: str
    ) -> tuple[OperatorCalibration, float]:
        """Reduce held-still wrist samples to a neutral, plus their spread.

        The median resists the odd tracking outlier. The spread is returned
        rather than judged here so the caller can refuse a capture the operator
        fidgeted through.
        """
        samples = np.asarray(wrist_positions, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1] != 3:
            raise ValueError(f"expected an (N, 3) array of positions, got {samples.shape}")
        if len(samples) < 2:
            raise ValueError(f"need at least 2 samples to measure spread, got {len(samples)}")

        neutral = np.median(samples, axis=0)
        spread_m = float(np.percentile(np.linalg.norm(samples - neutral, axis=1), 95))
        return (
            cls(
                neutral_wrist_flu_m=neutral,
                handedness=handedness,
                samples=len(samples),
                captured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
            spread_m,
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> OperatorCalibration | None:
        """Read the calibration, or return None when there is none to read.

        Returns None rather than raising for a missing file, so a caller can
        fall back and say so — the same choice `orca_core` makes for a missing
        `calibration.yaml`.
        """
        path = Path(path) if path is not None else default_calibration_path()
        if not path.is_file():
            return None

        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        version = document.get("version")
        if version != FORMAT_VERSION:
            raise ValueError(
                f"{path} has format version {version!r}, expected {FORMAT_VERSION}. "
                "Re-run the operator calibration."
            )

        return cls(
            neutral_wrist_flu_m=np.asarray(document["neutral_wrist_flu_m"], dtype=np.float64),
            handedness=document["handedness"],
            samples=int(document.get("samples", 0)),
            captured_at=str(document.get("captured_at", "unknown")),
        )

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path is not None else default_calibration_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        document = {
            "version": FORMAT_VERSION,
            "handedness": self.handedness,
            "neutral_wrist_flu_m": [float(value) for value in self.neutral_wrist_flu_m],
            "samples": int(self.samples),
            "captured_at": self.captured_at,
        }
        header = (
            "# Operator calibration for the AM-ARM200 arm.\n"
            "# Where this operator's wrist sits in their neutral posture, in the\n"
            "# publisher's FLU frame. Paired with the arm's home pose it fixes the\n"
            "# human-to-robot mapping, so the same posture commands the same robot\n"
            "# pose on every run. Regenerate with:\n"
            "#   python orca_am200/scripts/calibrate_operator.py\n"
        )
        # Written whole rather than merged: unlike orca_core's per-joint file
        # there are no independently-captured parts to preserve, so a partial
        # overwrite could only ever mix two operators' postures.
        path.write_text(header + yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        return path
