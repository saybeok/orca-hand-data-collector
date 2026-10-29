"""Fit effective fingertip contact points from confirmed physical contacts.

These local-frame corrections are not joint encoder calibration or a model of
tendon friction. Several different contact poses per pair, held-out checks,
and a small correction envelope are required before a profile is usable.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
MAX_OFFSET_M = 0.015
MAX_VALIDATION_ERROR_M = 0.003


@dataclass(frozen=True)
class TaughtContactTarget:
    finger: str
    pose: int
    indices: np.ndarray
    angles: np.ndarray
    strength: float


class TaughtContacts:
    """Use confirmed contact configurations without claiming a geometric fit.

    Select an actual sample, never an average of contact poses. Select on entry
    from the current preshape and latch until release to avoid pose hopping
    while holding an object. Multiple simultaneous full pinches use the base
    retargeter, since independent pair samples cannot describe that contact.
    """
    def __init__(self, path: Path, retargeter):
        dataset = json.loads(path.read_text())
        if dataset.get("schema") != 1 or dataset.get("fingerprint") != fingerprint(retargeter):
            raise ValueError("Taught contacts are stale for the current calibration/model")
        joints = retargeter._finger_joint_ids
        samples = dataset.get("samples", [])
        self.poses = {}
        self.indices = {}
        self.scale = retargeter._upper_phys - retargeter._lower_phys
        for sample in samples:
            if (sample.get("finger") not in FINGERS[1:]
                    or sample.get("operator_confirmed_contact") is not True):
                raise ValueError("Taught contacts require operator-confirmed physical contacts")
            pose_frames(retargeter, sample["angles_deg"])
        for f in FINGERS[1:]:
            rows = [s for s in samples if s["finger"] == f]
            if len(rows) < 6:
                raise ValueError(f"{f}: at least six confirmed contact poses are required")
            self.poses[f] = np.deg2rad([[s["angles_deg"][j] for j in joints] for s in rows])
            self.indices[f] = np.array([i for i, j in enumerate(joints)
                                       if j.startswith(("thumb_", f + "_"))], dtype=int)
        self.reset()

    def reset(self):
        self._finger = None
        self._pose = None

    def target(self, keypoints, qpos, d1, d2, enabled):
        gaps = np.linalg.norm(keypoints[[8, 12, 16, 20]] - keypoints[4], axis=1) * 100
        enabled = np.asarray(enabled) > 0
        if np.count_nonzero((gaps <= d1) & enabled) > 1:
            self.reset()
            return None
        if self._finger is not None:
            old = FINGERS[1:].index(self._finger)
            closest = int(np.argmin(np.where(enabled, gaps, np.inf)))
            # Switch only when another pair is decisively closer. A release
            # resets the sample selection even if another pair stays near.
            if (not enabled[old] or gaps[old] >= d2[old]
                    or (closest != old and gaps[closest] + .5 < gaps[old])):
                self.reset()
        if self._finger is None:
            scores = np.where(enabled & (gaps < d2), gaps, np.inf)
            if not np.isfinite(scores).any():
                return None
            pair = int(np.argmin(scores))
            self._finger = FINGERS[pair + 1]
            idx = self.indices[self._finger]
            poses = self.poses[self._finger]
            error = (poses[:, idx] - qpos[idx]) / self.scale[idx]
            self._pose = int(np.argmin(np.sum(error**2, axis=1)))
        pair = FINGERS[1:].index(self._finger)
        amount = float(np.clip((d2[pair]-gaps[pair])/(d2[pair]-d1[pair]), 0, 1))
        strength = amount**2 * (3-2*amount)
        idx = self.indices[self._finger]
        return TaughtContactTarget(self._finger, self._pose, idx,
                                  self.poses[self._finger][self._pose, idx].copy(), strength)


def fingerprint(retargeter) -> str:
    hand = retargeter._hand
    payload = {
        "type": hand.config.type,
        "map": hand.config.joint_to_motor_map,
        "inversion": hand.config.joint_inversion_dict,
        "rom": hand.config.joint_roms_dict,
        "limits": hand.motor_limits_dict,
        "ratios": hand.joint_to_motor_ratios_dict,
        "frames": list(retargeter._frame_map.tip),
        "urdf_sha256": hashlib.sha256(Path(retargeter.urdf_path).read_bytes()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_offsets(path: Path, retargeter) -> dict[str, np.ndarray]:
    profile = json.loads(path.read_text())
    if profile.get("schema") != 1 or profile.get("fingerprint") != fingerprint(retargeter):
        raise ValueError("Contact calibration is incompatible or stale; recapture after hardware calibration/model changes")
    if profile.get("validated") is not True:
        raise ValueError("Contact calibration has not passed held-out validation")
    raw = profile.get("offsets_m", {})
    if set(raw) != set(FINGERS):
        raise ValueError("Contact calibration must contain all five fingers")
    offsets = {f: np.asarray(raw[f], dtype=float) for f in FINGERS}
    for f, point in offsets.items():
        if point.shape != (3,) or not np.isfinite(point).all() or np.linalg.norm(point) > MAX_OFFSET_M:
            raise ValueError(f"Invalid or excessive contact correction: {f}")
    validation = profile.get("validation_errors_m", {})
    if set(validation) != set(FINGERS[1:]):
        raise ValueError("Missing held-out contact validation")
    for errors in validation.values():
        values = np.asarray(errors, dtype=float)
        if (values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all()
                or np.any(values < 0) or np.any(values > MAX_VALIDATION_ERROR_M)):
            raise ValueError("Contact validation exceeds the 3 mm acceptance envelope")
    return offsets


def pose_frames(retargeter, angles: dict) -> tuple[np.ndarray, np.ndarray]:
    joints = retargeter._finger_joint_ids
    if set(angles) != set(joints):
        raise ValueError("A contact sample must include every configured finger joint")
    q = np.deg2rad([angles[j] for j in joints])
    if (not np.isfinite(q).all() or np.any(q < retargeter._lower_phys - 1e-6)
            or np.any(q > retargeter._upper_phys + 1e-6)):
        raise ValueError("Contact sample has nonfinite or out-of-ROM angles")
    indices = [retargeter._computed_frame_indices[i] for i in retargeter._tip_indices]
    positions, _ = retargeter._robot.compute_positions_and_jacobians(
        retargeter._full_model_qpos(q), indices,
    )
    rotations = np.array([retargeter._robot.data.oMf[i].rotation.copy() for i in indices])
    return positions, rotations


def fit_offsets(samples: list[dict]) -> dict:
    """Least squares on p_f + R_f d_f = p_thumb + R_thumb d_thumb.

    Last two samples per pair are validation only. At least four differently
    posed training samples per pair must identify all 15 correction variables.
    Thresholds are local engineering checks, not ORCA/paper-prescribed values.
    """
    training, validation = [], []
    for finger in FINGERS[1:]:
        pair = [s for s in samples if s["finger"] == finger]
        if len(pair) < 6:
            raise ValueError(f"{finger}: need six distinct confirmed contact poses (four fit, two validation)")
        training.extend(pair[:-2])
        validation.extend(pair[-2:])

    def equation(sample):
        positions = np.asarray(sample["positions_m"], dtype=float)
        rotations = np.asarray(sample["rotations"], dtype=float)
        if (positions.shape != (5, 3) or rotations.shape != (5, 3, 3)
                or not np.isfinite(positions).all() or not np.isfinite(rotations).all()
                or not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-6)
                or not np.allclose(np.linalg.det(rotations), 1, atol=1e-6)):
            raise ValueError("Invalid contact-frame transforms")
        i = FINGERS.index(sample["finger"])
        a = np.zeros((3, 15))
        a[:, :3] = -rotations[0]
        a[:, 3*i:3*i+3] = rotations[i]
        return a, positions[0] - positions[i]

    equations = [equation(s) for s in training]
    a, b = np.concatenate([e[0] for e in equations]), np.concatenate([e[1] for e in equations])
    singular = np.linalg.svd(a, compute_uv=False)
    if singular[-1] <= singular[0] / 100:
        raise ValueError("Contact poses do not identify the corrections: vary both thumb and finger bends, not just wrist position")
    solution = np.linalg.lstsq(a, b, rcond=None)[0]
    offsets = solution.reshape(5, 3)
    if np.any(np.linalg.norm(offsets, axis=1) > MAX_OFFSET_M):
        raise ValueError("Required correction exceeds 15 mm: check joint mapping/calibration or model geometry first")
    errors = {}
    before = {}
    for s in validation:
        matrix, target = equation(s)
        errors.setdefault(s["finger"], []).append(float(np.linalg.norm(matrix @ solution - target)))
        before.setdefault(s["finger"], []).append(float(np.linalg.norm(target)))
    if any(max(e) > MAX_VALIDATION_ERROR_M for e in errors.values()):
        raise ValueError("Held-out contacts disagree by over 3 mm; a fixed contact-point correction is insufficient")
    # Do not accept a fit that degrades any pair's mean validation error.
    if any(np.mean(errors[f]) > np.mean(before[f]) + 1e-6 for f in errors):
        raise ValueError("Contact correction worsens a held-out finger pair")
    return {
        "schema": 1, "validated": True,
        "offsets_m": dict(zip(FINGERS, offsets.tolist(), strict=True)),
        "validation_errors_m": errors, "before_validation_errors_m": before,
        "condition_number": float(singular[0] / singular[-1]),
        "training_samples": len(training), "validation_samples": len(validation),
        "limitations": "Effective contact points for this assembly/calibration, not measured link angles or force control",
    }
