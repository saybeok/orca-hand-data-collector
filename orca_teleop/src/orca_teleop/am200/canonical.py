"""Canonical 20-DoF joint contract for the AM-ARM200 + ORCA right hand robot.

Indices 0..2 are the arm, 3..19 the hand, in the order recorded in
``joint_map.json``. Everything downstream — simulation, hardware, LeRobot
recording — is expected to speak this one ordering.

Two angle conventions meet here and must not be confused:

``physical``
    What ``orca_core`` and the retargeter use for the hand: degrees, measured
    from the hand's own zero pose.

``model``
    MuJoCo qpos in radians, offset by each joint's ``ref``. This is NOT the
    URDF hinge rotation: URDF FK uses ``qpos - ref`` for the same geometry.
    MuJoCo applies ``qpos - ref`` as the joint's spatial rotation, so a physical
    angle of zero corresponds to ``qpos == ref``, giving

        qpos = deg2rad(physical) + ref

    Skipping the ``+ ref`` term silently mis-drives the seven v1 joints that
    have a non-zero ``ref`` (both abduction joints of the ring and pinky, the
    index and thumb abduction, and the thumb pip/dip), by up to 42 degrees.

The canonical vector is in the ``model`` convention, because the arm half comes
from IK in radians and mixing units in one vector invites exactly the class of
bug described above.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from orca_teleop.retargeting.urdf_offsets import (
    PASSTHROUGH_JOINTS,
    model_to_physical,
    physical_to_model,
)

if TYPE_CHECKING:
    from orca_core import OrcaJointPositions

ARM_ROLES: tuple[str, ...] = (
    "arm.shoulder_pan",
    "arm.shoulder_lift",
    "arm.elbow_flex",
)

# The model's wrist carries a -8.6 degree `ref`, but honouring it here would put
# this sim that far from the hand-only path, which zeroes the wrist offset (see
# PASSTHROUGH_JOINTS). Unlike the finger joints, the wrist ROM does not reconcile
# with the model range under either sign, so this follows upstream rather than
# inventing a third convention.


@dataclass(frozen=True)
class RobotJointCommand:
    """A full-robot joint target in the canonical order and ``model`` convention."""

    joint_ids: tuple[str, ...]
    position: np.ndarray
    timestamp_ns: int

    def __post_init__(self) -> None:
        if self.position.shape != (len(self.joint_ids),):
            raise ValueError(
                f"position has shape {self.position.shape} but there are "
                f"{len(self.joint_ids)} joint ids"
            )

    @classmethod
    def now(cls, joint_ids: tuple[str, ...], position: np.ndarray) -> RobotJointCommand:
        return cls(
            joint_ids=joint_ids,
            position=np.asarray(position, dtype=np.float64),
            timestamp_ns=time.time_ns(),
        )


@dataclass(frozen=True)
class CanonicalJointMap:
    """The generated joint map, bound to a compiled MuJoCo model.

    ``actuator_index`` and ``qpos_index`` map canonical positions onto the
    model's own ordering, so no caller has to recompute a permutation.
    """

    roles: tuple[str, ...]
    hand_joint_ids: tuple[str, ...]
    actuator_index: np.ndarray
    qpos_index: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    ref: np.ndarray

    @property
    def arm_dof(self) -> int:
        return len(ARM_ROLES)

    @property
    def hand_dof(self) -> int:
        return len(self.hand_joint_ids)

    @property
    def dof(self) -> int:
        return len(self.roles)

    @property
    def arm_slice(self) -> slice:
        return slice(0, self.arm_dof)

    @property
    def hand_slice(self) -> slice:
        return slice(self.arm_dof, self.dof)

    @classmethod
    def load(cls, joint_map_path: str | Path, model: Any) -> CanonicalJointMap:
        """Read ``joint_map.json`` and check it still describes ``model``.

        Raises rather than silently reordering: a stale map is the one failure
        mode that would quietly drive the wrong joints.
        """
        import mujoco

        payload = json.loads(Path(joint_map_path).read_text(encoding="utf-8"))
        entries = payload["canonical_order"]
        if len(entries) != model.nu:
            raise ValueError(
                f"{joint_map_path} describes {len(entries)} DoF but the model has {model.nu}"
            )

        roles, hand_ids = [], []
        actuator_index, qpos_index = [], []
        lower, upper, ref = [], [], []
        for index, entry in enumerate(entries):
            if entry["index"] != index:
                raise ValueError(f"joint map is not in index order at position {index}")
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, entry["mujoco_joint"])
            if joint_id < 0:
                raise ValueError(f"model has no joint {entry['mujoco_joint']!r}")
            actuator = entry["model_actuator_index"]
            if int(model.actuator_trnid[actuator, 0]) != joint_id:
                raise ValueError(
                    f"joint map points {entry['role']} at actuator {actuator}, which drives a "
                    f"different joint than {entry['mujoco_joint']!r}"
                )

            core_id = entry["orca_core_joint_id"]
            roles.append(entry["role"])
            if core_id is not None:
                hand_ids.append(core_id)
            actuator_index.append(actuator)
            qpos_adr = int(model.jnt_qposadr[joint_id])
            qpos_index.append(qpos_adr)
            lower.append(model.jnt_range[joint_id][0])
            upper.append(model.jnt_range[joint_id][1])
            ref.append(0.0 if core_id in PASSTHROUGH_JOINTS else model.qpos0[qpos_adr])

        if tuple(roles[: len(ARM_ROLES)]) != ARM_ROLES:
            raise ValueError(f"expected the arm at indices 0..2, found {roles[: len(ARM_ROLES)]}")

        return cls(
            roles=tuple(roles),
            hand_joint_ids=tuple(hand_ids),
            actuator_index=np.asarray(actuator_index, dtype=np.int64),
            qpos_index=np.asarray(qpos_index, dtype=np.int64),
            lower=np.asarray(lower, dtype=np.float64),
            upper=np.asarray(upper, dtype=np.float64),
            ref=np.asarray(ref, dtype=np.float64),
        )

    def clip(self, canonical: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(canonical, dtype=np.float64), self.lower, self.upper)

    def to_ctrl(self, canonical: np.ndarray) -> np.ndarray:
        """Canonical order -> the model's actuator order."""
        ctrl = np.empty(self.dof, dtype=np.float64)
        ctrl[self.actuator_index] = canonical
        return ctrl

    def from_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """The model's qpos -> canonical order."""
        return np.asarray(qpos, dtype=np.float64)[self.qpos_index]

    def hand_from_physical(self, positions: OrcaJointPositions) -> np.ndarray:
        """orca_core physical degrees -> model radians, in canonical hand order."""
        degrees = np.asarray(positions.as_array(list(self.hand_joint_ids)), dtype=np.float64)
        return physical_to_model(np.deg2rad(degrees), self.ref[self.hand_slice])

    def hand_to_physical(self, hand_model_rad: np.ndarray) -> dict[str, float]:
        """model radians -> orca_core physical degrees, keyed by joint id."""
        model_rad = np.asarray(hand_model_rad, dtype=np.float64)
        degrees = np.rad2deg(model_to_physical(model_rad, self.ref[self.hand_slice]))
        return dict(zip(self.hand_joint_ids, degrees.tolist(), strict=True))
