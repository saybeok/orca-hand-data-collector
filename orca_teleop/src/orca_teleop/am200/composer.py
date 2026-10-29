"""Turn one ingress frame into a full 20-DoF command.

The hand comes from the unchanged upstream retargeter. The arm comes from the
operator's wrist *position*, which reaches the robot side only because
`HandFrame.wrist_transform` now carries the full pose.

Absolute wrist positions cannot be used raw: the Quest's origin is wherever the
headset established its reference space, and the operator's shoulder is not the
robot's base. Something has to relate the two, and *what* relates them decides
whether recorded demonstrations are reproducible.

Two mechanisms do it here, deliberately split:

* At episode start the relation comes from the persisted operator calibration —
  the operator's neutral posture maps to the arm's home pose. Being on disk, it
  is identical across episodes and across sessions, so every episode begins with
  the same human-to-robot mapping.
* Within an episode the clutch may re-anchor it. Releasing the clutch freezes
  the arm, letting the operator reposition their hand and re-engage, the way
  lifting a mouse works. This trades a bounded drift for the operator being
  able to work comfortably.

Splitting them this way is the point: reproducibility is a property of episode
starts, where it is enforced from a file, while comfort is a property of what
happens inside an episode, where drift costs nothing that a robot-absolute
recording would notice.

Without a calibration file the mapping falls back to anchoring on the first
frame, which is *not* reproducible, and says so loudly.

If a frame has no wrist pose — the webcam publisher never sends one — the hand
action is passed through untouched, so the same runner still works hand-only.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from orca_core import OrcaJointPositions

from orca_teleop.am200.arm_ik import FLANGE_SITE, ArmIk
from orca_teleop.am200.canonical import CanonicalJointMap, RobotJointCommand
from orca_teleop.am200.operator_cal import OperatorCalibration
from orca_teleop.ingress.server import HandLandmarks

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArmMappingConfig:
    """How operator wrist motion becomes flange motion.

    `scale` below 1 is the norm: a seated operator sweeps roughly 0.6 m while
    this arm reaches 0.49 m from its base, and the usable shell is smaller
    still.
    """

    scale: float = 0.5
    # Optional spherical shell around the arm base. ``None`` means no software
    # workspace clamp (IK / joint stops still apply).
    min_radius_m: float | None = None
    max_radius_m: float | None = None
    # Operator jitter below this is ignored, so the arm holds still when the
    # hand is nominally at rest.
    deadband_m: float = 0.005
    # A jump larger than this between consecutive frames means tracking was
    # lost and re-acquired somewhere else; re-anchor instead of lunging.
    reanchor_jump_m: float = 0.25
    # Optional command shaping; hardware shell wrappers enable it explicitly.
    max_velocity_rad_s: float | None = None
    max_acceleration_rad_s2: float | None = None

    def __post_init__(self) -> None:
        values = (self.max_velocity_rad_s, self.max_acceleration_rad_s2)
        if (values[0] is None) != (values[1] is None):
            raise ValueError("Arm velocity and acceleration limits must be supplied together")
        if any(v is not None and (not np.isfinite(v) or v <= 0) for v in values):
            raise ValueError("Arm motion limits must be finite and positive")


class HoldArmActionComposer:
    """Hand-only teleop in the 20-DoF schema; the arm holds its home pose."""

    def __init__(self, joint_map: CanonicalJointMap) -> None:
        self._map = joint_map

    def reset(self) -> None:
        pass

    def __call__(
        self,
        frame: HandLandmarks,
        hand_action: OrcaJointPositions,
    ) -> RobotJointCommand:
        canonical = np.zeros(self._map.dof, dtype=np.float64)
        canonical[self._map.hand_slice] = self._map.hand_from_physical(hand_action)
        return RobotJointCommand.now(self._map.roles, self._map.clip(canonical))


class Am200ActionComposer:
    """Composes arm IK and hand retargeting into one canonical command."""

    def __init__(
        self,
        joint_map: CanonicalJointMap,
        arm_ik: ArmIk,
        config: ArmMappingConfig | None = None,
        calibration: OperatorCalibration | None = None,
    ) -> None:
        self._map = joint_map
        self._ik = arm_ik
        self._config = config or ArmMappingConfig()
        self._calibration = calibration

        self._arm_q = np.zeros(len(self._map.roles[self._map.arm_slice]), dtype=np.float64)
        self._home_flange = arm_ik.flange_position(self._arm_q)
        self._operator_origin: np.ndarray | None = None
        self._flange_anchor: np.ndarray | None = None
        self._last_operator: np.ndarray | None = None
        self._engaged = False
        self._align = self._alignment(arm_ik)
        self._unreachable_logged = False
        self._arm_velocity = np.zeros_like(self._arm_q)
        self._last_motion_ns: int | None = None

        # Default teleop is first-frame relative follow. A calibration file is
        # optional absolute mapping for reproducible recording; absence is normal.
        if calibration is not None:
            logger.info(
                "Operator calibration loaded: neutral wrist %s (%d samples, %s)",
                np.round(calibration.neutral_wrist_flu_m, 3).tolist(),
                calibration.samples,
                calibration.captured_at,
            )

    @classmethod
    def from_model_path(
        cls,
        model_path: str | Path,
        joint_map_path: str | Path | None = None,
        config: ArmMappingConfig | None = None,
        site_name: str = FLANGE_SITE,
        calibration: OperatorCalibration | None = None,
    ) -> Am200ActionComposer:
        import mujoco

        model_path = Path(model_path)
        model = mujoco.MjModel.from_xml_path(str(model_path))
        joint_map = CanonicalJointMap.load(
            joint_map_path or model_path.parent / "joint_map.json", model
        )
        return cls(joint_map, ArmIk(model, site_name=site_name), config, calibration)

    @staticmethod
    def _alignment(arm_ik: ArmIk) -> np.ndarray:
        """Rotation taking operator FLU axes to the robot's base frame.

        Derived from the model rather than written down: the robot's forward
        direction is where the flange sits at the home pose, projected onto the
        ground plane. Both frames are z-up, so that one vector fixes the rest.
        """
        home = arm_ik.flange_position(np.zeros(3))
        forward = np.array([home[0], home[1], 0.0])
        norm = np.linalg.norm(forward)
        if norm < 1e-6:
            raise ValueError(
                "the flange sits on the base's vertical axis at the home pose, so the "
                "robot's forward direction is undefined; pass an explicit alignment"
            )
        forward /= norm
        up = np.array([0.0, 0.0, 1.0])
        left = np.cross(up, forward)
        return np.column_stack([forward, left, up])

    @property
    def arm_position(self) -> np.ndarray:
        return self._arm_q.copy()

    def reset(self) -> None:
        """Start a new episode: drop any clutch drift and restore the mapping.

        Called between episodes, this is what makes each one begin from the same
        human-to-robot relation. With a calibration on disk the next frame
        rebuilds that relation from the file; without one it re-anchors on the
        next frame, which is the reproducibility gap the file closes.
        """
        self._operator_origin = None
        self._flange_anchor = None
        self._last_operator = None
        self._engaged = False
        self._arm_velocity[:] = 0
        self._last_motion_ns = None

    def __call__(
        self,
        frame: HandLandmarks,
        hand_action: OrcaJointPositions,
    ) -> RobotJointCommand | OrcaJointPositions:
        if frame.wrist_transform is None:
            self._pause_motion()
            return hand_action

        wrist = np.asarray(frame.wrist_transform, dtype=np.float64)[:3, 3]
        if not np.all(np.isfinite(wrist)):
            self._pause_motion()
            return hand_action

        # A publisher with no clutch means follow unconditionally; a released
        # clutch means hold. The hand keeps moving either way, so releasing
        # parks the arm without freezing the fingers.
        if frame.clutch_supported and not frame.clutch_engaged:
            self._pause_motion()
            return self._command(hand_action)

        dt = 1.0 / 30.0
        if self._config.max_velocity_rad_s is not None:
            now = int(frame.timestamp_ns) if frame.timestamp_ns > 0 else time.monotonic_ns()
            if self._last_motion_ns is not None:
                if now <= self._last_motion_ns:
                    return self._command(hand_action)
                dt = (now - self._last_motion_ns) * 1e-9
                if dt > 0.2:
                    # A tracking gap must not spend the missing time moving.
                    self._anchor(wrist)
                    dt = 0.0
            self._last_motion_ns = now

        if not self._engaged:
            self._engaged = True
            # Re-anchoring here rather than on release keeps the arm still for
            # the whole disengaged interval, however long the operator takes.
            if self._operator_origin is not None:
                self._anchor(wrist)
                logger.info("Clutch re-engaged; arm re-anchored at its current pose.")

        target = self._flange_target(wrist)
        result = self._ik.solve(target, seed=self._arm_q)
        self._advance_arm(result.position, dt)

        if not result.converged and not self._unreachable_logged:
            logger.info(
                "Arm target %.0f mm out of reach; holding at the closest pose. "
                "Lower --arm-scale if this persists.",
                result.residual_m * 1e3,
            )
            self._unreachable_logged = True
        elif result.converged:
            self._unreachable_logged = False

        return self._command(hand_action)

    def _pause_motion(self) -> None:
        self._engaged = False
        self._arm_velocity[:] = 0
        self._last_motion_ns = None

    def _advance_arm(self, target: np.ndarray, dt: float) -> None:
        """Track IK with a damped second-order reference, not per-frame clipping.

        Integrate small substeps for stable behaviour across acquisition rates.
        Bounds apply to the joint command velocity/acceleration, not measured
        hardware motion. Clutch/tracking stops and joint stops override the
        acceleration bound to hold immediately.
        """
        config = self._config
        if config.max_velocity_rad_s is None:
            self._arm_q = target.copy()
            return
        if dt <= 0:
            return
        count = max(1, int(np.ceil(dt * 240.0)))
        h = dt / count
        lower, upper = self._ik.limits
        for _ in range(count):
            # Critical damping at omega=20 rad/s for the unsaturated reference.
            accel = 400.0 * (target - self._arm_q) - 40.0 * self._arm_velocity
            accel = np.clip(accel, -config.max_acceleration_rad_s2, config.max_acceleration_rad_s2)
            velocity = np.clip(self._arm_velocity + h * accel,
                               -config.max_velocity_rad_s, config.max_velocity_rad_s)
            position = self._arm_q + 0.5 * h * (self._arm_velocity + velocity)
            clipped = np.clip(position, lower, upper)
            velocity[position != clipped] = 0.0
            self._arm_q = clipped
            self._arm_velocity = velocity

    def _command(self, hand_action: OrcaJointPositions) -> RobotJointCommand:
        canonical = np.empty(self._map.dof, dtype=np.float64)
        canonical[self._map.arm_slice] = self._arm_q
        canonical[self._map.hand_slice] = self._map.hand_from_physical(hand_action)
        return RobotJointCommand.now(self._map.roles, self._map.clip(canonical))

    def _flange_target(self, wrist: np.ndarray) -> np.ndarray:
        config = self._config

        if self._operator_origin is None or self._flange_anchor is None:
            self._begin_episode(wrist)
        elif np.linalg.norm(wrist - self._last_operator) > config.reanchor_jump_m:
            logger.info("Wrist jumped; re-anchoring the arm to avoid a lunge.")
            self._anchor(wrist)

        displacement = wrist - self._operator_origin
        if np.linalg.norm(displacement) < config.deadband_m:
            displacement = np.zeros(3)
        self._last_operator = wrist

        target = self._flange_anchor + config.scale * (self._align @ displacement)
        return self._clamp_to_shell(target)

    def _begin_episode(self, wrist: np.ndarray) -> None:
        """Establish the mapping for a fresh episode.

        With a calibration this is the reproducible path: the operator's
        *calibrated* neutral maps to the arm's home pose, regardless of where
        their hand happens to be right now. Standing away from neutral therefore
        commands a correspondingly offset arm pose, which the rate limiter walks
        to rather than jumping.
        """
        if self._calibration is None:
            self._anchor(wrist)
            return

        self._operator_origin = self._calibration.neutral_wrist_flu_m.copy()
        self._flange_anchor = self._home_flange.copy()
        self._last_operator = wrist.copy()

        offset_m = float(np.linalg.norm(wrist - self._operator_origin))
        if offset_m > self._config.reanchor_jump_m:
            logger.warning(
                "Starting %.0f cm from your calibrated neutral, so the arm will travel "
                "%.0f cm to match before it tracks. Return to your neutral posture, or "
                "re-run the operator calibration if your seating changed.",
                offset_m * 100,
                offset_m * self._config.scale * 100,
            )

    def _anchor(self, wrist: np.ndarray) -> None:
        """Tie the operator's current wrist to the arm's current flange."""
        self._operator_origin = wrist.copy()
        self._flange_anchor = self._ik.flange_position(self._arm_q)
        self._last_operator = wrist.copy()
        self._arm_velocity[:] = 0

    def _clamp_to_shell(self, target: np.ndarray) -> np.ndarray:
        lo = self._config.min_radius_m
        hi = self._config.max_radius_m
        if lo is None and hi is None:
            return target
        radius = float(np.linalg.norm(target))
        lo_m = 0.0 if lo is None else float(lo)
        hi_m = float("inf") if hi is None else float(hi)
        if radius < 1e-9:
            return np.array([0.0, 0.0, lo_m if lo_m > 0 else 0.0])
        clamped = np.clip(radius, lo_m, hi_m)
        return target * (clamped / radius)
