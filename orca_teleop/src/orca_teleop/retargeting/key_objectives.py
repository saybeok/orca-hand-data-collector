"""Independent, hand-only adaptation of Xin et al.'s retargeting objectives.

Reference: Analyzing Key Objectives in Human-to-Robot Retargeting for Dexterous
Manipulation, DOI 10.1109/RAP.2026.3656110, equations 2--6. This is NOT the full
arm-hand method: global thumb/wrist terms are omitted because the existing arm
controller stays independent. ORCA-specific scaling, priors, and fifth-finger
support are explicit adaptations. Distances in this module are metres.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from orca_core import OrcaHand, OrcaJointPositions

from orca_teleop.retargeting import utils
from orca_teleop.retargeting.adaptive_analytical import (
    _PinocchioOrcaRobot,
    _default_urdf_path,
    _optional_imports,
)
from orca_teleop.retargeting.retargeter import FINGERS, TargetPose

logger = logging.getLogger(__name__)
TIPS = np.array([4, 8, 12, 16, 20])
DIPS = np.array([3, 7, 11, 15, 19])


def pinch_targets(vectors: np.ndarray, inner: float, outer: float, slope: float):
    """Paper eqs. 2--4: complementary weights and continuous distance rescaling."""
    distances = np.linalg.norm(vectors, axis=1)
    pinch_weight = 1.0 / (1.0 + np.exp(np.clip(slope * (distances - outer), -60, 60)))
    lengths = np.where(distances < inner, 0.0,
                       np.where(distances <= outer, outer * (distances - inner) / (outer - inner), distances))
    targets = vectors * (lengths / np.maximum(distances, 1e-12))[:, None]
    return targets, pinch_weight, 1.0 - pinch_weight


class KeyObjectivesRetargeter:
    """Same ingress/output contract, separate optimizer from adaptive_analytical."""

    def __init__(self, *, hand_config_path, urdf_path, config_path):
        self._nlopt, pin, yaml = _optional_imports()
        self._hand = OrcaHand(hand_config_path)  # Config only; never connects.
        self.hand_type = self._hand.config.type
        self.model_version = "v1"
        if self.hand_type not in ("left", "right"):
            raise ValueError("key_objectives expects a left or right ORCA hand")
        self.urdf_path = str(urdf_path or _default_urdf_path(self.hand_type, "v1"))
        if config_path is None:
            raise ValueError("key_objectives requires an explicit config_path")
        self.config = yaml.safe_load(Path(config_path).read_text())
        if self.config.get("optimizer", {}).get("type") != "key_objectives":
            raise ValueError("Expected optimizer.type: key_objectives; do not reuse adaptive pinch YAML")
        self._retarget_cfg = self.config["retarget"]
        self._build_retarget_params()
        velocity = float(self.config.get("execution", {}).get("max_joint_velocity_deg_s", 60))
        if not np.isfinite(velocity) or velocity < 0:
            raise ValueError("execution.max_joint_velocity_deg_s must be finite and non-negative")
        self._max_velocity = np.deg2rad(velocity)
        self._robot = _PinocchioOrcaRobot(self.urdf_path, pin)
        self._finger_joint_ids = [j for j in self._hand.config.joint_ids if j != "wrist"]
        if len(self._finger_joint_ids) != 16:
            raise ValueError("This profile requires the 16-finger-joint ORCA v1 layout")
        names = self._robot.dof_joint_names
        try:
            self._model_q_indices_np = np.array([names.index(f"{self.hand_type}_{j}") for j in self._finger_joint_ids])
        except ValueError as exc:
            raise ValueError("ORCA v1 URDF and hand joint names must match") from exc
        rom = self._hand.config.joint_roms_dict
        self._lower_phys = np.deg2rad([rom[j][0] for j in self._finger_joint_ids])
        self._upper_phys = np.deg2rad([rom[j][1] for j in self._finger_joint_ids])
        self._wrist_lower_deg, self._wrist_upper_deg = rom["wrist"]
        self._neutral_qpos_phys = np.clip(np.deg2rad([
            self._hand.config.neutral_position[j] for j in self._finger_joint_ids
        ]), self._lower_phys, self._upper_phys)
        side = self.hand_type
        self._computed_frame_names = [f"{side}_palm"] + [f"{side}_{f}_mp" for f in FINGERS]
        self._computed_frame_names += [f"{side}_{f}_fingertip" for f in FINGERS]
        self._computed_frame_names += [f"{side}_thumb_dp"] + [f"{side}_{f}_ip" for f in FINGERS[1:]]
        self._computed_frame_indices = [self._robot.get_frame_index(n) for n in self._computed_frame_names]
        self._tip_indices = np.arange(6, 11)
        self._dip_indices = np.arange(11, 16)
        # Exactly the same points for shape, orientation, pinch, scale, and eval.
        # Frame origins are a kinematic proxy; no claim of measured pad contact.
        self._contact_offsets = [np.zeros(3) for _ in self._computed_frame_names]
        self._frame_offsets = self._contact_offsets
        self._rotation_matrix = None
        pos, _ = self._positions(self._neutral_qpos_phys)
        self._scale_lengths = np.linalg.norm(pos[self._tip_indices] - pos[0], axis=1)
        zero, _ = self._positions(np.zeros(len(self._finger_joint_ids)))
        self._urdfhand_center, self._urdfhand_rot_matrix = utils.get_hand_center_and_rotation(
            thumb_base=zero[1], index_base=zero[2], middle_base=zero[3],
            ring_base=zero[4], pinky_base=zero[5], wrist=zero[0],
        )
        unknown = set(self._prior_cfg) - set(self._finger_joint_ids)
        if unknown:
            raise ValueError(f"Unknown joint_prior_weights: {sorted(unknown)}")
        self._prior_weights = np.array([float(self._prior_cfg.get(j, 0)) for j in self._finger_joint_ids])
        if not np.isfinite(self._prior_weights).all() or np.any(self._prior_weights < 0):
            raise ValueError("Joint prior weights must be finite and non-negative")
        self.reset()

    @classmethod
    def from_paths(cls, model_path=None, urdf_path=None, config_path=None):
        return cls(hand_config_path=model_path, urdf_path=urdf_path, config_path=config_path)

    def _build_retarget_params(self):
        c = self._retarget_cfg
        self._inner = float(c.get("pinch_inner_m", 0.01))
        self._outer = float(c.get("pinch_outer_m", 0.10))
        self._slope = float(c.get("switch_slope_per_m", 10.0))
        if not (np.isfinite([self._inner, self._outer, self._slope]).all()
                and 0 <= self._inner < self._outer and self._slope > 0):
            raise ValueError("Require finite 0 <= pinch_inner_m < pinch_outer_m and positive slope")
        self._weights = {k: float(c.get(k, default)) for k, default in
                         (("w_shape", 1), ("w_direction", 10), ("w_pinch", 10), ("w_temporal", 0.01))}
        if any(not np.isfinite(v) or v < 0 for v in self._weights.values()):
            raise ValueError("Objective weights must be finite and non-negative")
        if sum(self._weights[k] for k in ("w_shape", "w_direction", "w_pinch")) <= 0:
            raise ValueError("At least one task-space objective must be active")
        self._prior_cfg = c.get("joint_prior_weights", {})
        self._calibration_frames = int(c.get("calibration_frames", 30))
        if self._calibration_frames < 1:
            raise ValueError("calibration_frames must be positive")
        self._maxeval = int(c.get("maxeval", 100))
        self._ftol = float(c.get("ftol_abs", 1e-9))
        if self._maxeval < 1 or not np.isfinite(self._ftol) or self._ftol <= 0:
            raise ValueError("maxeval and ftol_abs must be positive")

    def reset(self):
        self._last_qpos_phys = None
        self._calibration_mags = []
        self._calibration_done = False
        self._mano_scale = 1.0
        self.last_status = "calibrating"
        self.last_terms = {}
        self._last_timestamp_ns = None

    def _full_model_qpos(self, q):
        full = np.zeros(self._robot.nq)
        # Preserve the tested local physical/MuJoCo convention, not theta-ref.
        full[self._model_q_indices_np] = q
        return full

    def _positions(self, q):
        pos, jac = self._robot.compute_positions_and_jacobians(
            self._full_model_qpos(q), self._computed_frame_indices, self._contact_offsets,
        )
        return pos, jac[:, :, self._model_q_indices_np]

    def _target_keypoints_in_urdf_frame(self, normalized):
        return normalized * self._mano_scale @ self._urdfhand_rot_matrix.T + self._urdfhand_center

    def _prepare_targets(self, keypoints):
        vectors = keypoints[TIPS] - keypoints[0]
        relative = keypoints[TIPS[1:]] - keypoints[TIPS[0]]
        target_pinch, switch, shape_other = pinch_targets(relative, self._inner, self._outer, self._slope)
        # ORCA extension: thumb takes the most active pair's complementary weight.
        shape = np.r_[1.0 - switch.max(), shape_other]
        directions = keypoints[TIPS] - keypoints[DIPS]
        return vectors, directions, target_pinch, switch, shape

    def _loss_and_grad(self, q, targets, previous=None):
        pos, jac = self._positions(q)
        tips, jt = pos[self._tip_indices], jac[self._tip_indices]
        dips, jd = pos[self._dip_indices], jac[self._dip_indices]
        target_shape, target_dir, target_pinch, switch, shape_weight = targets
        errors = {
            "shape": tips - pos[0] - target_shape,
            "direction": tips - dips - target_dir,
            "pinch": tips[1:] - tips[0] - target_pinch,
        }
        derivatives = {"shape": jt - jac[0], "direction": jt - jd, "pinch": jt[1:] - jt[0]}
        coefficients = {"shape": self._weights["w_shape"] * shape_weight,
                        "direction": np.full(5, self._weights["w_direction"]),
                        "pinch": self._weights["w_pinch"] * switch}
        gradient = np.zeros_like(q)
        terms = {}
        for name, error in errors.items():
            coeff = coefficients[name]
            terms[name] = float(np.sum(coeff * np.sum(error**2, axis=1)))
            gradient += 2 * np.einsum("i,ij,ijk->k", coeff, error, derivatives[name])
        delta = q - self._neutral_qpos_phys
        terms["joint_prior"] = float(np.sum(self._prior_weights * delta**2))
        gradient += 2 * self._prior_weights * delta
        terms["temporal"] = 0.0
        if previous is not None:
            delta = q - previous
            terms["temporal"] = self._weights["w_temporal"] * float(delta @ delta)
            gradient += 2 * self._weights["w_temporal"] * delta
        self.last_terms = terms
        return sum(terms.values()), gradient

    def _solve(self, keypoints, bounds=None):
        previous = self._last_qpos_phys
        initial = self._neutral_qpos_phys.copy() if previous is None else previous.copy()
        targets = self._prepare_targets(keypoints)
        opt = self._nlopt.opt(self._nlopt.LD_SLSQP, len(initial))
        lower, upper = (self._lower_phys, self._upper_phys) if bounds is None else bounds
        if np.max(upper - lower) < 1e-14:
            self._last_qpos_phys = initial.copy()
            self.last_status = "held_duplicate_timestamp"
            return initial
        opt.set_lower_bounds(lower)
        opt.set_upper_bounds(upper)
        opt.set_maxeval(self._maxeval)
        opt.set_ftol_abs(self._ftol)

        def objective(q, gradient):
            loss, grad = self._loss_and_grad(np.asarray(q), targets, previous)
            if gradient.size:
                gradient[:] = grad
            return loss

        opt.set_min_objective(objective)
        try:
            q = np.asarray(opt.optimize(initial))
            self.last_status = str(opt.last_optimize_result())
        except RuntimeError as exc:
            logger.warning("Key-objectives solve failed; holding last command: %s", exc)
            q = initial
            self.last_status = "held_after_solver_failure"
        if not np.isfinite(q).all():
            q = initial
            self.last_status = "held_after_nonfinite_result"
        q = np.clip(q, lower, upper)
        self._loss_and_grad(q, targets, previous)
        self._last_qpos_phys = q.copy()
        return q

    def _execution_bounds(self, timestamp_ns):
        """Local execution envelope, separate from paper objective equations.

        It bounds the optimizer's feasible command step. A soft temporal cost
        alone cannot guarantee a velocity limit when changing IK branches.
        """
        if self._max_velocity == 0:
            return None
        now = int(timestamp_ns) if timestamp_ns is not None and timestamp_ns > 0 else time.monotonic_ns()
        dt = 1 / 30 if self._last_timestamp_ns is None else np.clip((now - self._last_timestamp_ns) * 1e-9, 0, 0.1)
        if self._last_timestamp_ns is None or now > self._last_timestamp_ns:
            self._last_timestamp_ns = now
        previous = self._neutral_qpos_phys if self._last_qpos_phys is None else self._last_qpos_phys
        step = self._max_velocity * dt
        return np.maximum(self._lower_phys, previous - step), np.minimum(self._upper_phys, previous + step)

    def retarget(self, target_pose: TargetPose):
        if target_pose.source != "mediapipe" or target_pose.joint_positions.shape != (21, 3):
            raise ValueError("key_objectives requires canonical 21-point MediaPipe-layout landmarks")
        if not np.isfinite(target_pose.joint_positions).all() or not np.isfinite(target_pose.wrist_angle_degrees):
            raise ValueError("Hand frame must be finite")
        normalized = utils.get_normalized_local_manohand_joint_pos(target_pose.joint_positions, "mediapipe")
        if not self._calibration_done:
            self._calibration_mags.append(np.linalg.norm(normalized[TIPS] - normalized[0], axis=1))
            if len(self._calibration_mags) >= self._calibration_frames:
                lengths = np.median(self._calibration_mags, axis=0)
                if np.any(lengths < 1e-6):
                    self.reset()
                    raise ValueError("Degenerate hand scale calibration")
                self._mano_scale = float(np.median(self._scale_lengths / lengths))
                self._calibration_done = True
                logger.info("Key-objectives scale calibrated: %.4f", self._mano_scale)
            return None
        q = self._solve(self._target_keypoints_in_urdf_frame(normalized),
                        self._execution_bounds(target_pose.timestamp_ns))
        wrist = float(np.clip(target_pose.wrist_angle_degrees, self._wrist_lower_deg, self._wrist_upper_deg))
        return OrcaJointPositions(dict(zip(self._finger_joint_ids, np.rad2deg(q), strict=True)) |
                                  {"wrist": wrist if self.hand_type == "left" else -wrist})
