"""ORCA adapter for dex-retargeting's unmodified five-finger DexPilot solver.

Local responsibilities are coordinate alignment, physical joint bounds, and
the pipeline's degree-valued output contract. No adaptive pinch terms, learned
offsets, or finger-specific boosts are applied here.
"""

from pathlib import Path

import numpy as np
from orca_core import OrcaHand, OrcaJointPositions

from orca_teleop.retargeting import utils
from orca_teleop.retargeting.retargeter import FINGERS, TargetPose


class DexPilotRetargeter:
    def __init__(self, *, hand_config_path, urdf_path, config_path):
        try:
            import yaml
            from dex_retargeting.retargeting_config import RetargetingConfig
        except ImportError as exc:
            raise RuntimeError(
                "DexPilot requires the dexpilot extra: uv sync --extra dexpilot"
            ) from exc
        if config_path is None or urdf_path is None:
            raise ValueError("dexpilot requires explicit config_path and urdf_path")
        self.config = yaml.safe_load(Path(config_path).read_text())
        if self.config.get("optimizer", {}).get("type") != "dexpilot":
            raise ValueError("Expected optimizer.type: dexpilot; use the DexPilot profile")
        velocity = float(self.config.get("execution", {}).get("max_joint_velocity_deg_s", 60))
        if not np.isfinite(velocity) or velocity < 0:
            raise ValueError("execution.max_joint_velocity_deg_s must be finite and non-negative")
        self._max_velocity = np.deg2rad(velocity)
        self._hand = OrcaHand(hand_config_path)  # Configuration only, no serial connection.
        self.hand_type = self._hand.config.type
        if self.hand_type != "right":
            raise ValueError("This ORCA v1 DexPilot adapter currently supports the right hand only")
        self._finger_joint_ids = [j for j in self._hand.config.joint_ids if j != "wrist"]
        if len(self._finger_joint_ids) != 16:
            raise ValueError("Expected the ORCA v1 16-finger-joint layout")
        self.urdf_path = str(Path(urdf_path).resolve())
        settings = dict(self.config["retargeting"])
        settings.update(
            type="dexpilot", urdf_path=self.urdf_path,
            target_joint_names=[f"right_{j}" for j in self._finger_joint_ids],
            wrist_link_name="right_palm",
            finger_tip_link_names=[f"right_{f}_fingertip" for f in FINGERS],
        )
        self._seq = RetargetingConfig.from_dict(settings).build()
        self._optimizer = self._seq.optimizer
        self._robot = self._optimizer.robot
        rom = self._hand.config.joint_roms_dict
        self._lower_phys = np.deg2rad([rom[j][0] for j in self._finger_joint_ids])
        self._upper_phys = np.deg2rad([rom[j][1] for j in self._finger_joint_ids])
        self._wrist_lower_deg, self._wrist_upper_deg = rom["wrist"]
        self._neutral = np.clip(np.deg2rad([
            self._hand.config.neutral_position[j] for j in self._finger_joint_ids
        ]), self._lower_phys, self._upper_phys)
        self._model_q_indices_np = self._optimizer.idx_pin2target
        # Local convention, shared with the working adaptive backend:
        # URDF hinge angle == physical radians; canonical MuJoCo adds its ref.
        limits = np.column_stack((self._lower_phys, self._upper_phys))
        self._seq.joint_limits = limits
        self._optimizer.opt.set_lower_bounds(self._lower_phys)
        self._optimizer.opt.set_upper_bounds(self._upper_phys)
        self._fixed = np.zeros(len(self._optimizer.fixed_joint_names), dtype=np.float32)
        self._robot.compute_forward_kinematics(np.zeros(self._robot.dof))
        names = ["right_palm"] + [f"right_{f}_mp" for f in FINGERS]
        pos = [self._robot.get_link_pose(self._robot.get_link_index(n))[:3, 3].copy()
               for n in names]
        _, self._urdf_rotation = utils.get_hand_center_and_rotation(
            thumb_base=pos[1], index_base=pos[2], middle_base=pos[3],
            ring_base=pos[4], pinky_base=pos[5], wrist=pos[0],
        )
        self.reset()

    @classmethod
    def from_paths(cls, model_path=None, urdf_path=None, config_path=None):
        return cls(hand_config_path=model_path, urdf_path=urdf_path, config_path=config_path)

    def reset(self):
        # Upstream SeqRetargeting.reset alone does not clear projection/filter state.
        self._seq.reset()
        self._optimizer.projected.fill(False)
        q = np.zeros(self._robot.dof)
        q[self._model_q_indices_np] = self._neutral
        self._seq.set_qpos(q)
        if self._seq.filter is not None:
            self._seq.filter.reset()
            self._seq.filter.next(q)
        self._last_qpos_phys = None
        self._last_timestamp_ns = None

    def _frame_dt(self, timestamp_ns):
        if timestamp_ns is None:
            return 1 / 30
        previous = self._last_timestamp_ns
        if previous is not None and timestamp_ns <= previous:
            return 0.0
        self._last_timestamp_ns = timestamp_ns
        return 1 / 30 if previous is None else min((timestamp_ns - previous) * 1e-9, .1)

    def _target_vectors(self, pose: TargetPose):
        points = pose.joint_positions
        if pose.source != "mediapipe" or points.shape != (21, 3):
            raise ValueError("DexPilot expects canonical 21x3 hand landmarks (source='mediapipe')")
        if not np.isfinite(points).all():
            raise ValueError("DexPilot landmarks must be finite")
        local = utils.get_normalized_local_manohand_joint_pos(points, pose.source)
        aligned = local @ self._urdf_rotation.T
        if not np.isfinite(aligned).all():
            raise ValueError("Degenerate hand coordinate frame")
        origin, task = self._optimizer.target_link_human_indices
        return aligned[task] - aligned[origin]

    def retarget(self, target_pose: TargetPose):
        vectors = self._target_vectors(target_pose)
        if not np.isfinite(target_pose.wrist_angle_degrees):
            raise ValueError("Wrist angle must be finite")
        q = self._seq.retarget(vectors, self._fixed)
        finger = np.clip(q[self._model_q_indices_np], self._lower_phys, self._upper_phys)
        if not np.isfinite(finger).all():
            raise RuntimeError("DexPilot returned non-finite joint positions")
        dt = self._frame_dt(target_pose.timestamp_ns)
        if self._max_velocity > 0:
            previous = self._neutral if self._last_qpos_phys is None else self._last_qpos_phys
            step = self._max_velocity * dt
            finger = np.clip(finger, previous - step, previous + step)
            # Execution adaptation: subsequent solves start from the emitted pose.
            self._seq.last_qpos = finger.copy()
            if self._seq.filter is not None:
                self._seq.filter.y[self._model_q_indices_np] = finger
        self._last_qpos_phys = finger.copy()
        wrist = float(np.clip(target_pose.wrist_angle_degrees,
                              self._wrist_lower_deg, self._wrist_upper_deg))
        return OrcaJointPositions({
            **dict(zip(self._finger_joint_ids, np.rad2deg(finger), strict=True)),
            "wrist": -wrist,
        })
