"""Position-only inverse kinematics for the 3-DoF AM-ARM200 arm.

Three joints give three degrees of freedom, so the flange position can be
commanded but not its orientation: whatever the arm's structure produces is what
you get. The operator's wrist rotation is therefore not reproduced by the arm.

The solve is damped least squares on the model's own Jacobian rather than a
hand-derived analytic formula. The link origins, joint axes and the flange frame
all come from the CAD export with non-trivial offsets and a rotated J4 mesh, so
transcribing them into closed form is a needless place to introduce an error;
the compiled model is already the thing under test.

A scratch `MjData` is used so a solve never perturbs the simulation being
displayed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from orca_teleop.am200.canonical import ARM_ROLES

FLANGE_SITE = "orca_flange_center"


@dataclass(frozen=True)
class IkResult:
    """Outcome of one solve.

    `residual_m` is what the arm could not reach. The caller decides what an
    unreachable request means; this class never silently pretends it succeeded.
    """

    position: np.ndarray
    achieved: np.ndarray
    residual_m: float
    iterations: int
    clamped_by_rate: bool

    @property
    def converged(self) -> bool:
        return self.residual_m < 1e-3


class ArmIk:
    """Damped-least-squares position IK for the arm's flange site."""

    def __init__(
        self,
        model: mujoco.MjModel | str | Path,
        site_name: str = FLANGE_SITE,
        damping: float = 0.05,
        max_iterations: int = 60,
        tolerance_m: float = 5e-4,
        max_joint_step_rad: float = 0.25,  # used only when solve(..., rate_limit=True)
    ) -> None:
        if not isinstance(model, mujoco.MjModel):
            model = mujoco.MjModel.from_xml_path(str(model))
        self._model = model
        self._data = mujoco.MjData(model)

        self._site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self._site < 0:
            raise ValueError(f"model has no site {site_name!r}")

        names = [role.removeprefix("arm.") for role in ARM_ROLES]
        self._qpos_adr = np.empty(len(names), dtype=np.int64)
        self._dof_adr = np.empty(len(names), dtype=np.int64)
        lower, upper = [], []
        for index, name in enumerate(names):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint < 0:
                raise ValueError(f"model has no arm joint {name!r}")
            self._qpos_adr[index] = model.jnt_qposadr[joint]
            self._dof_adr[index] = model.jnt_dofadr[joint]
            lower.append(model.jnt_range[joint][0])
            upper.append(model.jnt_range[joint][1])
        self._lower = np.asarray(lower, dtype=np.float64)
        self._upper = np.asarray(upper, dtype=np.float64)

        self._damping = float(damping)
        self._max_iterations = int(max_iterations)
        self._tolerance_m = float(tolerance_m)
        self._max_joint_step_rad = float(max_joint_step_rad)

        self._jacp = np.zeros((3, model.nv), dtype=np.float64)

    @property
    def limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._lower.copy(), self._upper.copy()

    def flange_position(self, arm_q: np.ndarray) -> np.ndarray:
        """Forward kinematics: where the flange sits for these joint angles."""
        self._apply(arm_q)
        return self._data.site_xpos[self._site].copy()

    def solve(
        self,
        target_position: np.ndarray,
        seed: np.ndarray,
        rate_limit: bool = False,
    ) -> IkResult:
        """Drive the flange toward `target_position`, starting from `seed`.

        `seed` is normally the previous command, which keeps the arm on a
        continuous branch instead of jumping between elbow solutions.
        """
        target = np.asarray(target_position, dtype=np.float64)
        q = np.clip(np.asarray(seed, dtype=np.float64), self._lower, self._upper)

        iterations = 0
        while iterations < self._max_iterations:
            iterations += 1
            self._apply(q)
            error = target - self._data.site_xpos[self._site]
            if np.linalg.norm(error) < self._tolerance_m:
                break

            mujoco.mj_jacSite(self._model, self._data, self._jacp, None, self._site)
            jac = self._jacp[:, self._dof_adr]
            # (J J^T + lambda^2 I) is always invertible, so a singular or
            # out-of-reach configuration degrades into a small step instead of
            # blowing up.
            gram = jac @ jac.T + (self._damping**2) * np.eye(3)
            q = np.clip(q + jac.T @ np.linalg.solve(gram, error), self._lower, self._upper)

        self._apply(q)
        achieved = self._data.site_xpos[self._site].copy()
        residual = float(np.linalg.norm(target - achieved))

        clamped = False
        if rate_limit:
            delta = q - np.asarray(seed, dtype=np.float64)
            largest = float(np.abs(delta).max()) if delta.size else 0.0
            if largest > self._max_joint_step_rad:
                q = np.asarray(seed, dtype=np.float64) + delta * (
                    self._max_joint_step_rad / largest
                )
                q = np.clip(q, self._lower, self._upper)
                clamped = True
                self._apply(q)
                achieved = self._data.site_xpos[self._site].copy()

        return IkResult(
            position=q,
            achieved=achieved,
            residual_m=residual,
            iterations=iterations,
            clamped_by_rate=clamped,
        )

    def _apply(self, arm_q: np.ndarray) -> None:
        self._data.qpos[self._qpos_adr] = arm_q
        mujoco.mj_kinematics(self._model, self._data)
        mujoco.mj_comPos(self._model, self._data)
