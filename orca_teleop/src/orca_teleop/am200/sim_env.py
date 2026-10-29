"""Gymnasium env for the combined AM-ARM200 + ORCA right hand model.

Reuses ``orca_sim``'s ``BaseOrcaHandEnv`` for reset/step/render rather than
reimplementing them; the only orca_sim-specific behaviour it has to escape is
scene lookup, which normally resolves a filename inside the installed package.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from orca_sim.envs import BaseOrcaHandEnv

from orca_teleop.am200.dynamics import (
    CommandDelay,
    DynamicsRandomizer,
    SimDynamics,
    dynamics_path_for,
)

MODEL_PATH_ENV = "AM200_MODEL_PATH"
DEFAULT_MODEL_PATH = (
    Path(os.environ.get("ORCA_HAND_ROOT", Path.home() / "orca_hand"))
    / "orca_am200"
    / "robot_model"
    / "am_arm200_orca_right.xml"
)

# The arm and hand actuators are position servos, so a control step should
# advance about as much sim time as the sink's 30 Hz loop consumes in wall time.
# Otherwise the sim falls behind the operator and teleop feels sluggish.
_TARGET_CONTROL_HZ = 30.0


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    path = Path(model_path or os.environ.get(MODEL_PATH_ENV) or DEFAULT_MODEL_PATH)
    if not path.is_file():
        raise FileNotFoundError(
            f"Combined model not found at {path}. Pass --model-path or set ${MODEL_PATH_ENV}."
        )
    return path.resolve()


class AmArm200OrcaRightEnv(BaseOrcaHandEnv):
    """The 20-DoF combined embodiment: 3-DoF arm plus the 17-DoF ORCA v1 hand.

    ``dynamics`` (default: ``sim_dynamics.yaml`` beside the model) adds the arm
    command delay and optional per-episode domain randomization, resampled on
    every :meth:`reset`. ``randomize`` overrides the file's ``enabled`` flag.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        frame_skip: int | None = None,
        render_mode: str | None = None,
        dynamics: SimDynamics | str | Path | None = None,
        randomize: bool | None = None,
    ) -> None:
        path = resolve_model_path(model_path)

        # BaseOrcaHandEnv resolves `scene_file` under orca_sim's packaged scenes
        # via `SCENES_ROOT / version / scene_file`. An absolute scene_file makes
        # that join return the path unchanged, which is how this out-of-tree
        # model gets loaded without duplicating the base env. `version` still has
        # to name a real orca_sim version for the lookup to run at all.
        super().__init__(
            scene_file=str(path),
            version="v1",
            frame_skip=frame_skip or 1,
            render_mode=render_mode,
        )
        if self.scene_path != path:
            raise RuntimeError(
                f"orca_sim resolved the scene to {self.scene_path} instead of {path}; "
                "its scene lookup changed and this env needs updating."
            )

        self.version = "am_arm200_orca_right_v1"
        if frame_skip is None:
            self.frame_skip = max(1, round(1.0 / (_TARGET_CONTROL_HZ * self.model.opt.timestep)))

        if not isinstance(dynamics, SimDynamics):
            dynamics = SimDynamics.load(dynamics or dynamics_path_for(path))
        if randomize is not None:
            dynamics.randomization.enabled = randomize
        self.dynamics = dynamics
        self._randomizer = DynamicsRandomizer(self.model, dynamics)
        self._delay = CommandDelay(self._randomizer.nominal_delays)

    @property
    def dynamics_sample(self) -> dict[str, Any]:
        """Randomized factors of the current episode (empty when disabled)."""
        return self._randomizer.last_sample

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._delay = CommandDelay(self._randomizer.sample(self.np_random))
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self._delay.active:
            return super().step(action)

        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, got {action.shape}"
            )
        self._delay.push(self.data.time, np.clip(action, self.action_low, self.action_high))
        for _ in range(self.frame_skip):
            self.data.ctrl[:] = self._delay.at(self.data.time)
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        info = self._get_info()
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info
