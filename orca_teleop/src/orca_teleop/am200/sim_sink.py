"""Sim-backed sink for the combined 20-DoF AM-ARM200 + ORCA right hand robot.

The upstream ``OrcaHandSimSink`` cannot drive this robot: it only knows how to
build the hand-only ``orca_sim`` envs, and its actuator mapping requires the
model's actuators to be a 1:1 cover of the orca_core hand config, so the three
arm actuators make it raise. This sink speaks the canonical 20-DoF vector
instead, and accepts hand-only ``OrcaJointPositions`` as well so the existing
ingress and retargeter stack drives the hand while the arm holds its target.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from orca_core import OrcaHand, OrcaJointPositions

from orca_teleop.am200.canonical import CanonicalJointMap, RobotJointCommand
from orca_teleop.am200.sim_env import AmArm200OrcaRightEnv, resolve_model_path
from orca_teleop.am200.tasks import AmArm200TaskEnv, make_task, task_scene_path
from orca_teleop.cameras import CameraManager, OpenCVCameraConfig
from orca_teleop.pipeline import _SHUTDOWN, RecordableSink, SinkObservation
from orca_teleop.utils import RateTicker

logger = logging.getLogger(__name__)

DEFAULT_RENDER_FPS = 30
# How long the sink coasts on the last target before it says so. Teleop input
# stopping is normal (the operator lowers their hand); silence is not.
STALE_COMMAND_WARN_S = 2.0
TASK_RECORD_CAMERA = "task_front"


def configure_arm_servos(model: Any, data: Any, mapping: CanonicalJointMap, ratio: float) -> np.ndarray:
    """Damp the simulated position servos using local effective inertia.

    This stabilizes teleop; it is not an identified Feetech motor model.
    Only the three arm actuators are changed, not the hand or IK targets.
    """
    import mujoco

    actuators = mapping.actuator_index[mapping.arm_slice]
    dofs = model.jnt_dofadr[model.actuator_trnid[actuators, 0]]
    inertia = []
    for dof in dofs:
        unit = np.zeros(model.nv)
        unit[dof] = 1.0
        column = np.zeros(model.nv)
        mujoco.mj_mulM(model, data, column, unit)
        inertia.append(column[dof])
    kv = 2.0 * ratio * np.sqrt(model.actuator_gainprm[actuators, 0] * np.asarray(inertia))
    model.actuator_biasprm[actuators, 2] = -kv
    return kv


def parse_sim_camera_spec(spec: str) -> tuple[str, str]:
    """``KEY:MJCF_CAMERA`` (or just ``MJCF_CAMERA``, used as the key too)."""
    key, sep, cam = spec.partition(":")
    if not sep:
        key, cam = spec, spec
    if not key or not cam:
        raise ValueError(f"Bad sim camera spec {spec!r}; expected KEY:MJCF_CAMERA")
    return key, cam


@dataclass(frozen=True)
class SimCameraConfig:
    name: str = "frontal"
    width: int = 320
    height: int = 240


class Am200SimSink(RecordableSink):
    """Steps the combined model from the actions queue.

    Actions may be a full-robot :class:`RobotJointCommand` in the canonical
    order, or an :class:`OrcaJointPositions` for the hand alone. Either way the
    sink holds one 20-DoF target vector in the model's radian convention and
    writes it to every actuator each tick, so a dropped or partial command
    leaves the robot where it was rather than relaxing.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        hand_config_path: str | Path | None = None,
        render_mode: str = "human",
        camera_config: SimCameraConfig | None = None,
        camera_configs: list[OpenCVCameraConfig] | None = None,
        task: str | None = None,
        record_camera: str | None = None,
        respawn_after_s: float | None = None,
        dynamics: str | Path | None = None,
        randomize: bool | None = None,
        extra_cameras: dict[str, str] | None = None,
        arm_max_step_rad: float | None = None,
        arm_damping_ratio: float | None = None,
    ) -> None:
        """``task`` names a scene task (see :mod:`orca_teleop.am200.tasks`); its scene
        becomes the default model. ``record_camera`` names a fixed MJCF camera for
        the recorded image (task scenes default to ``task_front``).
        ``extra_cameras`` maps further observation keys to fixed MJCF cameras (e.g.
        ``{"wrist": "wrist_cam"}``), rendered at the same size. With
        ``respawn_after_s``, the object alone is respawned that long after a
        success or a drop, so live teleop can keep attempting without a reset.
        ``dynamics``/``randomize`` go to the env (see
        :mod:`orca_teleop.am200.dynamics`); randomization resamples per reset.
        """
        for name, value in (("arm_max_step_rad", arm_max_step_rad), ("arm_damping_ratio", arm_damping_ratio)):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        self._arm_max_step_rad = arm_max_step_rad
        self._arm_damping_ratio = arm_damping_ratio
        self._dynamics = dynamics
        self._randomize = randomize
        self._task = make_task(task) if task else None
        if model_path is None and self._task is not None:
            model_path = task_scene_path(self._task)
        self._model_path = resolve_model_path(model_path)
        self._joint_map_path = self._model_path.parent / "joint_map.json"
        self._hand_config_path = hand_config_path
        self._render_mode = render_mode
        self._camera_config = camera_config or SimCameraConfig()
        self._cameras = CameraManager(camera_configs or [])
        self._record_camera_name = record_camera or (
            TASK_RECORD_CAMERA if self._task is not None else None
        )
        self._extra_camera_names = dict(extra_cameras or {})
        self._extra_cameras: dict[str, Any] = {}
        self._respawn_after_s = respawn_after_s
        self._episode_done_at: float | None = None
        self._success_logged = False

        self._env: AmArm200OrcaRightEnv | None = None
        self._map: CanonicalJointMap | None = None
        self._hand_config: Any | None = None
        self._target: np.ndarray | None = None
        self._dt = 1.0 / DEFAULT_RENDER_FPS
        self._renderer: Any | None = None
        self._record_camera: Any | None = None
        self._last_command_ns: int | None = None
        self._stale_logged = False

    def connect(self) -> None:
        import mujoco

        if self._task is not None:
            env: AmArm200OrcaRightEnv = AmArm200TaskEnv(
                self._task,
                model_path=self._model_path,
                render_mode=self._render_mode,
                dynamics=self._dynamics,
                randomize=self._randomize,
            )
        else:
            env = AmArm200OrcaRightEnv(
                model_path=self._model_path,
                render_mode=self._render_mode,
                dynamics=self._dynamics,
                randomize=self._randomize,
            )
        env.reset()

        self._env = env
        self._map = CanonicalJointMap.load(self._joint_map_path, env.model)
        self._hand_config = self._load_hand_config()
        self._target = self._home_vector()

        self._dt = 1.0 / float(env.metadata.get("render_fps", DEFAULT_RENDER_FPS))

        self._renderer = mujoco.Renderer(
            env.model,
            height=self._camera_config.height,
            width=self._camera_config.width,
        )
        if self._record_camera_name is not None:
            self._record_camera = self._fixed_camera(env.model, self._record_camera_name)
        else:
            self._record_camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(env.model, self._record_camera)
        self._extra_cameras = {
            key: self._fixed_camera(env.model, cam)
            for key, cam in self._extra_camera_names.items()
        }

        sim_keys = [self._camera_config.name, *self._extra_cameras]
        if len(set(sim_keys)) != len(sim_keys):
            raise ValueError(f"Duplicate sim camera keys: {sim_keys}")
        for key in sim_keys:
            if key in self._cameras.names:
                raise ValueError(
                    f"Real camera name {key!r} collides with a sim render camera name; "
                    "rename one so both observations get distinct keys."
                )
        self._cameras.open()
        self._cameras.ensure_live()

        self.go_home()

        logger.info(
            "Am200SimSink connected: model=%s dof=%d (arm %d + hand %d) "
            "frame_skip=%d dt=%.3fs cameras=%s",
            self._model_path.name,
            self._map.dof,
            self._map.arm_dof,
            self._map.hand_dof,
            env.frame_skip,
            self._dt,
            self.camera_shapes,
        )

    @property
    def model_path(self) -> Path:
        """The combined MJCF this sink drives, so a composer can share it."""
        return self._model_path

    @property
    def joint_ids(self) -> list[str]:
        assert self._map is not None, "connect() must be called first"
        return list(self._map.roles)

    @property
    def retarget_model_path(self) -> str | None:
        """Lets the pipeline resolve the retargeter's hand config from the sink."""
        return None if self._hand_config_path is None else str(self._hand_config_path)

    @property
    def camera_shapes(self) -> dict[str, tuple[int, int, int]]:
        shape = (self._camera_config.height, self._camera_config.width, 3)
        return {
            self._camera_config.name: shape,
            **{key: shape for key in self._extra_camera_names},
            **self._cameras.shapes,
        }

    def get_observation(self) -> SinkObservation:
        return SinkObservation(
            joint_state=self._read_joint_state(),
            images={
                self._camera_config.name: self._render(),
                **{key: self._render(cam) for key, cam in self._extra_cameras.items()},
                **self._cameras.capture(),
            },
        )

    @staticmethod
    def _fixed_camera(model: Any, name: str) -> Any:
        import mujoco

        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cam_id < 0:
            raise ValueError(f"model has no camera {name!r}")
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = cam_id
        return cam

    def _read_joint_state(self) -> np.ndarray:
        """Measured joint positions in the canonical order, in model radians."""
        assert self._env is not None and self._map is not None
        return self._map.from_qpos(self._env.data.qpos).astype(np.float32)

    def _render(self, camera: Any | None = None) -> np.ndarray:
        assert self._env is not None and self._renderer is not None
        self._renderer.update_scene(
            self._env.data, camera=self._record_camera if camera is None else camera
        )
        return np.asarray(self._renderer.render(), dtype=np.uint8)

    def set_target(self, action: RobotJointCommand | OrcaJointPositions) -> None:
        """Update the held target without stepping the sim."""
        assert self._map is not None and self._target is not None

        if isinstance(action, RobotJointCommand):
            if tuple(action.joint_ids) != self._map.roles:
                raise ValueError(
                    "RobotJointCommand joint_ids do not match the canonical order; "
                    f"got {len(action.joint_ids)} ids starting {action.joint_ids[:3]}"
                )
            self._target = self._map.clip(action.position)
        elif isinstance(action, OrcaJointPositions):
            target = self._target.copy()
            target[self._map.hand_slice] = self._map.hand_from_physical(action)
            self._target = self._map.clip(target)
        else:
            raise TypeError(f"unsupported action type {type(action).__name__}")

        if self._arm_max_step_rad is not None:
            # Same connect/home-relative position envelope as AmArmFeetech.
            # This is a range limit, not a per-frame or per-second speed cap.
            self._target[self._map.arm_slice] = np.clip(
                self._target[self._map.arm_slice], -self._arm_max_step_rad, self._arm_max_step_rad
            )

        self._last_command_ns = time.monotonic_ns()
        self._stale_logged = False

    def dispatch_action(self, action: RobotJointCommand | OrcaJointPositions) -> None:
        self.set_target(action)
        self._step()

    @property
    def task_info(self) -> dict[str, Any] | None:
        """Latest task state (lift, contacts, success, object pose), if a task is loaded."""
        if not isinstance(self._env, AmArm200TaskEnv):
            return None
        return self._env._get_info()

    def _step(self) -> None:
        assert self._env is not None and self._map is not None and self._target is not None
        _, _, terminated, _, info = self._env.step(self._map.to_ctrl(self._target).astype(np.float32))
        if self._task is not None:
            self._track_task(terminated, info)

    def _track_task(self, terminated: bool, info: dict[str, Any]) -> None:
        if info.get("success") and not self._success_logged:
            logger.info("TASK SUCCESS: %s held %.1f cm up", self._task.name, 100 * info["lift_m"])
            self._success_logged = True
        if terminated and self._episode_done_at is None:
            if info.get("dropped"):
                logger.info("object dropped off the desk")
            self._episode_done_at = time.monotonic()
        if (
            self._respawn_after_s is not None
            and self._episode_done_at is not None
            and time.monotonic() - self._episode_done_at >= self._respawn_after_s
        ):
            self.respawn_object()

    def respawn_object(self) -> None:
        """Re-place the task object without touching the robot."""
        assert isinstance(self._env, AmArm200TaskEnv)
        pos = self._env.respawn_object()
        self._episode_done_at = None
        self._success_logged = False
        logger.info("object respawned at %s", np.round(pos, 3).tolist())

    def reset_scene(self, seed: int | None = None) -> None:
        """Full episode reset: robot to home at rest, object respawned."""
        assert isinstance(self._env, AmArm200TaskEnv) and self._map is not None
        import mujoco

        env = self._env
        env.reset(seed=seed)
        home = self._home_vector()
        env.data.qpos[self._map.qpos_index] = home
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        self._configure_arm_servos()
        self._target = home
        self._episode_done_at = None
        self._success_logged = False
        logger.info(
            "scene reset; object at %s", np.round(env.task.object_pose(env.data)[:3], 3).tolist()
        )
        if env.dynamics_sample:
            logger.info("dynamics sample: %s", env.dynamics_sample)

    def _home_vector(self) -> np.ndarray:
        """Arm at zero, hand at the orca_core neutral pose, in model radians."""
        assert self._map is not None and self._hand_config is not None
        home = np.zeros(self._map.dof, dtype=np.float64)
        home[self._map.hand_slice] = self._map.hand_from_physical(self.home_position())
        return self._map.clip(home)

    def home_position(self) -> OrcaJointPositions:
        assert self._hand_config is not None, "connect() must be called first"
        positions = dict(self._hand_config.neutral_position)
        positions["wrist"] = 0.0
        return OrcaJointPositions(positions)

    def go_home(self) -> None:
        assert self._map is not None
        if self._task is not None:
            self.reset_scene()
        else:
            self._target = self._home_vector()
            self._configure_arm_servos()
        self._step()

    def _configure_arm_servos(self) -> None:
        if self._arm_damping_ratio is not None:
            kv = configure_arm_servos(self._env.model, self._env.data, self._map, self._arm_damping_ratio)
            logger.info("Sim arm servo damping kv=%s (simulation tuning, not measured hardware)",
                        np.round(kv, 3).tolist())

    def run_loop(
        self,
        actions_q: queue.Queue[Any],
        stop_event: threading.Event,
    ) -> None:
        assert self._env is not None, "connect() must be called before run_loop()"
        ticker = RateTicker(dt=self._dt)

        while not stop_event.is_set():
            shutdown = False
            try:
                item = actions_q.get_nowait()
                if item is _SHUTDOWN:
                    shutdown = True
                else:
                    try:
                        self.set_target(item)
                    except (TypeError, ValueError):
                        logger.exception("Am200SimSink rejected an action; holding last target")
            except queue.Empty:
                pass

            if shutdown:
                break

            self._warn_if_stale()

            try:
                self._step()
            except Exception:
                logger.exception("combined env.step() failed")
                break

            ticker.tick()

    def _warn_if_stale(self) -> None:
        if self._stale_logged or self._last_command_ns is None:
            return
        idle_s = (time.monotonic_ns() - self._last_command_ns) / 1e9
        if idle_s >= STALE_COMMAND_WARN_S:
            logger.warning(
                "No action for %.1fs — holding the last target. Check the publisher.", idle_s
            )
            self._stale_logged = True

    def close(self) -> None:
        if self._env is None:
            return
        try:
            self._cameras.close()
            if self._renderer is not None:
                self._renderer.close()
            self._env.close()
        except Exception:
            logger.exception("Am200SimSink.close() encountered an error")
        finally:
            self._env = None
            self._renderer = None
            self._record_camera = None
            self._extra_cameras = {}

    def _load_hand_config(self) -> Any:
        """Load the orca_core hand config that defines the hand's joint order."""
        if self._hand_config_path is None:
            import os

            env_path = os.environ.get("ORCA_MODEL_PATH")
            if not env_path:
                raise RuntimeError(
                    "No hand config given. Pass hand_config_path or set $ORCA_MODEL_PATH "
                    "to the orca_core config.yaml for this hand."
                )
            self._hand_config_path = env_path
        return OrcaHand(str(self._hand_config_path)).config
