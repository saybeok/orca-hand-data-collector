"""Manipulation tasks for the combined AM-ARM200 + ORCA right hand sim.

A task owns what a scene adds on top of the robot: where objects spawn, what
counts as success, and a shaped reward. :class:`AmArm200TaskEnv` wires one into
the gym env; :class:`~orca_teleop.am200.sim_sink.Am200SimSink` uses the same env
for teleop and dataset recording, so demos and policy rollouts see one task.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from orca_teleop.am200.dynamics import SimDynamics
from orca_teleop.am200.sim_env import DEFAULT_MODEL_PATH, AmArm200OrcaRightEnv

THUMB_TIP_BODY = "right_thumb_dp"
FINGER_TIP_BODIES = ("right_index_ip", "right_middle_ip")
PALM_BODIES = ("right_palm", "right_tower")


@dataclass(frozen=True)
class PinchState:
    lift_m: float
    reach_m: float
    thumb_contact: bool
    finger_contact: bool
    palm_contact: bool
    pinching: bool
    held_s: float
    success: bool
    dropped: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlaceState(PinchState):
    picked: bool
    delivered: bool
    robot_contact: bool
    place_error_m: float
    placed_s: float


class PinchLiftTask:
    """Pick a small upright cylinder off the desk with a thumb-finger pinch.

    Success: the object is at least ``lift_m`` above where it rested, touched by
    the thumb and the index or middle fingertip but not the palm, continuously
    for ``hold_s``. Palm contact is excluded so a power grasp does not count.
    """

    name = "pinch"
    scene_file = "scene_pinch.xml"
    object_body = "pinch_object"
    object_joint = "pinch_object_free"
    support_geom = "floor"

    def __init__(
        self,
        radius_range_m: tuple[float, float] = (0.35, 0.42),
        azimuth_range_deg: float = 25.0,
        lift_m: float = 0.05,
        hold_s: float = 0.5,
    ) -> None:
        self.radius_range_m = radius_range_m
        self.azimuth_range_deg = azimuth_range_deg
        self.lift_m = lift_m
        self.hold_s = hold_s
        self._bound = False
        self._rest_z = 0.0
        self._held_s = 0.0
        self._succeeded = False

    def bind(self, model: mujoco.MjModel) -> None:
        def body(name: str) -> int:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(f"model has no body {name!r}; is this the {self.name} scene?")
            return bid

        self._obj = body(self.object_body)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self.object_joint)
        if jid < 0:
            raise ValueError(f"model has no joint {self.object_joint!r}")
        self._obj_qadr = int(model.jnt_qposadr[jid])
        self._obj_dadr = int(model.jnt_dofadr[jid])
        self._thumb = body(THUMB_TIP_BODY)
        self._fingers = {body(n) for n in FINGER_TIP_BODIES}
        self._palm = {body(n) for n in PALM_BODIES}
        self._tip_geoms = {
            b: [
                g for g in range(model.ngeom)
                if model.geom_bodyid[g] == b and model.geom_contype[g] | model.geom_conaffinity[g]
            ]
            for b in (self._thumb, *self._fingers)
        }
        # Resting centre height: half the object above the supporting surface.
        geom = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == self._obj)
        self._rest_z = _geom_top_z(model, self.support_geom) + _half_height(model, geom)
        self._bound = True

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> np.ndarray:
        """Respawn the object upright at a random reachable spot; returns its xyz."""
        assert self._bound, "bind() first"
        r = rng.uniform(*self.radius_range_m)
        az = np.deg2rad(rng.uniform(-self.azimuth_range_deg, self.azimuth_range_deg))
        # Azimuth 0 is straight ahead of the arm's zero pose (world -y).
        pos = np.array([r * np.sin(az), -r * np.cos(az), self._rest_z + 0.0005])
        yaw = rng.uniform(-np.pi, np.pi)
        quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        data.qpos[self._obj_qadr:self._obj_qadr + 3] = pos
        data.qpos[self._obj_qadr + 3:self._obj_qadr + 7] = quat
        data.qvel[self._obj_dadr:self._obj_dadr + 6] = 0.0
        self._held_s = 0.0
        self._succeeded = False
        return pos

    def _tip_point(self, data: mujoco.MjData, body: int) -> np.ndarray:
        return np.mean([data.geom_xpos[g] for g in self._tip_geoms[body]], axis=0)

    def update(self, model: mujoco.MjModel, data: mujoco.MjData, dt: float) -> PinchState:
        touching: set[int] = set()
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            if b1 == self._obj:
                touching.add(b2)
            elif b2 == self._obj:
                touching.add(b1)

        thumb = self._thumb in touching
        finger = bool(touching & self._fingers)
        palm = bool(touching & self._palm)
        obj = data.xpos[self._obj]
        lift = float(obj[2] - self._rest_z)

        pinch_pt = 0.5 * (
            self._tip_point(data, self._thumb)
            + np.mean([self._tip_point(data, b) for b in self._fingers], axis=0)
        )
        pinching = thumb and finger and not palm
        if pinching and lift >= self.lift_m:
            self._held_s += dt
        else:
            self._held_s = 0.0
        self._succeeded = self._succeeded or self._held_s >= self.hold_s

        return PinchState(
            lift_m=lift,
            reach_m=float(np.linalg.norm(pinch_pt - obj)),
            thumb_contact=thumb,
            finger_contact=finger,
            palm_contact=palm,
            pinching=pinching,
            held_s=self._held_s,
            success=self._succeeded,
            dropped=bool(obj[2] < self._rest_z - 0.05),
        )

    def reward(self, s: PinchState) -> float:
        reach = 1.0 - np.tanh(10.0 * s.reach_m)
        contact = 0.25 * (s.thumb_contact + s.finger_contact) - 0.25 * s.palm_contact
        lift = float(np.clip(s.lift_m / self.lift_m, 0.0, 1.0)) if s.pinching else 0.0
        return float(reach + contact + lift + 2.0 * s.success)

    def object_pose(self, data: mujoco.MjData) -> np.ndarray:
        """Object position + wxyz quaternion (7,)."""
        return data.qpos[self._obj_qadr:self._obj_qadr + 7].copy()


class HwatuLiftTask(PinchLiftTask):
    """Pick a flat hwatu card (35 x 53 x 1 mm) off a raised block.

    Same success rule as the cylinder pinch. The card spawns flat on the block,
    long axis roughly along +x, a few centimetres short of the block's +x side
    edge so it can be slid out to overhang before pinching.
    """

    name = "hwatu"
    scene_file = "scene_hwatu.xml"
    object_body = "hwatu_card"
    object_joint = "hwatu_card_free"
    support_geom = "hwatu_block"

    def __init__(
        self,
        y_range_m: tuple[float, float] = (-0.39, -0.35),
        edge_gap_range_m: tuple[float, float] = (0.01, 0.05),
        yaw_range_deg: float = 10.0,
        lift_m: float = 0.05,
        hold_s: float = 0.5,
    ) -> None:
        super().__init__(lift_m=lift_m, hold_s=hold_s)
        self.y_range_m = y_range_m
        self.edge_gap_range_m = edge_gap_range_m
        self.yaw_range_deg = yaw_range_deg

    def bind(self, model: mujoco.MjModel) -> None:
        super().bind(model)
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self.support_geom)
        self.block_edge_x = float(model.geom_pos[g][0] + model.geom_size[g][0])
        card = next(i for i in range(model.ngeom) if model.geom_bodyid[i] == self._obj)
        self.card_half_extents = model.geom_size[card].copy()

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> np.ndarray:
        """Spawn the card flat; ``edge_gap`` is from its +x end to the block's +x edge."""
        assert self._bound, "bind() first"
        gap = rng.uniform(*self.edge_gap_range_m)
        pos = np.array([
            self.block_edge_x - gap - self.card_half_extents[0],
            rng.uniform(*self.y_range_m),
            self._rest_z + 0.0002,
        ])
        yaw = np.deg2rad(rng.uniform(-self.yaw_range_deg, self.yaw_range_deg))
        quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        data.qpos[self._obj_qadr:self._obj_qadr + 3] = pos
        data.qpos[self._obj_qadr + 3:self._obj_qadr + 7] = quat
        data.qvel[self._obj_dadr:self._obj_dadr + 6] = 0.0
        self._held_s = 0.0
        self._succeeded = False
        return pos


class HwatuMatTask(HwatuLiftTask):
    """Pick a hwatu card straight off a soft blanket, away from any edge."""

    name = "hwatu_mat"
    scene_file = "scene_hwatu_mat.xml"
    support_geom = "hwatu_mat"

    def __init__(
        self,
        x_range_m: tuple[float, float] = (-0.03, 0.05),
        y_range_m: tuple[float, float] = (-0.39, -0.35),
        yaw_range_deg: float = 15.0,
        lift_m: float = 0.05,
        hold_s: float = 0.5,
    ) -> None:
        super().__init__(y_range_m=y_range_m, yaw_range_deg=yaw_range_deg, lift_m=lift_m, hold_s=hold_s)
        self.x_range_m = x_range_m

    def bind(self, model: mujoco.MjModel) -> None:
        super().bind(model)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hwatu_target")
        self._target_mocap = int(model.body_mocapid[bid]) if bid >= 0 else -1

    def _spawn_card(self, data: mujoco.MjData, rng: np.random.Generator) -> np.ndarray:
        pos = np.array([rng.uniform(*self.x_range_m), rng.uniform(*self.y_range_m), self._rest_z])
        yaw = np.deg2rad(rng.uniform(-self.yaw_range_deg, self.yaw_range_deg))
        data.qpos[self._obj_qadr:self._obj_qadr + 3] = pos
        data.qpos[self._obj_qadr + 3:self._obj_qadr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        data.qvel[self._obj_dadr:self._obj_dadr + 6] = 0.0
        self._held_s = 0.0
        self._succeeded = False
        return pos

    def _set_target(self, data: mujoco.MjData, xy: np.ndarray | None) -> None:
        if self._target_mocap < 0:
            return
        top = self._rest_z - self.card_half_extents[2]
        data.mocap_pos[self._target_mocap] = [0.0, 0.0, -1.0] if xy is None else [xy[0], xy[1], top]

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> np.ndarray:
        assert self._bound, "bind() first"
        self._set_target(data, None)
        return self._spawn_card(data, rng)


class HwatuPlaceTask(HwatuMatTask):
    """Pick the card off the blanket and put it down flat on the yellow target.

    Success: the card was pinched and lifted at least ``pick_lift_m``, carried
    still pinched to within ``deliver_radius_m`` of the target and
    ``deliver_height_m`` of the blanket (so a card dropped from the air does not
    count), then rests flat on the blanket with its centre within
    ``place_tol_m`` of the target, untouched by the robot, for ``hold_s``. The
    target spawns ``target_dist_range_m`` away from the card, anywhere in the
    same region.
    """

    name = "hwatu_place"

    def __init__(
        self,
        x_range_m: tuple[float, float] = (-0.12, 0.12),
        y_range_m: tuple[float, float] = (-0.42, -0.34),
        target_dist_range_m: tuple[float, float] = (0.08, 0.18),
        yaw_range_deg: float = 15.0,
        pick_lift_m: float = 0.015,
        place_tol_m: float = 0.02,
        deliver_radius_m: float = 0.04,
        deliver_height_m: float = 0.01,
        hold_s: float = 0.5,
    ) -> None:
        super().__init__(x_range_m=x_range_m, y_range_m=y_range_m, yaw_range_deg=yaw_range_deg, hold_s=hold_s)
        self.target_dist_range_m = target_dist_range_m
        self.pick_lift_m = pick_lift_m
        self.place_tol_m = place_tol_m
        self.deliver_radius_m = deliver_radius_m
        self.deliver_height_m = deliver_height_m
        self.target_xy = np.zeros(2)
        self._picked = False
        self._delivered = False
        self._placed_s = 0.0
        self._placed = False

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> np.ndarray:
        assert self._bound, "bind() first"
        pos = self._spawn_card(data, rng)
        lo, hi = self.target_dist_range_m
        while True:
            xy = np.array([rng.uniform(*self.x_range_m), rng.uniform(*self.y_range_m)])
            if lo <= np.linalg.norm(xy - pos[:2]) <= hi:
                break
        self.target_xy = xy
        self._set_target(data, xy)
        self._picked = False
        self._delivered = False
        self._placed_s = 0.0
        self._placed = False
        return pos

    def update(self, model: mujoco.MjModel, data: mujoco.MjData, dt: float) -> PlaceState:
        s = super().update(model, data, dt)
        robot = False
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            if self._obj in (b1, b2) and 0 not in (b1, b2):
                robot = True
                break
        self._picked = self._picked or (s.pinching and s.lift_m >= self.pick_lift_m)
        card = data.xpos[self._obj]
        err = float(np.linalg.norm(card[:2] - self.target_xy))
        self._delivered = self._delivered or (
            self._picked and s.pinching and err <= self.deliver_radius_m and s.lift_m <= self.deliver_height_m
        )
        flat = data.xmat[self._obj][8] > np.cos(np.deg2rad(10.0))
        still = np.linalg.norm(data.qvel[self._obj_dadr:self._obj_dadr + 3]) < 0.02
        resting = flat and still and abs(card[2] - self._rest_z) < 0.003
        if self._delivered and resting and not robot and err <= self.place_tol_m:
            self._placed_s += dt
        else:
            self._placed_s = 0.0
        self._placed = self._placed or self._placed_s >= self.hold_s
        fields = {**s.as_dict(), "success": self._placed}
        return PlaceState(
            **fields,
            picked=self._picked,
            delivered=self._delivered,
            robot_contact=robot,
            place_error_m=err,
            placed_s=self._placed_s,
        )

    def reward(self, s: PinchState) -> float:
        assert isinstance(s, PlaceState)
        reach = 1.0 - np.tanh(10.0 * s.reach_m)
        if not s.picked:
            return float(reach + 0.25 * (s.thumb_contact + s.finger_contact) - 0.25 * s.palm_contact)
        return float(1.5 + (1.0 - np.tanh(10.0 * s.place_error_m)) + 2.0 * s.success)


def _geom_top_z(model: mujoco.MjModel, name: str) -> float:
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if g < 0:
        return 0.0
    top = float(model.geom_pos[g][2])
    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
        top += float(model.geom_size[g][2])
    return top


def _half_height(model: mujoco.MjModel, geom: int) -> float:
    kind = model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return float(model.geom_size[geom][2])
    if kind in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        return float(model.geom_size[geom][1])
    return float(model.geom_size[geom][0])


TASKS: dict[str, type[PinchLiftTask]] = {
    PinchLiftTask.name: PinchLiftTask,
    HwatuLiftTask.name: HwatuLiftTask,
    HwatuMatTask.name: HwatuMatTask,
    HwatuPlaceTask.name: HwatuPlaceTask,
}


def make_task(name: str) -> PinchLiftTask:
    try:
        return TASKS[name]()
    except KeyError:
        raise ValueError(f"unknown task {name!r}; choose from {sorted(TASKS)}") from None


def task_scene_path(task: PinchLiftTask) -> Path:
    return DEFAULT_MODEL_PATH.parent / task.scene_file


class AmArm200TaskEnv(AmArm200OrcaRightEnv):
    """The combined env plus a task: randomized reset, reward, success in ``info``.

    ``terminated`` flips on success or when the object falls off the desk.
    """

    def __init__(
        self,
        task: PinchLiftTask,
        model_path: str | Path | None = None,
        frame_skip: int | None = None,
        render_mode: str | None = None,
        dynamics: SimDynamics | str | Path | None = None,
        randomize: bool | None = None,
    ) -> None:
        self.task = task
        self._state: PinchState | None = None
        super().__init__(
            model_path=model_path or task_scene_path(task),
            frame_skip=frame_skip,
            render_mode=render_mode,
            dynamics=dynamics,
            randomize=randomize,
        )
        task.bind(self.model)

    @property
    def control_dt(self) -> float:
        return self.frame_skip * float(self.model.opt.timestep)

    @property
    def state(self) -> PinchState | None:
        return self._state

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.respawn_object()
        return self._get_obs(), self._get_info()

    def respawn_object(self) -> np.ndarray:
        pos = self.task.reset(self.model, self.data, self.np_random)
        mujoco.mj_forward(self.model, self.data)
        self._state = self.task.update(self.model, self.data, 0.0)
        return pos

    def _get_reward(self) -> float:
        self._state = self.task.update(self.model, self.data, self.control_dt)
        return self.task.reward(self._state)

    def _get_terminated(self) -> bool:
        return self._state is not None and (self._state.success or self._state.dropped)

    def _get_info(self) -> dict[str, Any]:
        if self._state is None:
            return {}
        return {**self._state.as_dict(), "object_pose": self.task.object_pose(self.data)}
