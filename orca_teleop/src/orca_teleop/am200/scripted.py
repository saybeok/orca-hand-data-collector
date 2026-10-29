"""Scripted pinch expert for the AM-ARM200 + ORCA pinch task (privileged state).

At each episode start it reads the object pose from the sim and solves four
whole-robot keyframes by least squares over the arm, wrist, thumb and index
joints: open hand above the object, open hand around it, closed pinch, and the
closed pinch lifted. Each keyframe pins the thumb/index tip midpoint to a
target, sets the tip gap, keeps the tip line level and the palm off the desk.
Between keyframes it follows min-jerk joint-space interpolation and adds a
gravity feedforward so the position servos land where the plan says.

Uses: a success baseline for the evaluator, a check that the task is
physically solvable, and a source of automatic demonstrations.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from scipy.optimize import least_squares

PLAN_JOINTS = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "right_wrist",
    "right_thumb_mcp", "right_thumb_abd", "right_thumb_pip", "right_thumb_dip",
    "right_index_abd", "right_index_mcp", "right_index_pip",
)
INDEX_TIP_BODY = "right_index_ip"


@dataclass
class PinchPlanConfig:
    # Gaps are between tip *reference points* (centres of the distal collision
    # meshes, ~9 mm inside the pad), so the 22 mm cylinder is first touched at
    # about 40 mm. The closed gap squeezes it lightly; squeezing much harder makes
    # the thumb and index slide past each other and pop the cylinder out sideways.
    open_gap_m: float = 0.08
    closed_gap_m: float = 0.040
    # Heights are relative to the object's resting centre.
    approach_height_m: float = 0.08
    grasp_height_m: float = -0.006
    lift_height_m: float = 0.11
    palm_clearance_m: float = 0.06
    # Closing passes through these intermediate gaps, each re-solved with the
    # midpoint on the object, so both tips converge on it symmetrically.
    close_gaps_m: tuple[float, ...] = (0.065, 0.052)
    # phase durations (s): approach, descend, close (split over the gaps), settle, lift, hold
    durations: tuple[float, ...] = (2.0, 1.5, 1.5, 0.4, 1.5, 3.0)


def _min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


class PinchKinematics:
    """Whole-robot pinch IK on a private MjData."""

    def __init__(self, model: mujoco.MjModel, task: Any) -> None:
        self.m = model
        self.d = mujoco.MjData(model)
        self.task = task
        jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in PLAN_JOINTS]
        if min(jid) < 0:
            raise ValueError("model is missing planned joints")
        self.qadr = np.array([model.jnt_qposadr[j] for j in jid])
        self.lo = model.jnt_range[jid, 0].copy()
        self.hi = model.jnt_range[jid, 1].copy()
        self.lo[:3], self.hi[:3] = -np.pi, np.pi
        self.index_tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, INDEX_TIP_BODY)
        self.palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_palm")
        self.tip_vertical_weight = 1.0
        self.avoid_geoms: tuple[int, ...] = ()
        self.avoid_margin_m = 0.0
        # Bodies allowed into ``avoid_geoms`` while pinching (e.g. tips pressing a soft mat).
        self.pinch_skip: frozenset[int] = frozenset()
        self._obj_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == task._obj}

    def set_avoid(self, geom_names: tuple[str, ...], margin_m: float = 0.003) -> None:
        """Keep the robot ``margin_m`` away from static scene geoms (e.g. a block edge).

        Works on a private model copy, since contact margins would change the sim.
        """
        self.m = copy.deepcopy(self.m)
        self.d = mujoco.MjData(self.m)
        ids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in geom_names]
        if min(ids, default=0) < 0:
            raise ValueError(f"unknown geoms in {geom_names}")
        self.avoid_geoms = tuple(ids)
        self.avoid_margin_m = margin_m
        self.m.geom_margin[ids] = margin_m

    def tips(self, qfull: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.d.qpos[:] = qfull
        if self.avoid_geoms:
            mujoco.mj_fwdPosition(self.m, self.d)
        else:
            mujoco.mj_kinematics(self.m, self.d)
        return (
            self.task._tip_point(self.d, self.task._thumb),
            self.task._tip_point(self.d, self.index_tip),
        )

    def penetration(self, skip_bodies: frozenset[int] = frozenset()) -> float:
        """Root-sum-square robot penetration into ``avoid_geoms`` (after :meth:`tips`)."""
        total = 0.0
        for c in self.d.contact[: self.d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in self.avoid_geoms:
                other = g2
            elif g2 in self.avoid_geoms:
                other = g1
            else:
                continue
            if other in self._obj_geoms or self.m.geom_bodyid[other] in skip_bodies:
                continue
            total += max(0.0, self.avoid_margin_m - float(c.dist)) ** 2
        return float(np.sqrt(total))

    def solve(
        self,
        base: np.ndarray,
        seed: np.ndarray,
        target_mid: np.ndarray,
        gap: float | None,
        palm_clearance: float,
        reg: float = 0.03,
        free: slice | None = None,
        line_dir: np.ndarray | None = None,
        thumb_lift_m: float = 0.04,
        level: bool = False,
    ) -> tuple[np.ndarray, float]:
        """Return (full qpos with planned joints set, residual norm).

        ``free`` selects which of :data:`PLAN_JOINTS` may move; the rest stay
        at ``seed``. The gap term is skipped when the hand is frozen; ``level``
        still keeps the thumb-index line level then (e.g. arm + wrist moving a
        held grip).

        ``gap=None`` is a single-finger poke: ``target_mid`` is the index tip
        and the thumb is kept ``thumb_lift_m`` above it. ``line_dir`` (a
        horizontal unit vector, either sign) fixes the thumb-index direction.
        """
        free = free or slice(None)
        qfull = base.copy()
        x_all = np.clip(seed, self.lo + 1e-6, self.hi - 1e-6)
        hand_free = free == slice(None)

        def resid(x: np.ndarray) -> np.ndarray:
            x_all[free] = x
            qfull[self.qadr] = x_all
            a, b = self.tips(qfull)
            palm_z = self.d.xpos[self.palm][2]
            if not self.avoid_geoms:
                avoid = []
            elif gap is None:
                # The poking pad has to get within a card's thickness of the surface.
                avoid = [200.0 * self.penetration(frozenset({self.index_tip}))]
            else:
                avoid = [200.0 * self.penetration(self.pinch_skip)]
            if gap is None:
                index_z = self.d.xmat[self.index_tip].reshape(3, 3)[2, 2]
                return np.concatenate([
                    20.0 * (b - target_mid),
                    [5.0 * max(0.0, b[2] + thumb_lift_m - a[2]), index_z + 1.0],
                    [5.0 * max(0.0, palm_clearance - palm_z)],
                    avoid,
                    reg * (x - seed[free]),
                ])
            terms = [20.0 * ((a + b) / 2 - target_mid), avoid]
            if line_dir is not None:
                u = (b - a) / max(float(np.linalg.norm(b - a)), 1e-6)
                terms.append(3.0 * (u - float(u @ line_dir) * line_dir))
            if hand_free:
                # Distal links point down (local +z along the phalanx) so the pads
                # squeeze horizontally instead of wedging the object downward.
                thumb_z = self.d.xmat[self.task._thumb].reshape(3, 3)[2, 2]
                index_z = self.d.xmat[self.index_tip].reshape(3, 3)[2, 2]
                terms.append([
                    10.0 * (np.linalg.norm(a - b) - gap),
                    3.0 * (a[2] - b[2]),
                    self.tip_vertical_weight * (thumb_z + 1.0),
                    self.tip_vertical_weight * (index_z + 1.0),
                ])
            elif level:
                terms.append([10.0 * (a[2] - b[2])])
            terms.append([5.0 * max(0.0, palm_clearance - palm_z)])
            terms.append(reg * (x - seed[free]))
            return np.concatenate(terms)

        sol = least_squares(
            resid, x_all[free].copy(), bounds=(self.lo[free], self.hi[free]), xtol=1e-10, ftol=1e-10
        )
        x_all[free] = sol.x
        qfull[self.qadr] = x_all
        a, b = self.tips(qfull)
        err = float(np.linalg.norm((b if gap is None else (a + b) / 2) - target_mid))
        return qfull.copy(), err


class ScriptedPinchPolicy:
    """Privileged scripted expert; call :meth:`reset` after the scene reset."""

    def __init__(self, sink: Any, config: PinchPlanConfig | None = None) -> None:
        self.sink = sink
        self.cfg = config or PinchPlanConfig()
        env = sink._env
        self.m = env.model
        self.task = env.task
        self.kin = PinchKinematics(self.m, self.task)
        self.dt = env.control_dt
        self._scratch = mujoco.MjData(self.m)
        self._act_q = np.array([self.m.jnt_qposadr[self.m.actuator_trnid[a, 0]] for a in range(self.m.nu)])
        self._act_d = np.array([self.m.jnt_dofadr[self.m.actuator_trnid[a, 0]] for a in range(self.m.nu)])
        self.keyframes: list[np.ndarray] = []
        self.segment_s: list[float] = []
        # Segments chained linearly (no min-jerk easing), e.g. finely sampled paths.
        self._linear: set[int] = set()
        self.plan_error_m: float = float("nan")
        self._step = 0

    def reset(self) -> None:
        env = self.sink._env
        obj = self.task.object_pose(env.data)[:3]
        start = env.data.qpos.copy()
        c = self.cfg
        up = np.array([0.0, 0.0, 1.0])

        grasp_mid = obj + up * c.grasp_height_m
        errs = []

        def solve(
            prev: np.ndarray, mid: np.ndarray, gap: float, clearance: float = 0.0, free: slice | None = None
        ) -> np.ndarray:
            q, e = self.kin.solve(prev, prev[self.kin.qadr], mid, gap, clearance, free=free)
            errs.append(e)
            return q

        above = solve(start, obj + up * c.approach_height_m, c.open_gap_m, c.palm_clearance_m)
        around = solve(above, grasp_mid, c.open_gap_m)
        closing = [around]
        for gap in (*c.close_gaps_m, c.closed_gap_m):
            closing.append(solve(closing[-1], grasp_mid, gap))
        closed = closing[-1]
        # Lift with the arm only: re-solving the fingers here would loosen the grip.
        lifted = solve(closed, obj + up * c.lift_height_m, c.closed_gap_m, free=slice(0, 3))

        approach_s, descend_s, close_s, settle_s, lift_s, hold_s = c.durations
        n_close = len(closing) - 1
        self.keyframes = [start, above, *closing, closed, lifted, lifted]
        self.segment_s = [approach_s, descend_s, *([close_s / n_close] * n_close), settle_s, lift_s, hold_s]
        self._linear = set(range(2, 2 + n_close))
        self.plan_error_m = float(max(errs))
        self._step = 0

    def _planned_qpos(self, t: float) -> np.ndarray:
        acc = 0.0
        for i, dur in enumerate(self.segment_s):
            if t < acc + dur:
                s = (t - acc) / dur
                if i not in self._linear:
                    s = _min_jerk(s)
                return (1 - s) * self.keyframes[i] + s * self.keyframes[i + 1]
            acc += dur
        return self.keyframes[-1]

    def _with_gravity_ff(self, qpos: np.ndarray) -> np.ndarray:
        """Actuator targets so position servos hold ``qpos`` against gravity."""
        m, d = self.m, self._scratch
        d.qpos[:] = qpos
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        kp = m.actuator_gainprm[:, 0]
        ctrl = qpos[self._act_q] + d.qfrc_bias[self._act_d] / kp
        return np.clip(ctrl, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])

    def select_action(self, observation: dict[str, Any]) -> np.ndarray:
        if not self.keyframes:
            self.reset()
        qpos = self._planned_qpos(self._step * self.dt)
        self._step += 1
        ctrl = self._with_gravity_ff(qpos)
        return self.sink._map.from_qpos(self._ctrl_as_qpos(ctrl))

    def _ctrl_as_qpos(self, ctrl: np.ndarray) -> np.ndarray:
        out = np.zeros(self.m.nq)
        out[self._act_q] = ctrl
        return out


@dataclass
class CardPlanConfig:
    # With the distal link pointing down, the index tip reference is ~27 mm
    # above the end of the pad.
    pad_offset_m: float = 0.027
    press_depth_m: float = 0.004
    # Press this far behind the card centre, so the pressed spot stays over
    # the block once the card overhangs.
    press_back_m: float = 0.012
    # Planned overhang past the edge; the pad slips ~6 mm on the card, so it
    # ends up ~22 mm (card half-length is 26.5 mm; aiming 32 mm tips it off).
    overhang_m: float = 0.028
    # Pinch across the card's width this far past the edge. Closer than ~13 mm
    # the fingertip meshes hit the side of the block.
    pinch_out_m: float = 0.016
    open_gap_m: float = 0.085
    close_gaps_m: tuple[float, ...] = (0.070, 0.060)
    closed_gap_m: float = 0.050
    grasp_height_m: float = 0.0
    lift_height_m: float = 0.10
    # Palm height above the card while hovering.
    palm_clearance_m: float = 0.05
    drag_steps: int = 5
    # Middle/ring/pinky MCP+PIP flexion while poking (curled out of the way of
    # the block) and while pinching.
    curl_others_rad: float = 1.3
    pinch_curl_rad: float = 0.0
    # phase durations (s): approach, press, drag, release, pinch approach,
    # descend, close, settle, lift, hold
    durations: tuple[float, ...] = (2.0, 1.0, 2.0, 0.6, 1.5, 1.2, 1.2, 0.4, 1.5, 2.0)


class ScriptedHwatuPolicy(ScriptedPinchPolicy):
    """Privileged expert for the hwatu task: slide the card over the edge, pinch the overhang.

    The slide is planned at reset from the spawned pose; the pinch is re-planned
    from the card's actual pose once the index finger has lifted off it.
    """

    def __init__(self, sink: Any, config: CardPlanConfig | None = None) -> None:
        super().__init__(sink)
        self.card_cfg = config or CardPlanConfig()
        self.kin.set_avoid((self.task.support_geom,), 0.003)
        curl = [f"right_{f}_{j}" for f in ("middle", "ring", "pinky") for j in ("mcp", "pip")]
        self._curl_adr = [
            self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in curl
        ]
        self._pinch_planned = False
        self._errs: list[float] = []

    def _card_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pose = self.task.object_pose(self.sink._env.data)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, pose[3:])
        rot = rot.reshape(3, 3)
        return pose[:3], rot[:, 0], rot[:, 1]

    def _solve(self, prev: np.ndarray, target: np.ndarray, gap: float | None, **kw: Any) -> np.ndarray:
        clearance = kw.pop("clearance", 0.0)
        q, e = self.kin.solve(prev, prev[self.kin.qadr], target, gap, clearance, **kw)
        self._errs.append(e)
        return q

    def reset(self) -> None:
        c = self.card_cfg
        env = self.sink._env
        start = env.data.qpos.copy()
        center, ux, uy = self._card_frame()
        half = self.task.card_half_extents
        up = np.array([0.0, 0.0, 1.0])
        ex = np.array([1.0, 0.0, 0.0])
        hover = center[2] + c.palm_clearance_m + c.pad_offset_m

        press = center - ux * c.press_back_m
        press[2] = center[2] + half[2] + c.pad_offset_m - c.press_depth_m
        far_end_x = center[0] + half[0] * abs(ux[0]) + half[1] * abs(uy[0])
        travel = self.task.block_edge_x + c.overhang_m - far_end_x

        self._errs = []
        curled = start.copy()
        curled[self._curl_adr] = c.curl_others_rad
        above = self._solve(curled, press + up * 0.03, None, clearance=hover)
        pressed = self._solve(above, press, None)
        drag = [pressed]
        for k in range(1, c.drag_steps + 1):
            drag.append(self._solve(drag[-1], press + ex * travel * k / c.drag_steps, None))
        released = self._solve(drag[-1], press + ex * travel + up * 0.03, None)

        approach_s, press_s, drag_s, release_s = c.durations[:4]
        self.keyframes = [start, above, *drag, released]
        self.segment_s = [approach_s, press_s, *([drag_s / c.drag_steps] * c.drag_steps), release_s]
        self._linear = set(range(2, 2 + c.drag_steps))
        self.plan_error_m = float(max(self._errs))
        self._pinch_planned = False
        self._step = 0

    def _plan_pinch(self) -> None:
        c = self.card_cfg
        center, ux, _ = self._card_frame()
        x_line = self.task.block_edge_x + c.pinch_out_m
        mid = center + ux * (x_line - center[0]) / ux[0]
        mid[2] = center[2] + c.grasp_height_m
        self._append_pinch(mid)

    def _append_pinch(self, mid: np.ndarray) -> None:
        """Append open-above, descend, close, lift keyframes pinching across the card at ``mid``."""
        c = self.card_cfg
        center, ux, uy = self._card_frame()
        up = np.array([0.0, 0.0, 1.0])
        line = np.array([uy[0], uy[1], 0.0])
        line /= np.linalg.norm(line)
        hover = center[2] + c.palm_clearance_m

        prev = self.keyframes[-1].copy()
        prev[self._curl_adr] = c.pinch_curl_rad
        above = self._solve(prev, mid + np.array([0.01, 0.0, 0.05]), c.open_gap_m, clearance=hover, line_dir=line)
        around = self._solve(above, mid, c.open_gap_m, line_dir=line)
        closing = [around]
        for gap in (*c.close_gaps_m, c.closed_gap_m):
            closing.append(self._solve(closing[-1], mid, gap, line_dir=line))
        closed = closing[-1]
        lifted = self._solve(closed, mid + up * c.lift_height_m, c.closed_gap_m, free=slice(0, 3))

        pinch_s, descend_s, close_s, settle_s, lift_s, hold_s = c.durations[4:]
        n_close = len(closing) - 1
        first_close = len(self.segment_s) + 2
        self.keyframes += [above, *closing, closed, lifted, lifted]
        self.segment_s += [pinch_s, descend_s, *([close_s / n_close] * n_close), settle_s, lift_s, hold_s]
        self._linear |= set(range(first_close, first_close + n_close))
        self.plan_error_m = float(max(self._errs))
        self.overhang_at_pinch_m = float(
            center[0] + self.task.card_half_extents[0] * abs(ux[0]) - self.task.block_edge_x
        )
        self._pinch_planned = True

    def select_action(self, observation: dict[str, Any]) -> np.ndarray:
        if not self.keyframes:
            self.reset()
        if not self._pinch_planned and self._step * self.dt >= sum(self.segment_s):
            self._plan_pinch()
        return super().select_action(observation)


@dataclass
class MatPinchConfig:
    # Tip-midpoint heights above the card centre. With the distal links pointing
    # down the pads are ~22 mm below the tip reference points, so the press
    # target is well below the mat; the servos stall against it. Shallower, the
    # rounded pads meet the card's top corner and push it down; much deeper
    # the grip gets less repeatable again.
    touch_height_m: float = 0.022
    press_height_m: float = -0.012
    open_gap_m: float = 0.085
    # Close to just outside the card's long edges before pressing: pads pressed
    # into the mat cannot slide far against its friction.
    pre_gap_m: float = 0.052
    # No closing past the pre-close gap: the pads pressed into the blanket
    # already hold the edges, and squeezing further wedges the card into a roll
    # (0.044 dropped every card picked from x < 0).
    closed_gap_m: float = 0.052
    lift_height_m: float = 0.10
    palm_clearance_m: float = 0.05
    # Middle/ring/pinky MCP+PIP flexion. Straight, the middle finger's IP
    # reaches the blanket before the pads are pressed in; curled, the fingers
    # land on the card. Slightly extended keeps them up and clear.
    curl_others_rad: float = -0.3
    # phase durations (s): approach, descend, pre-close, press, close, settle, lift, hold
    durations: tuple[float, ...] = (2.0, 1.2, 0.8, 0.6, 0.8, 0.4, 1.5, 2.0)


class ScriptedMatPinchPolicy(ScriptedHwatuPolicy):
    """Pinch a flat card straight off a soft mat, across its width, with no slide.

    The tips close to just outside the card's long edges, press into the mat so
    the pads sit below the card's bottom face, then close on the edges.
    """

    def __init__(self, sink: Any, config: MatPinchConfig | None = None) -> None:
        super().__init__(sink)
        self.mat_cfg = config or MatPinchConfig()
        self.kin.set_avoid((self.task.support_geom, "hwatu_block"), 0.003)
        self.kin.pinch_skip = frozenset({self.task._thumb, self.kin.index_tip})
        # Joints (of PLAN_JOINTS) that move the closed grip, and whether they
        # also keep the tip line level; the fingers themselves stay put.
        self.grip_free = slice(0, 3)
        self.grip_level = False

    def reset(self) -> None:
        c = self.mat_cfg
        start = self.sink._env.data.qpos.copy()
        start[self._curl_adr] = c.curl_others_rad
        center, _, uy = self._card_frame()
        line = np.array([uy[0], uy[1], 0.0])
        line /= np.linalg.norm(line)
        up = np.array([0.0, 0.0, 1.0])
        touch = center + up * c.touch_height_m
        press = center + up * c.press_height_m

        self._errs = []
        kw = {"line_dir": line}
        above = self._solve(start, touch + up * 0.05, c.open_gap_m, clearance=center[2] + c.palm_clearance_m, **kw)
        around = self._solve(above, touch, c.open_gap_m, **kw)
        pre = self._solve(around, touch, c.pre_gap_m, **kw)
        pressed = self._solve(pre, press, c.pre_gap_m, **kw)
        closed = self._solve(pressed, press, c.closed_gap_m, **kw)
        lifted = self._solve(
            closed, press + up * c.lift_height_m, c.closed_gap_m, free=self.grip_free, level=self.grip_level
        )

        self.keyframes = [start, above, around, pre, pressed, closed, closed, lifted, lifted]
        self.segment_s = list(c.durations)
        self._linear = set()
        self.plan_error_m = float(max(self._errs))
        self.overhang_at_pinch_m = float("nan")
        self._pinch_planned = True
        self._step = 0


@dataclass
class MatPlaceConfig:
    # Tip-midpoint lift above the press point while carrying. The press point
    # is below where the pads stall in the blanket, so the card itself rises
    # ~20 mm less. Kept low otherwise: the edge grip lets the card swing about
    # its long axis.
    carry_height_m: float = 0.055
    # Card bottom this far above the blanket when the fingers open.
    drop_height_m: float = 0.002
    closed_gap_m: float = MatPinchConfig.closed_gap_m
    # Move the held card with arm + wrist, keeping the tip line level. Arm-only
    # moves tilt it 7-10 deg per 5 cm of lift.
    level_wrist: bool = True
    open_gap_m: float = 0.085
    retreat_height_m: float = 0.08
    # The edge-held card creeps about its long axis (a few deg/s) and falls out
    # past ~10 deg, whatever the squeeze (the finger servos are already at their
    # force limit), so the lift-carry-lower stretch is kept under ~2 s: with a
    # 1.8 s carry the card fell in 6/8 seeds.
    # pick phase durations (s): approach, descend, pre-close, press, close,
    # settle, lift, pause
    pick_durations: tuple[float, ...] = (2.0, 1.2, 0.8, 0.6, 0.8, 0.4, 0.6, 0.02)
    # place phase durations (s): carry, lower, settle, open, retreat, hold
    durations: tuple[float, ...] = (0.7, 0.5, 0.3, 0.6, 1.0, 1.0)


class ScriptedMatPlacePolicy(ScriptedMatPinchPolicy):
    """Pick the card off the blanket, carry it low and put it down on the target.

    The pick is :class:`ScriptedMatPinchPolicy` with a short lift. Once lifted,
    the card's actual offset from the tip midpoint (in the palm frame) is read
    from the sim, and the carry/lower keyframes put the card centre, not the
    fingertips, over the target.
    """

    def __init__(self, sink: Any, config: MatPlaceConfig | None = None) -> None:
        self.place_cfg = config or MatPlaceConfig()
        super().__init__(
            sink,
            MatPinchConfig(
                lift_height_m=self.place_cfg.carry_height_m,
                closed_gap_m=self.place_cfg.closed_gap_m,
                durations=self.place_cfg.pick_durations,
            ),
        )
        if self.place_cfg.level_wrist:
            self.grip_free = slice(0, 4)
            self.grip_level = True
        self._place_planned = False
        self.card_offset_m = np.zeros(3)

    def reset(self) -> None:
        super().reset()
        self._place_planned = False

    def _palm_rot(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a, b = self.kin.tips(q)
        return (a + b) / 2, b - a, self.kin.d.xmat[self.kin.palm].reshape(3, 3).copy()

    def _plan_place(self) -> None:
        c = self.place_cfg
        up = np.array([0.0, 0.0, 1.0])
        gap = self.mat_cfg.closed_gap_m
        grip = {"free": self.grip_free, "level": self.grip_level}
        arm = slice(0, 3)
        lifted = self.keyframes[-1]
        mid, _, rot = self._palm_rot(lifted)
        card = self.task.object_pose(self.sink._env.data)[:3]
        offset = rot.T @ (card - mid)
        self.card_offset_m = card - mid
        goal = np.array([*self.task.target_xy, self.task._rest_z + c.drop_height_m])

        lowered = lifted
        for _ in range(3):  # the hand turns as the arm moves; re-express the offset
            _, _, rot = self._palm_rot(lowered)
            place_mid = goal - rot @ offset
            lowered = self._solve(lifted, place_mid, gap, **grip)
        carry_mid = place_mid.copy()
        carry_mid[2] = mid[2]
        carry = self._solve(lifted, carry_mid, gap, **grip)
        lowered = self._solve(carry, place_mid, gap, **grip)
        _, tip_line, _ = self._palm_rot(lowered)
        line = np.array([tip_line[0], tip_line[1], 0.0])
        line /= max(float(np.linalg.norm(line)), 1e-6)
        opened = self._solve(lowered, place_mid, c.open_gap_m, line_dir=line)
        retreat = self._solve(opened, place_mid + up * c.retreat_height_m, c.open_gap_m, free=arm)

        self.keyframes += [carry, lowered, lowered, opened, retreat, retreat]
        self.segment_s += list(c.durations)
        self.plan_error_m = float(max(self._errs))
        self._place_planned = True

    def select_action(self, observation: dict[str, Any]) -> np.ndarray:
        if not self.keyframes:
            self.reset()
        if not self._place_planned and self._step * self.dt >= sum(self.segment_s):
            self._plan_place()
        return super().select_action(observation)


EXPERTS: dict[str, type[ScriptedPinchPolicy]] = {
    "pinch": ScriptedPinchPolicy,
    "hwatu": ScriptedHwatuPolicy,
    "hwatu_mat": ScriptedMatPinchPolicy,
    "hwatu_place": ScriptedMatPlacePolicy,
}


def make_expert(sink: Any) -> ScriptedPinchPolicy:
    """The scripted expert for the sink's task."""
    return EXPERTS[sink._env.task.name](sink)
