"""AM-ARM200 arm system identification: excitation schedule + MuJoCo replay.

Shared by ``scripts/arm_sysid_record.py`` (hardware) and
``scripts/arm_sysid_fit.py`` (fitting). All joint quantities are model radians
relative to the pose the arm had when :class:`AmArmFeetech` connected, which
the teleop stack already treats as model ``qpos = 0``.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
from mujoco import rollout

ARM_JOINTS: tuple[str, ...] = ("shoulder_pan", "shoulder_lift", "elbow_flex")
ARM_ACTUATORS: tuple[str, ...] = tuple(f"act_{j}" for j in ARM_JOINTS)


def default_model_path() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "orca_am200"
        / "robot_model"
        / "am_arm200_orca_right.xml"
    )


@dataclass
class JointParams:
    kp: float
    damping: float
    armature: float
    frictionloss: float
    forcerange: float
    delay_s: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


# ---------------------------------------------------------------- schedule


def build_schedule(
    joint_idx: int,
    amplitudes: list[float],
    direction: str = "both",
    hold_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-constant staircase targets for one joint: 0 -> a1 -> ... -> an -> ... -> 0.

    Every intermediate level is reached once from below and once from above,
    which lets static friction be separated from gravity in the current fit.
    ``direction`` is ``both`` / ``pos`` / ``neg`` so a joint can avoid the side
    where it would hit the desk. Returns ``(t_start, targets[N, 3])``.
    """
    signs = {"both": (1, -1), "pos": (1,), "neg": (-1,)}[direction]
    amps = sorted(abs(a) for a in amplitudes)
    levels = [0.0]
    for s in signs:
        levels += [s * a for a in amps] + [s * a for a in amps[-2::-1]] + [0.0]
    targets = np.zeros((len(levels), len(ARM_JOINTS)))
    targets[:, joint_idx] = levels
    t_start = np.arange(len(levels), dtype=np.float64) * hold_s
    return t_start, targets


# ------------------------------------------------------------------ model


class ArmReplay:
    """Replays recorded arm commands through the combined MuJoCo model.

    Batches of parameter sets run in parallel through ``mujoco.rollout``.
    """

    _STATE = mujoco.mjtState.mjSTATE_FULLPHYSICS

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path or default_model_path())
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.jnt = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        self.act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_ACTUATORS]
        if min(self.jnt + self.act) < 0:
            raise ValueError(f"{self.model_path} is missing arm joints/actuators")
        self.qadr = [int(m.jnt_qposadr[j]) for j in self.jnt]
        self.dadr = [int(m.jnt_dofadr[j]) for j in self.jnt]
        self.dt = float(m.opt.timestep)
        nthread = max(1, (os.cpu_count() or 2) - 1)
        self._datas = [mujoco.MjData(m) for _ in range(nthread)]
        self._models: list[mujoco.MjModel] = []

    def get_params(self, i: int) -> JointParams:
        m, a, d = self.model, self.act[i], self.dadr[i]
        return JointParams(
            kp=float(m.actuator_gainprm[a, 0]),
            damping=float(m.dof_damping[d]),
            armature=float(m.dof_armature[d]),
            frictionloss=float(m.dof_frictionloss[d]),
            forcerange=float(m.actuator_forcerange[a, 1]),
        )

    def _set_params(self, m: mujoco.MjModel, i: int, p: JointParams) -> None:
        a, d = self.act[i], self.dadr[i]
        m.actuator_gainprm[a, 0] = p.kp
        m.actuator_biasprm[a, 1] = -p.kp
        m.actuator_forcerange[a] = (-p.forcerange, p.forcerange)
        m.dof_damping[d] = p.damping
        m.dof_armature[d] = p.armature
        m.dof_frictionloss[d] = p.frictionloss

    def gravity_torque(self, q_arm: np.ndarray, home_qpos: np.ndarray) -> tuple[np.ndarray, int]:
        """Static joint torque the servos must supply to hold ``q_arm``, and the
        number of contacts the model has in that pose (non-zero means the
        gravity-only balance does not hold)."""
        d = self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[self.qadr] = home_qpos + q_arm
        mujoco.mj_forward(self.model, d)
        return d.qfrc_bias[self.dadr].copy(), int(d.ncon)

    def rollout(
        self,
        t_cmd: np.ndarray,
        cmd: np.ndarray,
        q0: np.ndarray,
        t_eval: np.ndarray,
        param_sets: list[list[JointParams]],
        home_qpos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate each parameter set from ``q0`` under piecewise-constant ``cmd``.

        ``cmd``/``q0``/outputs are home-relative radians. Returns
        ``(q[B, len(t_eval), 3], actuator_torque[B, len(t_eval), 3])``.
        """
        m0 = self.model
        nb = len(param_sets)
        while len(self._models) < nb:
            self._models.append(copy.copy(m0))
        models = self._models[:nb]
        for mb, ps in zip(models, param_sets):
            for i, p in enumerate(ps):
                self._set_params(mb, i, p)

        d = self.data
        mujoco.mj_resetData(m0, d)
        d.qpos[self.qadr] = home_qpos + q0
        mujoco.mj_forward(m0, d)
        state0 = np.empty(mujoco.mj_stateSize(m0, self._STATE))
        mujoco.mj_getState(m0, d, state0, self._STATE)

        n = int(np.ceil((t_eval[-1] - t_eval[0]) / self.dt)) + 2
        ts = t_eval[0] + np.arange(n) * self.dt
        ctrl = np.zeros((nb, n, m0.nu))
        for b, ps in enumerate(param_sets):
            for i, p in enumerate(ps):
                idx = np.searchsorted(t_cmd, ts - p.delay_s, side="right") - 1
                ctrl[b, :, self.act[i]] = home_qpos[i] + cmd[np.clip(idx, 0, len(t_cmd) - 1), i]

        state, _ = rollout.rollout(
            models, self._datas[: min(nb, len(self._datas))], state0, ctrl
        )
        # state[:, k] is the state after step k, i.e. at ts[k] + dt.
        q_full = state[:, :, 1 + np.asarray(self.qadr)]
        t_state = ts + self.dt

        q_out = np.empty((nb, len(t_eval), 3))
        tau_out = np.empty((nb, len(t_eval), 3))
        for b, ps in enumerate(param_sets):
            for i, p in enumerate(ps):
                q_out[b, :, i] = np.interp(t_eval, t_state, q_full[b, :, i])
                u = np.interp(t_eval, t_state, ctrl[b, :, self.act[i]])
                tau_out[b, :, i] = np.clip(
                    p.kp * (u - q_out[b, :, i]), -p.forcerange, p.forcerange
                )
        return q_out - home_qpos, tau_out

    def simulate(
        self,
        t_cmd: np.ndarray,
        cmd: np.ndarray,
        q0: np.ndarray,
        t_eval: np.ndarray,
        params: list[JointParams],
        home_qpos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        q, tau = self.rollout(t_cmd, cmd, q0, t_eval, [params], home_qpos)
        return q[0], tau[0]


# ------------------------------------------------------------- apply to MJCF


def apply_params_to_mjcf(path: str | Path, params: dict[str, JointParams]) -> None:
    """Write identified per-joint values onto the arm joint/actuator elements."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    def set_attr(tag: str, attr: str, value: str) -> str:
        pat = re.compile(rf'\s{attr}="[^"]*"')
        if pat.search(tag):
            return pat.sub(f' {attr}="{value}"', tag)
        return tag.replace(" />", f' {attr}="{value}" />', 1)

    for joint, p in params.items():
        j_pat = re.compile(rf'<joint class="am_sts3095_joint" name="{joint}"[^>]*/>')
        a_pat = re.compile(rf'<position class="am_sts3095_joint" name="act_{joint}"[^>]*/>')
        jm, am = j_pat.search(text), a_pat.search(text)
        if not jm or not am:
            raise ValueError(f"could not find joint/actuator for {joint} in {path}")
        tag = jm.group(0)
        for attr, v in (
            ("damping", p.damping),
            ("armature", p.armature),
            ("frictionloss", p.frictionloss),
        ):
            tag = set_attr(tag, attr, f"{v:.5g}")
        text = text.replace(jm.group(0), tag)
        tag = am.group(0)
        tag = set_attr(tag, "kp", f"{p.kp:.5g}")
        tag = set_attr(tag, "forcerange", f"{-p.forcerange:.4g} {p.forcerange:.4g}")
        text = text.replace(am.group(0), tag)
    path.write_text(text, encoding="utf-8")
