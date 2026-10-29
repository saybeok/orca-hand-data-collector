"""Sim-only dynamics for the combined AM-ARM200 + ORCA hand env.

Two things MJCF attributes cannot express:

* **Arm command delay** — the Feetech bus + servo firmware react to a new goal
  some milliseconds late. ``scripts/arm_sysid_fit.py --apply`` writes the
  identified per-joint delay here.
* **Domain randomization** — per-episode scaling of actuator gains, joint
  damping/friction, contact friction and body masses around the nominal model,
  so policies do not overfit to parameters that are only estimates (the hand's
  tendon drive in particular is modelled as direct joint servos).

Both live in ``sim_dynamics.yaml`` next to the model; a missing file means no
delay and no randomization.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

from orca_teleop.am200.sysid import ARM_ACTUATORS, ARM_JOINTS

DYNAMICS_FILE = "sim_dynamics.yaml"

Range = tuple[float, float]


@dataclass
class RandomizationConfig:
    enabled: bool = False
    arm_kp_scale: Range = (0.85, 1.15)
    arm_damping_scale: Range = (0.7, 1.3)
    arm_armature_scale: Range = (0.7, 1.3)
    arm_frictionloss_scale: Range = (0.5, 1.5)
    arm_delay_add_s: Range = (0.0, 0.01)
    hand_kp_scale: Range = (0.7, 1.3)
    hand_forcerange_scale: Range = (0.7, 1.3)
    hand_damping_scale: Range = (0.7, 1.3)
    friction_scale: Range = (0.7, 1.3)
    mass_scale: Range = (0.9, 1.1)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> RandomizationConfig:
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown randomization keys: {sorted(unknown)}")
        return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()})


@dataclass
class SimDynamics:
    arm_command_delay_s: dict[str, float] = field(
        default_factory=lambda: {j: 0.0 for j in ARM_JOINTS}
    )
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)

    @classmethod
    def load(cls, path: str | Path | None) -> SimDynamics:
        if path is None or not Path(path).is_file():
            return cls()
        with Path(path).open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        delay = {j: 0.0 for j in ARM_JOINTS}
        delay.update({k: float(v) for k, v in (raw.get("arm_command_delay_s") or {}).items()})
        if set(delay) - set(ARM_JOINTS):
            raise ValueError(f"unknown joints in arm_command_delay_s: {set(delay) - set(ARM_JOINTS)}")
        return cls(delay, RandomizationConfig.from_dict(raw.get("randomization")))


def dynamics_path_for(model_path: str | Path) -> Path:
    return Path(model_path).parent / DYNAMICS_FILE


def write_arm_delays(path: str | Path, delays: dict[str, float]) -> None:
    """Update only ``arm_command_delay_s`` in the yaml, keeping everything else."""
    path = Path(path)
    raw: dict[str, Any] = {}
    if path.is_file():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cur = dict(raw.get("arm_command_delay_s") or {})
    cur.update({k: round(float(v), 4) for k, v in delays.items()})
    raw["arm_command_delay_s"] = cur
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


class CommandDelay:
    """Per-actuator transport delay applied at physics-step resolution."""

    def __init__(self, delays_s: np.ndarray) -> None:
        self.delays = np.asarray(delays_s, dtype=np.float64)
        self._t: list[float] = []
        self._u: list[np.ndarray] = []

    @property
    def active(self) -> bool:
        return bool(np.any(self.delays > 0))

    def reset(self) -> None:
        self._t.clear()
        self._u.clear()

    def push(self, t: float, ctrl: np.ndarray) -> None:
        self._t.append(float(t))
        self._u.append(np.asarray(ctrl, dtype=np.float64).copy())
        # Keep one command older than the longest delay window.
        cutoff = t - float(self.delays.max()) - 1e-9
        while len(self._t) > 1 and self._t[1] <= cutoff:
            self._t.pop(0)
            self._u.pop(0)

    def at(self, t: float) -> np.ndarray:
        """Command each actuator sees at sim time ``t``. Before the first
        command has arrived an actuator holds that first command."""
        times = np.asarray(self._t)
        idx = np.searchsorted(times, t - self.delays + 1e-9, side="right") - 1
        idx = np.clip(idx, 0, len(times) - 1)
        u = np.stack(self._u)
        return u[idx, np.arange(len(self.delays))]


class DynamicsRandomizer:
    """Restores nominal model parameters and applies a fresh random sample."""

    def __init__(self, model: mujoco.MjModel, dyn: SimDynamics) -> None:
        self.model = model
        self.dyn = dyn
        m = model

        def act_id(name: str) -> int:
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if i < 0:
                raise ValueError(f"model has no actuator {name!r}")
            return i

        self.arm_act = np.array([act_id(n) for n in ARM_ACTUATORS])
        self.hand_act = np.array([a for a in range(m.nu) if a not in set(self.arm_act)])
        self.arm_dof = np.array([m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in self.arm_act])
        self.hand_dof = np.array([m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in self.hand_act])
        self.nominal_delays = np.zeros(m.nu)
        for j, a in zip(ARM_JOINTS, self.arm_act):
            self.nominal_delays[a] = dyn.arm_command_delay_s.get(j, 0.0)

        self._nom = {
            "gain": m.actuator_gainprm[:, 0].copy(),
            "bias": m.actuator_biasprm[:, 1].copy(),
            "force": m.actuator_forcerange.copy(),
            "damping": m.dof_damping.copy(),
            "armature": m.dof_armature.copy(),
            "frictionloss": m.dof_frictionloss.copy(),
            "friction": m.geom_friction.copy(),
            "mass": m.body_mass.copy(),
            "inertia": m.body_inertia.copy(),
        }
        self.last_sample: dict[str, Any] = {}

    def restore(self) -> None:
        m, n = self.model, self._nom
        m.actuator_gainprm[:, 0] = n["gain"]
        m.actuator_biasprm[:, 1] = n["bias"]
        m.actuator_forcerange[:] = n["force"]
        m.dof_damping[:] = n["damping"]
        m.dof_armature[:] = n["armature"]
        m.dof_frictionloss[:] = n["frictionloss"]
        m.geom_friction[:] = n["friction"]
        m.body_mass[:] = n["mass"]
        m.body_inertia[:] = n["inertia"]

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Resample (or restore, if disabled) and return per-actuator delays."""
        self.restore()
        cfg = self.dyn.randomization
        if not cfg.enabled:
            self.last_sample = {}
            return self.nominal_delays.copy()

        m, n = self.model, self._nom

        def u(r: Range, size: int | None = None) -> np.ndarray:
            return rng.uniform(r[0], r[1], size=size)

        s: dict[str, Any] = {}
        arm_kp = u(cfg.arm_kp_scale, len(self.arm_act))
        hand_kp = u(cfg.hand_kp_scale, len(self.hand_act))
        for acts, scale in ((self.arm_act, arm_kp), (self.hand_act, hand_kp)):
            m.actuator_gainprm[acts, 0] = n["gain"][acts] * scale
            m.actuator_biasprm[acts, 1] = n["bias"][acts] * scale
        hand_force = u(cfg.hand_forcerange_scale, len(self.hand_act))
        m.actuator_forcerange[self.hand_act] = n["force"][self.hand_act] * hand_force[:, None]

        arm_damp = u(cfg.arm_damping_scale, len(self.arm_dof))
        m.dof_damping[self.arm_dof] = n["damping"][self.arm_dof] * arm_damp
        m.dof_armature[self.arm_dof] = n["armature"][self.arm_dof] * u(
            cfg.arm_armature_scale, len(self.arm_dof)
        )
        m.dof_frictionloss[self.arm_dof] = n["frictionloss"][self.arm_dof] * u(
            cfg.arm_frictionloss_scale, len(self.arm_dof)
        )
        hand_damp = u(cfg.hand_damping_scale, len(self.hand_dof))
        m.dof_damping[self.hand_dof] = n["damping"][self.hand_dof] * hand_damp

        fric = float(u(cfg.friction_scale))
        m.geom_friction[:, 0] = n["friction"][:, 0] * fric

        mass = u(cfg.mass_scale, m.nbody)
        mass[0] = 1.0
        m.body_mass[:] = n["mass"] * mass
        m.body_inertia[:] = n["inertia"] * mass[:, None]

        delays = self.nominal_delays.copy()
        delays[self.arm_act] += u(cfg.arm_delay_add_s, len(self.arm_act))

        s.update(
            arm_kp_scale=arm_kp.tolist(),
            hand_kp_scale_mean=float(hand_kp.mean()),
            hand_forcerange_scale_mean=float(hand_force.mean()),
            arm_damping_scale=arm_damp.tolist(),
            friction_scale=fric,
            mass_scale_range=[float(mass[1:].min()), float(mass[1:].max())],
            arm_delay_s=delays[self.arm_act].tolist(),
        )
        self.last_sample = s
        return delays
