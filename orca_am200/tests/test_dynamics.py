"""Arm command delay and domain randomization in the combined env."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from orca_teleop.am200.dynamics import (
    CommandDelay,
    RandomizationConfig,
    SimDynamics,
    write_arm_delays,
)
from orca_teleop.am200.sim_env import AmArm200OrcaRightEnv

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "robot_model" / "am_arm200_orca_right.xml"


def make_env(delay_s: float = 0.0, randomize: bool = False) -> AmArm200OrcaRightEnv:
    dyn = SimDynamics(
        arm_command_delay_s={"shoulder_pan": delay_s, "shoulder_lift": delay_s, "elbow_flex": delay_s},
        randomization=RandomizationConfig(enabled=randomize),
    )
    env = AmArm200OrcaRightEnv(model_path=MODEL_PATH, dynamics=dyn)
    env.reset(seed=0)
    return env


def pan_after_step(env: AmArm200OrcaRightEnv, target: float) -> float:
    hold = env.data.ctrl.copy()
    env.step(hold.astype(np.float32))
    cmd = hold.copy()
    cmd[0] = target
    env.step(cmd.astype(np.float32))
    return float(env.data.qpos[0])


def test_command_delay_lookup():
    d = CommandDelay(np.array([0.0, 0.02]))
    d.push(0.0, np.array([1.0, 1.0]))
    d.push(0.1, np.array([2.0, 2.0]))
    np.testing.assert_allclose(d.at(0.1), [2.0, 1.0])
    np.testing.assert_allclose(d.at(0.12), [2.0, 2.0])


def test_zero_delay_matches_plain_step():
    a, b = make_env(0.0), make_env(0.0)
    assert not a._delay.active
    assert pan_after_step(a, 0.3) == pytest.approx(pan_after_step(b, 0.3))


def test_arm_delay_makes_the_arm_lag():
    fast = pan_after_step(make_env(0.0), 0.3)
    slow = pan_after_step(make_env(0.02), 0.3)
    assert 0.0 < slow < fast


def test_randomization_is_seeded_and_restores_nominal():
    env = make_env(randomize=True)
    nominal_mass = env._randomizer._nom["mass"].copy()
    env.reset(seed=1)
    s1, m1 = dict(env.dynamics_sample), env.model.body_mass.copy()
    env.reset(seed=1)
    np.testing.assert_allclose(env.model.body_mass, m1)
    assert env.dynamics_sample == s1
    env.reset(seed=2)
    assert not np.allclose(env.model.body_mass, m1)

    env.dynamics.randomization.enabled = False
    env.reset(seed=3)
    np.testing.assert_allclose(env.model.body_mass, nominal_mass)
    assert env.dynamics_sample == {}


def test_write_arm_delays_keeps_other_keys(tmp_path):
    path = tmp_path / "sim_dynamics.yaml"
    path.write_text("randomization:\n  enabled: true\n", encoding="utf-8")
    write_arm_delays(path, {"elbow_flex": 0.016})
    dyn = SimDynamics.load(path)
    assert dyn.randomization.enabled
    assert dyn.arm_command_delay_s == {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.016}


def test_teleop_servo_damping_reduces_measured_arm_overshoot_and_hold_oscillation():
    from orca_teleop.am200.canonical import CanonicalJointMap
    from orca_teleop.am200.sim_sink import configure_arm_servos

    traces = []
    for damped in (False, True):
        env = make_env()
        try:
            mapping = CanonicalJointMap.load(MODEL_PATH.parent / "joint_map.json", env.model)
            hand_bias = env.model.actuator_biasprm[mapping.actuator_index[mapping.hand_slice]].copy()
            if damped:
                configure_arm_servos(env.model, env.data, mapping, 1.0)
                np.testing.assert_array_equal(
                    env.model.actuator_biasprm[mapping.actuator_index[mapping.hand_slice]], hand_bias
                )
            target = mapping.from_qpos(env.data.qpos)
            target[:3] = [0.15, -0.15, 0.15]
            values = []
            for _ in range(120):
                env.step(mapping.to_ctrl(target))
                values.append(mapping.from_qpos(env.data.qpos)[:3])
            traces.append(np.array(values))
        finally:
            env.close()
    assert traces[0][:, 0].max() - 0.15 > 0.08
    assert traces[1][:, 0].max() - 0.15 < 0.02
    assert np.max(np.ptp(traces[1][-30:], axis=0)) < 0.1 * np.max(np.ptp(traces[0][-30:], axis=0))
