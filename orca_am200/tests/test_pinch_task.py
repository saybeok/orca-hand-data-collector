"""The pinch task's success detector fires on a real thumb-index pinch.

A policy's success rate is only meaningful if the detector can say yes, so this
poses the hand with thumb and index tips 18 mm apart, parks the cylinder
between them above the desk with gravity off, and requires success after the
hold time. It checks the detector, not whether the grasp is physically stable.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from scipy.optimize import minimize

from orca_teleop.am200.tasks import AmArm200TaskEnv, make_task

TIP_GAP_M = 0.018


@pytest.fixture(scope="module")
def env():
    env = AmArm200TaskEnv(make_task("pinch"))
    env.reset(seed=0)
    return env


def pinch_pose(env: AmArm200TaskEnv) -> tuple[np.ndarray, np.ndarray]:
    m, d, task = env.model, env.data, env.task
    joints = [
        j for j in range(m.njnt)
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        and any(k in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for k in ("thumb", "index"))
    ]
    qadr = np.array([m.jnt_qposadr[j] for j in joints])
    bounds = m.jnt_range[joints]
    index_tip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_index_ip")

    def tips(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d.qpos[qadr] = x
        mujoco.mj_kinematics(m, d)
        return task._tip_point(d, task._thumb), task._tip_point(d, index_tip)

    def cost(x: np.ndarray) -> float:
        a, b = tips(x)
        return (np.linalg.norm(a - b) - TIP_GAP_M) ** 2

    rng = np.random.default_rng(0)
    best = min(
        (
            minimize(cost, rng.uniform(bounds[:, 0], bounds[:, 1]), bounds=list(map(tuple, bounds)), method="L-BFGS-B")
            for _ in range(20)
        ),
        key=lambda r: r.fun,
    )
    return tips(best.x)


def test_detector_reports_success_for_a_held_pinch(env):
    m, d, task = env.model, env.data, env.task
    a, b = pinch_pose(env)
    assert np.linalg.norm(a - b) == pytest.approx(TIP_GAP_M, abs=1e-3)

    x = (a - b) / np.linalg.norm(a - b)
    z = np.cross(x, [0.0, 0.0, 1.0])
    z /= np.linalg.norm(z)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.stack([x, np.cross(z, x), z], axis=1).ravel())

    gravity = m.opt.gravity.copy()
    m.opt.gravity[:] = 0.0
    try:
        d.qpos[task._obj_qadr:task._obj_qadr + 3] = (a + b) / 2
        d.qpos[task._obj_qadr + 3:task._obj_qadr + 7] = quat
        state = None
        for _ in range(int(np.ceil(task.hold_s / env.control_dt)) + 2):
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            state = task.update(m, d, env.control_dt)
    finally:
        m.opt.gravity[:] = gravity

    assert state.lift_m >= task.lift_m
    assert state.thumb_contact and state.finger_contact and not state.palm_contact
    assert state.success


def test_scripted_expert_physically_solves_the_task():
    """With gravity on and nominal dynamics the scripted expert lifts the cylinder."""
    from orca_teleop.am200 import Am200SimSink, RobotJointCommand
    from orca_teleop.am200.scripted import ScriptedPinchPolicy

    sink = Am200SimSink(render_mode="rgb_array", task="pinch", randomize=False)
    sink.connect()
    try:
        policy = ScriptedPinchPolicy(sink)
        sink.reset_scene(seed=1000)
        policy.reset()
        assert policy.plan_error_m < 1e-3
        success = False
        for _ in range(int(sum(policy.segment_s) / policy.dt)):
            target = policy.select_action({})
            sink.dispatch_action(RobotJointCommand.now(tuple(sink.joint_ids), target))
            if sink.task_info["success"]:
                success = True
                break
        assert success
    finally:
        sink.close()
