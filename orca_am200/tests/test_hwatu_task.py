"""Hwatu card tasks: the card rests flat, the scripted slide-then-pinch expert
picks it off the hard block, and the direct pinch expert picks it off the mat."""

from __future__ import annotations

import numpy as np
import pytest

from orca_teleop.am200.tasks import AmArm200TaskEnv, make_task


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_card_spawns_flat_on_block_short_of_the_edge(seed):
    env = AmArm200TaskEnv(make_task("hwatu"))
    env.reset(seed=seed)
    task = env.task
    for _ in range(30):
        env.step(env.data.ctrl.copy())
    pose = task.object_pose(env.data)
    assert pose[2] == pytest.approx(task._rest_z, abs=5e-4)
    # still flat: local z stays vertical
    w, x, y, _ = pose[3:]
    assert 1.0 - 2.0 * (x * x + y * y) > 0.999
    far_end = pose[0] + task.card_half_extents[0]
    gap = task.block_edge_x - far_end
    lo, hi = task.edge_gap_range_m
    assert lo - 0.01 < gap < hi + 0.01


def _run_expert(task: str, policy_cls, seconds: float):
    from orca_teleop.am200 import Am200SimSink, RobotJointCommand

    sink = Am200SimSink(render_mode="rgb_array", task=task, randomize=False)
    sink.connect()
    try:
        policy = policy_cls(sink)
        sink.reset_scene(seed=0)
        policy.reset()
        start = sink._env.task.object_pose(sink._env.data)[:3]
        success = False
        for _ in range(int(seconds / policy.dt)):
            target = policy.select_action({})
            sink.dispatch_action(RobotJointCommand.now(tuple(sink.joint_ids), target))
            if sink.task_info["success"]:
                success = True
                break
        return policy, start, sink._env.task.object_pose(sink._env.data)[:3], success
    finally:
        sink.close()


def test_scripted_expert_slides_and_pinches_the_card():
    from orca_teleop.am200.scripted import ScriptedHwatuPolicy

    policy, start, _, success = _run_expert("hwatu", ScriptedHwatuPolicy, 16.0)
    assert policy._pinch_planned
    assert policy.overhang_at_pinch_m > 0.015, "slide did not push the card over the edge"
    assert success


def test_card_barely_sinks_into_the_mat():
    env = AmArm200TaskEnv(make_task("hwatu_mat"))
    env.reset(seed=0)
    for _ in range(30):
        env.step(env.data.ctrl.copy())
    assert env.task.object_pose(env.data)[2] == pytest.approx(env.task._rest_z, abs=3e-4)


def test_direct_pinch_lifts_the_card_off_the_mat_without_sliding():
    from orca_teleop.am200.scripted import ScriptedMatPinchPolicy

    _, start, end, success = _run_expert("hwatu_mat", ScriptedMatPinchPolicy, 10.0)
    assert success
    assert np.linalg.norm(end[:2] - start[:2]) < 0.02
