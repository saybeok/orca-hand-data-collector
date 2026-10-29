"""Hwatu pick-and-place on the blanket: the target marker, the success rule and
the scripted pick-carry-place expert."""

from __future__ import annotations

import numpy as np
import pytest

from orca_teleop.am200.tasks import AmArm200TaskEnv, make_task


def _mat_bounds(env: AmArm200TaskEnv) -> tuple[np.ndarray, np.ndarray]:
    g = env.model.geom("hwatu_mat")
    return g.pos[:2] - g.size[:2], g.pos[:2] + g.size[:2]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_target_spawns_on_the_blanket_away_from_the_card(seed):
    env = AmArm200TaskEnv(make_task("hwatu_place"))
    env.reset(seed=seed)
    task = env.task
    card = task.object_pose(env.data)[:2]
    lo, hi = task.target_dist_range_m
    assert lo <= np.linalg.norm(task.target_xy - card) <= hi

    marker = env.data.mocap_pos[task._target_mocap]
    np.testing.assert_allclose(marker[:2], task.target_xy)
    assert marker[2] == pytest.approx(env.model.geom("hwatu_mat").pos[2] + env.model.geom("hwatu_mat").size[2])
    mat_lo, mat_hi = _mat_bounds(env)
    margin = task.card_half_extents[0]
    for xy in (card, task.target_xy):
        assert np.all(xy > mat_lo + margin) and np.all(xy < mat_hi - margin)


def test_pick_only_task_parks_the_target():
    env = AmArm200TaskEnv(make_task("hwatu_mat"))
    env.reset(seed=0)
    assert env.data.mocap_pos[env.task._target_mocap][2] < -0.5


def _card_at_target(env: AmArm200TaskEnv) -> None:
    task = env.task
    adr = task._obj_qadr
    env.data.qpos[adr:adr + 2] = task.target_xy
    env.data.qvel[task._obj_dadr:task._obj_dadr + 6] = 0.0


def test_card_on_the_target_counts_only_if_it_was_put_down():
    env = AmArm200TaskEnv(make_task("hwatu_place"))
    env.reset(seed=0)
    task = env.task
    steps = int(2 * task.hold_s / env.control_dt) + 2

    # Picked, but let go in the air: landing on the target is luck, not a place.
    task._picked = True
    _card_at_target(env)
    for _ in range(steps):
        _, _, _, _, info = env.step(env.data.ctrl.copy())
    assert info["place_error_m"] < task.place_tol_m and not info["delivered"]
    assert not info["success"]

    task._delivered = True
    for _ in range(steps):
        _, _, terminated, _, info = env.step(env.data.ctrl.copy())
        if terminated:
            break
    assert info["success"]


def test_every_task_has_a_scripted_expert():
    from orca_teleop.am200.scripted import EXPERTS
    from orca_teleop.am200.tasks import TASKS

    assert set(EXPERTS) == set(TASKS)


def test_scripted_expert_carries_the_card_to_the_target():
    from orca_teleop.am200 import Am200SimSink, RobotJointCommand
    from orca_teleop.am200.scripted import ScriptedMatPlacePolicy

    sink = Am200SimSink(render_mode="rgb_array", task="hwatu_place", randomize=False)
    sink.connect()
    try:
        policy = ScriptedMatPlacePolicy(sink)
        sink.reset_scene(seed=0)
        policy.reset()
        info = {}
        for _ in range(int(14.0 / policy.dt)):
            sink.dispatch_action(RobotJointCommand.now(tuple(sink.joint_ids), policy.select_action({})))
            info = sink.task_info
            if info["success"] or info["dropped"]:
                break
        assert policy._place_planned
        assert info["delivered"], "card was not held until it was over the target"
        assert info["success"]
        assert info["place_error_m"] < 0.01
    finally:
        sink.close()
