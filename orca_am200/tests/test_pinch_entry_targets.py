"""Entry-anchored geometric goals; these checks do not verify physical contact."""
import numpy as np
import pytest
import yaml

from test_pinch_retarget import (
    AdaptiveAnalyticalRetargeter, HAND_CONFIG, ROOT, URDF, _free_thumb_pose,
)

PROFILE = ROOT / 'configs/adaptive_analytical_orca_v1_pinch_surface.yaml'


@pytest.fixture
def robot():
    return AdaptiveAnalyticalRetargeter(
        hand_config_path=str(HAND_CONFIG), urdf_path=str(URDF), config_path=str(PROFILE))


def input_pose(r, gap, pair=0):
    kp = _free_thumb_pose(r, 53)
    kp[8 + 4*pair] = kp[4] + [gap/100, 0, 0]
    return kp


def prepare(r, kp, q=None):
    return r._prepare_pinch_targets(
        kp, r._neutral_qpos_phys if q is None else q,
        r._compute_pinch_alpha(kp), r._compute_pinch_vectors(kp))


@pytest.mark.parametrize('pair', range(4))
def test_approach_and_release_share_one_robot_entry_and_preserve_contact_endpoint(robot, pair):
    r = robot
    d1, d2 = r._pinch_d1[pair], r._pinch_d2[pair]
    gaps = np.linspace(d2-1e-6, d1, 21)
    inward = [prepare(r, input_pose(r, gap, pair))[1][pair] for gap in gaps]
    entry = r._pinch_entry_clearance_cm[pair]
    endpoint = (r._surface_contacts.biases*100-r._surface_contact_closure_margin_cm)[pair]
    assert inward[0] == pytest.approx(entry, abs=1e-10)
    assert inward[-1] == pytest.approx(endpoint)
    assert np.all(np.diff(inward) <= 0)
    vectors, _ = prepare(r, input_pose(r, d1, pair))
    np.testing.assert_allclose(vectors[pair], 0, atol=1e-12)
    outward = [prepare(r, input_pose(r, gap, pair))[1][pair] for gap in gaps[::-1]]
    np.testing.assert_allclose(outward, inward[::-1], atol=1e-12)
    prepare(r, input_pose(r, d2, pair))
    assert np.isnan(r._pinch_entry_clearance_cm[pair])


def test_already_close_robot_is_not_given_the_old_outward_approach_goal(robot):
    r = robot
    # A synthetic FK probe based on a previously commanded posture; this is
    # not an independent measurement of real skin clearance.
    q = r._neutral_qpos_phys.copy()
    for joint, degrees in dict(thumb_mcp=-6.55, thumb_abd=17.22, thumb_pip=60.44,
                              thumb_dip=21.42, index_abd=20.10, index_mcp=75.19,
                              index_pip=40.73).items():
        q[r._finger_joint_ids.index(joint)] = np.deg2rad(degrees)
    kp = input_pose(r, 4)
    old = r._surface_contact_targets_cm(r._compute_pinch_vectors(kp), r._compute_pinch_alpha(kp))[0]
    vectors, goals = prepare(r, kp, q)
    entry = r._pinch_entry_clearance_cm[0]
    assert old > entry + 1  # Prior surface term asked for >10 mm of reopening.
    assert goals[0] < entry
    assert np.linalg.norm(vectors[0]) < np.linalg.norm(r._pinch_entry_vectors_cm[0])


def test_holding_input_does_not_ratchet_goals_as_robot_moves(robot):
    r = robot; kp = input_pose(r, 4)
    first_q = r._neutral_qpos_phys.copy()
    first_vectors, first_surface = prepare(r, kp, first_q)
    anchor = r._pinch_entry_vectors_cm.copy()
    for _ in range(12):
        r._solve(kp)
        np.testing.assert_allclose(r._last_solve_pinch_vectors, first_vectors, atol=1e-12)
        np.testing.assert_allclose(r._last_solve_surface_targets_cm, first_surface, atol=1e-12)
        np.testing.assert_array_equal(r._pinch_entry_vectors_cm, anchor)


def test_pairs_enter_independently_and_release_can_capture_a_new_pose(robot):
    r = robot
    kp = input_pose(r, 4); prepare(r, kp)
    index_anchor = r._pinch_entry_vectors_cm[0].copy()
    q = r._neutral_qpos_phys + .05
    kp[12] = kp[4] + [.04, 0, 0]
    prepare(r, kp, q)
    np.testing.assert_array_equal(r._pinch_entry_vectors_cm[0], index_anchor)
    assert np.isfinite(r._pinch_entry_vectors_cm[1]).all()
    kp[8] = kp[4] + [.08, 0, 0]; prepare(r, kp, q)
    assert np.isnan(r._pinch_entry_clearance_cm[0])
    kp[8] = kp[4] + [.04, 0, 0]; prepare(r, kp, q)
    assert not np.allclose(r._pinch_entry_vectors_cm[0], index_anchor)


def test_zero_time_and_diagnostics_cannot_advance_or_recompute_the_frozen_goal(robot):
    r = robot; kp = input_pose(r, 4)
    r._solve(kp, dt_s=0)
    assert np.isnan(r._pinch_entry_vectors_cm).all()
    assert r._last_solve_surface_targets_cm is None
    q = r._solve(kp)
    goals = r._last_solve_surface_targets_cm.copy()
    entry = r._pinch_entry_vectors_cm.copy()
    r._solve(input_pose(r, 0), dt_s=0)
    command = dict(zip(r._finger_joint_ids, np.rad2deg(q)))
    final = dict(command, thumb_dip=100)
    trace = r.command_diagnostics(command, final, 1234)
    np.testing.assert_allclose(trace['target_clearance_mm'], goals*10, atol=.001)
    np.testing.assert_array_equal(r._last_solve_surface_targets_cm, goals)
    np.testing.assert_array_equal(r._pinch_entry_vectors_cm, entry)


def test_reset_and_fold_mode_clear_entries_and_disable_folded_pairs(robot):
    r = robot
    r._solve(input_pose(r, 4, 1))
    assert np.isfinite(r._pinch_entry_clearance_cm[1])
    r.reset()
    assert np.isnan(r._pinch_entry_vectors_cm).all()
    assert r._last_solve_surface_targets_cm is None
    r._solve(input_pose(r, 4, 1))
    fixed = {j: float(np.rad2deg(r._neutral_qpos_phys[i]))
             for i, j in enumerate(r._finger_joint_ids) if j.startswith('middle_')}
    r.set_fixed_hand_joints(fixed)
    r._solve(input_pose(r, 0, 1))
    assert np.isnan(r._pinch_entry_clearance_cm[1])
    for j, value in fixed.items():
        assert np.rad2deg(r._last_qpos_phys[r._finger_joint_ids.index(j)]) == pytest.approx(value)


def test_frozen_entry_objective_gradient_matches_finite_difference(robot):
    r = robot; kp = input_pose(r, 4)
    vectors, goals = prepare(r, kp)
    args = (r._compute_tip_vectors(kp), r._compute_tip_dirs(kp), r._compute_full_hand_vectors(kp),
            r._compute_pinch_alpha(kp), r._neutral_qpos_phys, vectors,
            r._thumb_ip_target_bend(kp), None, goals)
    q = r._neutral_qpos_phys + .1
    _, analytic = r._loss_and_grad(q, *args)
    eps = 1e-6
    numeric = np.array([(r._loss_and_grad(q+d, *args)[0]-r._loss_and_grad(q-d, *args)[0])/(2*eps)
                        for d in np.eye(16)*eps])
    np.testing.assert_allclose(analytic, numeric, rtol=2e-4, atol=1e-5)


def test_direct_thumb_constraint_has_no_finite_jump_at_approach_onset(robot):
    r = robot
    seed = r._neutral_qpos_phys.copy()
    results = []
    for gap in (5, 5-1e-4, 5-1e-5):
        r.reset(); r._last_qpos_phys = seed.copy()
        q = r._solve(input_pose(r, gap))
        assert np.max(abs(q-seed)) <= np.deg2rad(2)+1e-8
        results.append(q[r._thumb_ip_index])
    assert results[0] == pytest.approx(seed[r._thumb_ip_index]+np.deg2rad(2), abs=1e-10)
    np.testing.assert_allclose(results[1:], results[0], atol=1e-7, rtol=0)


@pytest.mark.parametrize('override', [dict(pinch_target_mode='typo'),
    dict(surface_contact_calibration=None, surface_contact_closure_margin_mm=0),
    dict(pinch_max_joint_velocity_deg_s=0), dict(w_pinch=0)])
def test_invalid_entry_configuration_is_rejected(tmp_path, override):
    cfg = yaml.safe_load(PROFILE.read_text())
    cfg['retarget']['surface_contact_calibration'] = str(
        (PROFILE.parent/cfg['retarget']['surface_contact_calibration']).resolve())
    cfg['retarget'].update(override)
    path = tmp_path/'entry.yaml'; path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='pinch_target_mode|entry_anchored'):
        AdaptiveAnalyticalRetargeter(hand_config_path=str(HAND_CONFIG), urdf_path=str(URDF),
                                     config_path=str(path))
