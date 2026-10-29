"""Continuous surface-clearance candidate: geometry, gradients and commands.

Synthetic fixtures verify the algorithm, not physical contacts or grasp force.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from orca_teleop.retargeting.adaptive_analytical import AdaptiveAnalyticalRetargeter
from orca_teleop.retargeting.contact_calibration import FINGERS, fingerprint
from orca_teleop.retargeting.surface_contacts import (
    CalibratedSurfaceContacts, SkinContactGeometry, fit_clearance_biases,
)

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'configs/adaptive_analytical_orca_v1_pinch_refined.yaml'


def build(config=BASE):
    return AdaptiveAnalyticalRetargeter(
        hand_config_path=str(ROOT/'configs/orcahand-right-v1-feetech/config.yaml'),
        urdf_path=str(ROOT/'orcahand_description/v1/models/urdf/orcahand_right.urdf'),
        config_path=str(config))


def measurements():
    return {f: (np.array([-.001, 0, 0, .001, -.0005, .0005]) + .001*n).tolist()
            for n, f in enumerate(FINGERS[1:])}


@pytest.fixture
def calibrated(tmp_path):
    r = build()
    geometry = SkinContactGeometry(r)
    profile = fit_clearance_biases(measurements())
    profile.update(fingerprint=fingerprint(r), mesh_sha256=geometry.mesh_hashes,
                   surface_frames=geometry.frames,
                   measured_model_clearances_m=measurements())
    path = tmp_path/'surface.json'
    path.write_text(json.dumps(profile))
    cfg = r.config
    cfg['retarget']['surface_contact_calibration'] = path.name
    cfg['retarget']['surface_contact_closure_margin_mm'] = dict(index=8, middle=8, ring=10, pinky=10)
    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump(cfg))
    return build(config), path, config


def keypoints(finger=None, gap_cm=0):
    kp = np.zeros((21, 3))
    kp[1:5] = [[.02, 0, .03], [.02, 0, .06], [.01, 0, .09], [0, 0, .12]]
    for n in range(4):
        x = .07*(n+1)
        kp[5+4*n:9+4*n] = [[x, .02, .04], [x, .02, .07], [x, .02, .10], [x, .02, .13]]
    if finger:
        kp[8+4*FINGERS[1:].index(finger)] = kp[4] + [gap_cm/100, 0, 0]
    return kp


def test_bias_fit_uses_only_training_and_preserves_signed_distance_meaning():
    values = measurements()
    fit = fit_clearance_biases(values)
    assert fit['status'] == 'candidate' and not fit['powered_contact_verified']
    np.testing.assert_allclose(list(fit['clearance_bias_m'].values()), [0, .001, .002, .003])
    values['ring'][-1] += .001
    changed = fit_clearance_biases(values)
    assert fit['clearance_bias_m'] == changed['clearance_bias_m']
    assert fit['validation_signed_errors_m'] != changed['validation_signed_errors_m']


@pytest.mark.parametrize('kind', ['missing', 'nonfinite', 'excessive', 'validation', 'worse'])
def test_inconsistent_contact_data_is_rejected(kind):
    values = measurements()
    if kind == 'missing': values.pop('pinky')
    if kind == 'nonfinite': values['ring'][0] = float('nan')
    if kind == 'excessive': values['ring'] = [.02]*6
    if kind == 'validation': values['ring'][-1] = .02
    if kind == 'worse': values['index'] = [.004]*4 + [0, 0]
    with pytest.raises(ValueError):
        fit_clearance_biases(values)


def test_skin_distance_gradients_for_separation_and_penetration():
    r = build(); geometry = SkinContactGeometry(r)
    rng = np.random.default_rng(42)
    signs = set()
    poses = [r._lower_phys + rng.uniform(.15, .85, 16)*(r._upper_phys-r._lower_phys)
             for _ in range(15)]
    # Add an overlapping hull configuration explicitly: randomly sampled
    # interior-ROM poses rarely bring these small distal skins together.
    overlap = r._neutral_qpos_phys.copy()
    values = dict(thumb_mcp=35, thumb_abd=20, thumb_pip=8, thumb_dip=30,
                  middle_abd=-25, middle_mcp=82, middle_pip=0)
    for joint, degrees in values.items():
        overlap[r._finger_joint_ids.index(joint)] = np.deg2rad(degrees)
    poses.append(overlap)
    for q in poses:
        def evaluate(value):
            r._robot.compute_positions_and_jacobians(r._full_model_qpos(value), r._computed_frame_indices)
            return geometry.distances()
        d, analytic = evaluate(q)
        signs.update(np.sign(d).tolist())
        eps = 1e-6
        numeric = np.array([(evaluate(q+delta)[0]-evaluate(q-delta)[0])/(2*eps)
                            for delta in np.eye(16)*eps]).T
        np.testing.assert_allclose(analytic, numeric, rtol=2e-3, atol=2e-6)
    assert -1 in signs and 1 in signs


@pytest.mark.parametrize('gap', [0, 3.5, 5.1])
def test_entire_optimization_gradient_with_contact_blend(calibrated, gap):
    r, _, _ = calibrated; kp = keypoints('ring', gap)
    args = (r._compute_tip_vectors(kp), r._compute_tip_dirs(kp), r._compute_full_hand_vectors(kp),
            r._compute_pinch_alpha(kp), r._neutral_qpos_phys,
            r._compute_pinch_vectors(kp), r._human_thumb_ip_bend(kp))
    q = r._neutral_qpos_phys + .1
    _, analytic = r._loss_and_grad(q, *args)
    eps = 1e-6
    numeric = np.array([(r._loss_and_grad(q+d, *args)[0]-r._loss_and_grad(q-d, *args)[0])/(2*eps)
                        for d in np.eye(16)*eps])
    np.testing.assert_allclose(analytic, numeric, rtol=2e-4, atol=1e-5)


def test_open_hand_is_numerically_identical_to_original(calibrated):
    r, _, _ = calibrated; base = build()
    for _ in range(5):
        np.testing.assert_allclose(r._solve(keypoints()), base._solve(keypoints()), atol=1e-10)


def test_closure_margin_moves_all_targets_inward_and_fades_out(calibrated):
    r, _, _ = calibrated
    for finger in FINGERS[1:]:
        i = FINGERS[1:].index(finger)
        changes = []
        for gap in np.linspace(1, 6, 51):
            kp = keypoints(finger, gap)
            vectors, alpha = r._compute_pinch_vectors(kp), r._compute_pinch_alpha(kp)
            goal = r._surface_contact_targets_cm(vectors, alpha)[i]
            baseline = r._surface_contacts.biases[i]*100 + np.linalg.norm(vectors[i])
            changes.append((baseline-goal)*10)
        assert changes[0] == pytest.approx([8, 8, 10, 10][i])
        assert changes[-1] == pytest.approx(0.0)
        assert np.all(np.diff(changes) <= 1e-12)


@pytest.mark.parametrize('margin', [-1, 10.1, float('inf'), float('nan')])
def test_invalid_closure_margin_is_rejected(calibrated, margin):
    _, _, config = calibrated
    cfg = yaml.safe_load(config.read_text())
    cfg['retarget']['surface_contact_closure_margin_mm'] = margin
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='surface_contact_closure_margin_mm'):
        build(config)


@pytest.mark.parametrize('margin', [dict(index=8, middle=8, ring=10),
    dict(index=8, middle=8, ring=10, pinky=10, thumb=10),
    dict(index=8, middle=8, ring=10, pinky=11),
    dict(index=8, middle=8, ring=float('nan'), pinky=10)])
def test_invalid_pair_margins_are_rejected(calibrated, margin):
    _, _, config = calibrated
    cfg = yaml.safe_load(config.read_text())
    cfg['retarget']['surface_contact_closure_margin_mm'] = margin
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='surface_contact_closure_margin_mm'):
        build(config)


def test_pair_tuning_preserves_index_middle_targets_and_scalar_compatibility(calibrated):
    tuned, _, config = calibrated
    cfg = yaml.safe_load(config.read_text())
    cfg['retarget']['surface_contact_closure_margin_mm'] = 8.0
    config.write_text(yaml.safe_dump(cfg))
    previous = build(config)
    for gap in (0, 3.5, 6):
        kp = keypoints('index', gap)
        # Activate all four pairs at the same distance to inspect all goals.
        kp[[8, 12, 16, 20]] = kp[8]
        args = (tuned._compute_pinch_vectors(kp), tuned._compute_pinch_alpha(kp))
        before = previous._surface_contact_targets_cm(*args)
        after = tuned._surface_contact_targets_cm(*args)
        np.testing.assert_allclose(after[:2], before[:2], atol=1e-12)
        assert np.all(after[2:] <= before[2:])
        if gap == 0:
            np.testing.assert_allclose(before[2:]-after[2:], .2, atol=1e-12)


def test_disabled_pair_has_no_surface_objective(calibrated):
    r, _, _ = calibrated; base = build()
    r._pinch_pair_weights[2] = base._pinch_pair_weights[2] = 0
    kp = keypoints('ring')
    for _ in range(5):
        np.testing.assert_allclose(r._solve(kp), base._solve(kp), atol=1e-10)


def test_closed_pinch_can_still_change_its_joint_posture(calibrated):
    r, _, _ = calibrated; kp = keypoints('middle')
    for _ in range(40):
        first = r._solve(kp)
    # Keep the thumb-tip/middle-tip pair closed and change the middle chain.
    kp[10:12] += [.02, -.025, -.01]
    for _ in range(40):
        second = r._solve(kp)
    pair = [i for i, j in enumerate(r._finger_joint_ids) if j.startswith(('thumb_', 'middle_'))]
    assert r._last_taught_target is None
    assert np.max(abs(second[pair]-first[pair])) > np.deg2rad(2)


@pytest.mark.parametrize('finger', FINGERS[1:])
def test_pairs_remain_continuous_and_bounded_without_a_recorded_pose(calibrated, finger):
    r, _, _ = calibrated
    assert r._taught_contacts is None
    previous = r._neutral_qpos_phys.copy()
    for kp in (keypoints(finger), keypoints(finger, 3.5), keypoints()):
        for _ in range(12):
            q = r._solve(kp)
            assert np.max(abs(q-previous)) <= np.deg2rad(2)+1e-8
            assert np.all(q >= r._lower_phys) and np.all(q <= r._upper_phys)
            assert r._last_taught_target is None
            previous = q.copy()


@pytest.mark.parametrize('kind', ['calibration', 'mesh', 'frames', 'bias', 'residual'])
def test_stale_or_tampered_profiles_are_rejected(calibrated, kind):
    r, path, _ = calibrated
    profile = json.loads(path.read_text())
    if kind == 'calibration': profile['fingerprint'] = 'stale'
    if kind == 'mesh': profile['mesh_sha256'] = {}
    if kind == 'frames': profile['surface_frames'] = profile['surface_frames'][::-1]
    if kind == 'bias': profile['clearance_bias_m']['ring'] += .001
    if kind == 'residual': profile['measured_model_clearances_m']['ring'][-1] = .03
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError):
        CalibratedSurfaceContacts(path, r)


def test_combining_pose_or_point_and_surface_profiles_is_rejected(calibrated):
    _, _, config = calibrated
    cfg = yaml.safe_load(config.read_text())
    cfg['retarget']['contact_pose_library'] = 'unused.json'
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='not both'):
        build(config)
