"""An operator pinch must close the robot's thumb onto the fingertip.

Matching each fingertip to the palm separately leaves the tips centimetres
apart whenever the operator pinches somewhere the v1 thumb cannot follow; the
thumb-to-finger term is what closes the gap.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from orca_teleop.retargeting.adaptive_analytical import (
    MANO_OFFSET_CM,
    AdaptiveAnalyticalRetargeter,
)
from orca_teleop.retargeting.retargeter import TargetPose

ROOT = Path(__file__).resolve().parents[2]
HAND_CONFIG = ROOT / "configs" / "orcahand-right-v1-feetech" / "config.yaml"
RETARGET_CONFIG = ROOT / "configs" / "adaptive_analytical_orca_v1.yaml"
URDF = Path(
    os.environ.get(
        "ORCA_URDF_PATH",
        ROOT / "orcahand_description" / "v1" / "models" / "urdf" / "orcahand_right.urdf",
    )
)
PINCH_DEG = dict(
    thumb_mcp=50, thumb_abd=-8, thumb_pip=79, thumb_dip=112,
    index_abd=-37, index_mcp=88, index_pip=107,
    middle_abd=-33, middle_mcp=-15, middle_pip=11,
    ring_abd=-16, ring_mcp=-20, ring_pip=-5,
    pinky_abd=-25, pinky_mcp=-4, pinky_pip=9,
)


@pytest.fixture(scope="module")
def retargeter() -> AdaptiveAnalyticalRetargeter:
    if not URDF.is_file():
        pytest.skip(f"v1 URDF not found at {URDF}")
    return AdaptiveAnalyticalRetargeter(
        hand_config_path=str(HAND_CONFIG),
        urdf_path=str(URDF),
        config_path=str(RETARGET_CONFIG),
    )


def _positions(r, q, offsets=None):
    pos, _ = r._robot.compute_positions_and_jacobians(
        r._full_model_qpos(q),
        r._computed_frame_indices,
        r._frame_offsets if offsets is None else offsets,
    )
    return pos


def _tip_gap_cm(r, q) -> float:
    """Visible fingertip gap: URDF tip frames, not the extra pad offsets."""
    tips = _positions(r, q, r._contact_offsets)[r._tip_indices] * 100.0
    return float(np.linalg.norm(tips[0] - tips[1]))


def _operator_pinch(r, shift_m) -> np.ndarray:
    """Human-proportioned thumb-index pinch at a point the robot cannot match."""
    q = np.deg2rad([PINCH_DEG[j] for j in r._finger_joint_ids])
    pos = _positions(r, q)
    kp = np.zeros((21, 3))
    for f, (b, p, d, t) in enumerate(
        [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
    ):
        kp[b] = pos[r._base_indices[f]] - MANO_OFFSET_CM / 100.0
        kp[p], kp[d], kp[t] = pos[r._pip_indices[f]], pos[r._dip_indices[f]], pos[r._tip_indices[f]]
    palm = 0.5 * (kp[1] + kp[17])
    kp[2:5] = palm + 1.15 * (kp[2:5] - palm)
    kp[6:9] = palm + 0.9 * (kp[6:9] - palm)
    kp[4] = kp[8] = 0.5 * (kp[4] + kp[8])
    shift = np.asarray(shift_m)
    kp[[4, 8]] += shift
    kp[[3, 7]] += 0.5 * shift
    return kp


def _solve(r, kp, w_pinch: float) -> np.ndarray:
    r._w_pinch = w_pinch
    r._last_qpos_phys = None
    for _ in range(15):
        q = r._solve(kp)
    return q


def test_pinch_gradient_matches_finite_difference(retargeter):
    r = retargeter
    r._w_pinch = 8.0
    kp = _operator_pinch(r, [0.03, 0.0, 0.0])
    args = (
        r._compute_tip_vectors(kp),
        r._compute_tip_dirs(kp),
        r._compute_full_hand_vectors(kp),
        r._compute_pinch_alpha(kp),
        None,
        r._compute_pinch_vectors(kp),
    )
    q = r._neutral_qpos_phys + 0.1
    _, grad = r._loss_and_grad(q, *args)
    eps = 1e-6
    numeric = np.array(
        [
            (r._loss_and_grad(q + e, *args)[0] - r._loss_and_grad(q - e, *args)[0]) / (2 * eps)
            for e in np.eye(len(q)) * eps
        ]
    )
    np.testing.assert_allclose(grad, numeric, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("shift", [[0.03, 0, 0], [-0.03, 0, 0], [0, 0, 0.03], [0, 0, -0.03]])
def test_operator_pinch_closes_robot_thumb_onto_index(retargeter, shift):
    kp = _operator_pinch(retargeter, shift)
    without = _tip_gap_cm(retargeter, _solve(retargeter, kp, 0.0))
    with_term = _tip_gap_cm(retargeter, _solve(retargeter, kp, 8.0))
    assert without > 1.0
    assert with_term < 1.0
    assert with_term < 0.5 * without


def test_pinch_targets_collapse_only_below_threshold(retargeter):
    r = retargeter
    kp = np.zeros((21, 3))
    kp[8] = [0.01, 0.0, 0.0]
    kp[12] = [0.08, 0.0, 0.0]
    vectors = r._compute_pinch_vectors(kp)
    np.testing.assert_allclose(vectors[0], 0.0)
    np.testing.assert_allclose(vectors[1], [8.0, 0.0, 0.0])


def test_smooth_profile_limits_motion_and_still_closes_pinch():
    r = AdaptiveAnalyticalRetargeter(
        hand_config_path=str(HAND_CONFIG), urdf_path=str(URDF),
        config_path=str(ROOT / "configs/adaptive_analytical_orca_v1_pinch.yaml"),
    )
    kp = _operator_pinch(r, [0.03, 0, 0])
    previous = r._neutral_qpos_phys.copy()
    for _ in range(100):
        q = r._solve(kp)
        assert np.max(np.abs(q - previous)) <= np.deg2rad(2.0) + 1e-8
        previous = q.copy()
    assert _tip_gap_cm(r, q) < 1.0
    # Open the target abruptly: the same bound must apply on release.
    kp[4] += np.array([0.10, 0, 0])
    q = r._solve(kp)
    assert np.max(np.abs(q - previous)) <= np.deg2rad(2.0) + 1e-8


@pytest.fixture
def smooth_retargeter():
    return AdaptiveAnalyticalRetargeter(
        hand_config_path=str(HAND_CONFIG), urdf_path=str(URDF),
        config_path=str(ROOT / "configs/adaptive_analytical_orca_v1_pinch.yaml"),
    )


def test_smooth_distance_target_is_monotone_and_has_matching_boundary_slopes(smooth_retargeter):
    r = smooth_retargeter
    direction = np.array([0.6, 0.0, 0.8])

    def mapped(distance_cm):
        kp = np.zeros((21, 3))
        kp[8] = direction * distance_cm / 100.0
        vector = r._compute_pinch_vectors(kp)[0]
        np.testing.assert_allclose(np.cross(vector, direction), 0, atol=1e-12)
        return float(vector @ direction)

    distances = np.linspace(0, 8, 401)
    gaps = np.array([mapped(d) for d in distances])
    assert np.all(np.diff(gaps) >= -1e-12)
    d1, d2 = r._pinch_d1[0], r._pinch_d2[0]
    assert mapped(d1) == pytest.approx(0)
    assert mapped(d2) == pytest.approx(d2)
    eps = 1e-5
    assert (mapped(d1 + eps) - mapped(d1)) / eps == pytest.approx(0, abs=1e-4)
    assert (mapped(d2) - mapped(d2 - eps)) / eps == pytest.approx(1, abs=1e-4)
    assert (mapped(d2 + eps) - mapped(d2)) / eps == pytest.approx(1, abs=1e-4)


@pytest.mark.parametrize("hz", [15, 30, 60])
def test_velocity_bound_applies_at_different_frame_rates_and_on_release(smooth_retargeter, hz):
    r = smooth_retargeter
    closed = _operator_pinch(r, [0.03, 0, 0])
    opened = closed.copy()
    opened[4] += [0.10, 0, 0]
    previous = r._neutral_qpos_phys.copy()
    moved = []
    for target in (closed, opened):
        segment_motion = 0.0
        for _ in range(hz // 2):
            q = r._solve(target, dt_s=1.0 / hz)
            delta = np.max(np.abs(q - previous))
            assert delta <= np.deg2rad(60.0 / hz) + 1e-8
            segment_motion += delta
            previous = q.copy()
        moved.append(segment_motion)
    # A frozen solution would trivially satisfy a step bound.
    assert all(m > np.deg2rad(5) for m in moved)


def test_source_timing_rejects_old_frames_caps_tracking_pause_and_resets(smooth_retargeter):
    r = smooth_retargeter
    assert r._frame_dt(1_000_000_000) == pytest.approx(1 / 30)
    assert r._frame_dt(1_020_000_000) == pytest.approx(0.02)
    assert r._frame_dt(1_020_000_000) == 0
    assert r._frame_dt(1_010_000_000) == 0
    assert r._frame_dt(1_040_000_000) == pytest.approx(0.02)
    assert r._frame_dt(4_000_000_000) == pytest.approx(0.1)
    r.reset()
    assert r._frame_dt(5_000_000_000) == pytest.approx(1 / 30)


def test_temporal_reference_is_the_emitted_filtered_command(smooth_retargeter, monkeypatch):
    r = smooth_retargeter
    r._calibration_done = True
    r._filter.alpha = 0.25
    kp = _operator_pinch(r, [0.03, 0, 0])
    monkeypatch.setattr(r, "_target_keypoints_in_urdf_frame", lambda _: kp)
    for n in range(3):
        out = r.retarget(TargetPose(kp, timestamp_ns=1_000_000_000 + n * 33_333_333))
        emitted = np.deg2rad([out.data[j] for j in r._finger_joint_ids])
        np.testing.assert_allclose(r._last_qpos_phys, emitted, atol=1e-12)


def test_nonuniform_pinch_pair_weights_have_correct_analytic_gradient(smooth_retargeter):
    r = smooth_retargeter
    r._retarget_cfg["pinch_pair_weights"] = {"index": 1, "middle": 1.5, "ring": 2, "pinky": 2.5}
    r._build_retarget_params()
    kp = _operator_pinch(r, [0.03, 0, 0])
    for n, tip in enumerate((8, 12, 16, 20), 1):
        kp[tip] = kp[4] + [n * 0.001, 0.001, 0]
    args = (r._compute_tip_vectors(kp), r._compute_tip_dirs(kp),
            r._compute_full_hand_vectors(kp), r._compute_pinch_alpha(kp),
            r._neutral_qpos_phys, r._compute_pinch_vectors(kp))
    q = r._neutral_qpos_phys + 0.1
    _, actual = r._loss_and_grad(q, *args)
    eps = 1e-6
    expected = np.array([(r._loss_and_grad(q + d, *args)[0] - r._loss_and_grad(q - d, *args)[0]) / (2 * eps)
                         for d in np.eye(len(q)) * eps])
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_non_index_weights_do_not_change_an_isolated_index_pinch(smooth_retargeter):
    r = smooth_retargeter
    kp = _operator_pinch(r, [0.03, 0, 0])
    for n, tip in enumerate((12, 16, 20), 1):
        kp[tip] = kp[4] + [0.2, 0.1 * n, 0]
    traces = []
    for weights in ({}, {"index": 1, "middle": 2, "ring": 2, "pinky": 2}):
        r._retarget_cfg["pinch_pair_weights"] = weights
        r._build_retarget_params()
        r._last_qpos_phys = None
        traces.append(np.array([r._solve(kp).copy() for _ in range(30)]))
    np.testing.assert_array_equal(traces[0], traces[1])


@pytest.mark.parametrize("weights", [{"thumb": 2}, {"middle": -1}, {"ring": float("nan")}])
def test_invalid_pinch_pair_weights_are_rejected(smooth_retargeter, weights):
    smooth_retargeter._retarget_cfg["pinch_pair_weights"] = weights
    with pytest.raises(ValueError, match="pinch_pair_weights"):
        smooth_retargeter._build_retarget_params()


def test_pinch_diagnostics_report_concurrent_activation_without_changing_state(smooth_retargeter, caplog):
    r = smooth_retargeter
    kp = np.zeros((21, 3))
    kp[[8, 12, 16, 20], 0] = [0.04, 0.01, 0.10, 0.12]
    q = r._neutral_qpos_phys.copy()
    with caplog.at_level("INFO", logger="orca_teleop.retargeting.adaptive_analytical"):
        r._log_pinch_diagnostic(kp, np.rad2deg(q))
        r._log_pinch_diagnostic(kp, np.rad2deg(q))
    assert "active=index,middle" in caplog.text
    assert "cmd_index_deg" in caplog.text
    assert "target_gap_cm=" in caplog.text
    assert "model_gap_cm=" in caplog.text
    assert "thumb_at_limit=" in caplog.text
    assert "cmd_ring_deg" in caplog.text
    assert "cmd_pinky_deg" in caplog.text
    assert "cmd_middle_deg" in caplog.text
    assert sum("Pinch diag" in record.message for record in caplog.records) == 1
    assert r._last_qpos_phys is None
    np.testing.assert_array_equal(r._neutral_qpos_phys, q)


def test_pinch_diagnostic_uses_emitted_pose_and_reports_thumb_limits(smooth_retargeter, caplog):
    r = smooth_retargeter
    q = r._neutral_qpos_phys.copy()
    i = r._finger_joint_ids.index('thumb_mcp')
    q[i] = r._upper_phys[i]
    for joint in ('ring_pip', 'pinky_mcp'):
        i = r._finger_joint_ids.index(joint)
        q[i] = r._upper_phys[i]
    pos, _ = r._robot.compute_positions_and_jacobians(
        r._full_model_qpos(q), r._computed_frame_indices, r._contact_offsets)
    tips = pos[r._tip_indices]
    expected = np.round(np.linalg.norm(tips[1:] - tips[0], axis=1) * 100, 2).tolist()
    # Leave the FK cache at another pose to catch accidental use of stale FK.
    r._robot.compute_positions_and_jacobians(
        r._full_model_qpos(r._neutral_qpos_phys), r._computed_frame_indices)
    kp = np.zeros((21, 3))
    kp[[8, 12, 16, 20], 0] = [.04, .01, .10, .12]
    with caplog.at_level('INFO', logger='orca_teleop.retargeting.adaptive_analytical'):
        r._log_pinch_diagnostic(kp, np.rad2deg(q))
    assert f'model_gap_cm={expected}' in caplog.text
    assert 'thumb_mcp:upper' in caplog.text
    assert 'ring_pip:upper' in caplog.text
    assert 'pinky_mcp:upper' in caplog.text
    assert r._last_qpos_phys is None


def test_relaxed_tip_position_gradient_matches_finite_differences(smooth_retargeter):
    r = smooth_retargeter
    r._retarget_cfg['pinch_relax_tip_position'] = True
    r._build_retarget_params()
    kp = _operator_pinch(r, [0.03, 0, 0])
    kp[12] = kp[4] + [0.035, 0, 0]  # Partially released middle-pinch position.
    kp[16] = kp[4] + [0.08, 0, 0]
    kp[20] = kp[4] + [0.10, 0, 0]
    args = (r._compute_tip_vectors(kp), r._compute_tip_dirs(kp),
            r._compute_full_hand_vectors(kp), r._compute_pinch_alpha(kp),
            r._neutral_qpos_phys, r._compute_pinch_vectors(kp))
    q = r._neutral_qpos_phys + .1
    _, actual = r._loss_and_grad(q, *args)
    eps = 1e-6
    expected = np.array([(r._loss_and_grad(q + d, *args)[0] - r._loss_and_grad(q - d, *args)[0]) / (2 * eps)
                         for d in np.eye(len(q)) * eps])
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('mode', ['no_active_pinch', 'pinch_disabled', 'pairs_disabled'])
def test_relaxation_leaves_non_pinch_objective_identical(smooth_retargeter, mode):
    r = smooth_retargeter
    kp = _operator_pinch(r, [0.03, 0, 0])
    alphas = r._compute_pinch_alpha(kp)
    if mode == 'no_active_pinch':
        alphas = np.zeros(5)
    elif mode == 'pinch_disabled':
        r._w_pinch = 0
    else:
        r._pinch_pair_weights[:] = 0
    args = (r._compute_tip_vectors(kp), r._compute_tip_dirs(kp),
            r._compute_full_hand_vectors(kp), alphas,
            r._neutral_qpos_phys, r._compute_pinch_vectors(kp))
    q = r._neutral_qpos_phys.copy()
    r._pinch_relax_tip_position = False
    before = r._loss_and_grad(q, *args)
    r._pinch_relax_tip_position = True
    after = r._loss_and_grad(q, *args)
    assert after[0] == before[0]
    np.testing.assert_array_equal(after[1], before[1])


def test_closed_pinch_releases_only_involved_absolute_tip_targets(smooth_retargeter):
    r = smooth_retargeter
    r._pinch_relax_tip_position = True
    kp = _operator_pinch(r, [0.03, 0, 0])
    args = [r._compute_tip_vectors(kp), r._compute_tip_dirs(kp),
            r._compute_full_hand_vectors(kp), np.array([.7, .7, .35, 0, 0]),
            r._neutral_qpos_phys, r._compute_pinch_vectors(kp)]
    q = r._neutral_qpos_phys.copy()
    loss, grad = r._loss_and_grad(q, *args)
    args[0] = args[0].copy()
    args[0][:2] += [10, 0, 0]  # Closed thumb/index: absolute target no longer wins.
    changed_loss, changed_grad = r._loss_and_grad(q, *args)
    assert changed_loss == pytest.approx(loss, abs=1e-10)
    np.testing.assert_allclose(changed_grad, grad, atol=1e-12)
    args[0][2] += [10, 0, 0]  # Partial middle: its position objective still matters.
    assert r._loss_and_grad(q, *args)[0] != pytest.approx(loss)


def test_human_thumb_ip_bend_tracks_shape_and_is_rigid_scale_invariant():
    kp = np.zeros((21, 3))
    kp[2], kp[3] = [0, 0, 0], [0, 0, .03]
    angle = np.deg2rad(45)
    kp[4] = kp[3] + [.02 * np.sin(angle), 0, .02 * np.cos(angle)]
    fn = AdaptiveAnalyticalRetargeter._human_thumb_ip_bend
    assert fn(kp) == pytest.approx(angle)
    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert fn(2.5 * kp @ rot + [1, 2, 3]) == pytest.approx(angle)
    kp[4] = kp[3] + [0, 0, .02]
    assert fn(kp) == pytest.approx(0)
    assert fn(np.zeros((21, 3))) is None


@pytest.mark.parametrize('dip_deg', [-20, 0, 30, 80])
def test_model_thumb_bend_reference_matches_fk_under_parent_rotations(smooth_retargeter, dip_deg):
    r = smooth_retargeter
    q = r._neutral_qpos_phys.copy()
    for joint, degrees in [('thumb_mcp', 20), ('thumb_abd', 10), ('thumb_pip', 40), ('thumb_dip', dip_deg)]:
        q[r._finger_joint_ids.index(joint)] = np.deg2rad(degrees)
    pos = _positions(r, q, r._contact_offsets)
    a = pos[r._dip_indices[0]] - pos[r._pip_indices[0]]
    b = pos[r._tip_indices[0]] - pos[r._dip_indices[0]]
    geometric = np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))
    assert geometric == pytest.approx(abs(np.deg2rad(dip_deg) + r._thumb_ip_rest_bend), abs=1e-10)


def test_thumb_bend_loss_gradient_and_inactive_pinch_behavior(smooth_retargeter):
    r = smooth_retargeter
    r._w_thumb_ip_bend = 1
    kp = _operator_pinch(r, [.03, 0, 0])
    args = [r._compute_tip_vectors(kp), r._compute_tip_dirs(kp),
            r._compute_full_hand_vectors(kp), r._compute_pinch_alpha(kp),
            r._neutral_qpos_phys, r._compute_pinch_vectors(kp), np.deg2rad(35)]
    q = r._neutral_qpos_phys + .1
    _, actual = r._loss_and_grad(q, *args)
    eps = 1e-6
    expected = np.array([(r._loss_and_grad(q+d, *args)[0] - r._loss_and_grad(q-d, *args)[0]) / (2*eps)
                         for d in np.eye(len(q))*eps])
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    args[3] = np.zeros(5)
    with_target = r._loss_and_grad(q, *args)
    args[-1] = None
    without_target = r._loss_and_grad(q, *args)
    assert with_target[0] == without_target[0]
    np.testing.assert_array_equal(with_target[1], without_target[1])


def test_thumb_bend_prefers_flexion_over_hyperextension(smooth_retargeter):
    r = smooth_retargeter
    r._w_thumb_ip_bend = 1
    kp = _operator_pinch(r, [.03, 0, 0])
    args = [r._compute_tip_vectors(kp), r._compute_tip_dirs(kp),
            r._compute_full_hand_vectors(kp), np.array([.7, .7, 0, 0, 0]),
            None, r._compute_pinch_vectors(kp)]
    q = r._neutral_qpos_phys.copy()
    q[r._thumb_ip_index] = np.deg2rad(-15)
    before = r._loss_and_grad(q, *args)
    after = r._loss_and_grad(q, *args, thumb_ip_bend=np.deg2rad(40))
    delta = after[1] - before[1]
    assert delta[r._thumb_ip_index] < 0  # Descent increases the DIP hinge angle.
    assert np.count_nonzero(np.abs(delta) > 1e-12) == 1


@pytest.mark.parametrize('weight', [-1, float('nan'), float('inf')])
def test_invalid_thumb_bend_weights_are_rejected(smooth_retargeter, weight):
    smooth_retargeter._retarget_cfg['w_thumb_ip_bend'] = weight
    with pytest.raises(ValueError, match='w_thumb_ip_bend'):
        smooth_retargeter._build_retarget_params()


@pytest.mark.parametrize('pinch_weight', [0, 4])
def test_always_thumb_bend_follows_without_active_pinch(smooth_retargeter, pinch_weight):
    r = smooth_retargeter
    r._retarget_cfg.update(w_thumb_ip_bend=5, thumb_ip_bend_scope='always', w_pinch=pinch_weight)
    r._build_retarget_params()
    kp = _operator_pinch(r, [.03, 0, 0])
    args = (r._compute_tip_vectors(kp), r._compute_tip_dirs(kp),
            r._compute_full_hand_vectors(kp), np.zeros(5), None, r._compute_pinch_vectors(kp))
    q = r._neutral_qpos_phys.copy()
    without = r._loss_and_grad(q, *args)
    outputs = []
    for bend in [0, 20, 40, 60]:
        target = np.deg2rad(bend)
        loss, grad = r._loss_and_grad(q, *args, thumb_ip_bend=target)
        error = q[r._thumb_ip_index] + r._thumb_ip_rest_bend - target
        assert loss - without[0] == pytest.approx(5 * error**2)
        correction = grad - without[1]
        assert correction[r._thumb_ip_index] == pytest.approx(10 * error)
        np.testing.assert_allclose(np.delete(correction, r._thumb_ip_index), 0, atol=1e-12)
        outputs.append(correction[r._thumb_ip_index])
    # Greater measured bending must increasingly pull the joint toward flexion.
    assert np.all(np.diff(outputs) < 0)


def test_always_thumb_bend_gradient_and_missing_tracking(smooth_retargeter):
    r = smooth_retargeter
    r._retarget_cfg.update(w_thumb_ip_bend=5, thumb_ip_bend_scope='always')
    r._build_retarget_params()
    kp = _operator_pinch(r, [.03, 0, 0])
    args = (r._compute_tip_vectors(kp), r._compute_tip_dirs(kp),
            r._compute_full_hand_vectors(kp), np.zeros(5), None, r._compute_pinch_vectors(kp))
    q = r._neutral_qpos_phys.copy() + .1
    _, actual = r._loss_and_grad(q, *args, thumb_ip_bend=.6)
    eps = 1e-6
    expected = np.array([(r._loss_and_grad(q+d, *args, thumb_ip_bend=.6)[0]
                          - r._loss_and_grad(q-d, *args, thumb_ip_bend=.6)[0]) / (2*eps)
                         for d in np.eye(len(q))*eps])
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    without = r._loss_and_grad(q, *args)
    r._w_thumb_ip_bend = 0
    disabled = r._loss_and_grad(q, *args)
    assert without[0] == disabled[0]
    np.testing.assert_array_equal(without[1], disabled[1])


def test_invalid_thumb_bend_scope_is_rejected(smooth_retargeter):
    smooth_retargeter._retarget_cfg['thumb_ip_bend_scope'] = 'unknown'
    with pytest.raises(ValueError, match='thumb_ip_bend_scope'):
        smooth_retargeter._build_retarget_params()


def _free_thumb_pose(r, bend_deg):
    kp = _operator_pinch(r, [.03, 0, 0])
    kp[2], kp[3] = [-.03, 0, 0], [-.03, 0, .03]
    angle = np.deg2rad(bend_deg)
    kp[4] = kp[3] + [.02 * np.sin(angle), 0, .02 * np.cos(angle)]
    # Keep every other fingertip outside the pinch activation distance.
    for n, tip in enumerate((8, 12, 16, 20), 1):
        kp[tip] = [.15, .05 * n, .05]
    return kp


@pytest.mark.parametrize('scope', ['always', 'free'])
def test_direct_free_thumb_follows_bending_and_opening_with_velocity_bounds(smooth_retargeter, scope):
    r = smooth_retargeter
    r._retarget_cfg.update(thumb_ip_tracking='direct_free', thumb_ip_bend_scope=scope,
                           w_thumb_ip_bend=5)
    r._build_retarget_params()
    before_lower, before_upper = r._lower_phys.copy(), r._upper_phys.copy()
    previous = r._neutral_qpos_phys.copy()
    for angle in (0, 20, 40, 60, 0):
        kp = _free_thumb_pose(r, angle)
        assert not r._compute_pinch_alpha(kp).any()
        for _ in range(40):
            q = r._solve(kp, dt_s=1/30)
            assert np.max(abs(q - previous)) <= np.deg2rad(2) + 1e-8
            previous = q.copy()
        assert np.rad2deg(q[r._thumb_ip_index] + r._thumb_ip_rest_bend) == pytest.approx(angle, abs=1e-7)
    np.testing.assert_array_equal(r._lower_phys, before_lower)
    np.testing.assert_array_equal(r._upper_phys, before_upper)


@pytest.mark.parametrize('finger', range(4), ids=['index', 'middle', 'ring', 'pinky'])
@pytest.mark.parametrize('fraction,remaining', [(0, 1), (.5, .5), (1, 0)])
def test_free_thumb_bend_releases_loss_and_gradient_for_each_pinch(
        smooth_retargeter, finger, fraction, remaining):
    r = smooth_retargeter
    r._retarget_cfg.update(thumb_ip_tracking='direct_free', thumb_ip_bend_scope='free',
                           w_thumb_ip_bend=5)
    r._build_retarget_params()
    kp = _free_thumb_pose(r, 40)
    # Distances drive the actual pair activation, including simultaneous pairs.
    kp[(8, 12, 16, 20)[finger]] = kp[4] + [.05 - fraction * .021, 0, 0]
    alphas = r._compute_pinch_alpha(kp)
    args = (r._compute_tip_vectors(kp), r._compute_tip_dirs(kp),
            r._compute_full_hand_vectors(kp), alphas, None, r._compute_pinch_vectors(kp))
    q = r._neutral_qpos_phys + .1
    baseline_loss, baseline_grad = r._loss_and_grad(q, *args)
    target = np.deg2rad(40)
    loss, grad = r._loss_and_grad(q, *args, thumb_ip_bend=target)
    error = q[r._thumb_ip_index] + r._thumb_ip_rest_bend - target
    assert loss - baseline_loss == pytest.approx(5 * remaining * error**2, abs=1e-7)
    correction = grad - baseline_grad
    assert correction[r._thumb_ip_index] == pytest.approx(10 * remaining * error, abs=1e-7)
    np.testing.assert_allclose(np.delete(correction, r._thumb_ip_index), 0, atol=1e-12)
    eps = 1e-6
    numeric = np.array([(r._loss_and_grad(q+d, *args, thumb_ip_bend=target)[0]
                         - r._loss_and_grad(q-d, *args, thumb_ip_bend=target)[0]) / (2*eps)
                        for d in np.eye(len(q))*eps])
    np.testing.assert_allclose(grad, numeric, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('finger', range(4), ids=['index', 'middle', 'ring', 'pinky'])
def test_full_pinch_free_scope_solve_matches_no_human_thumb_bend(smooth_retargeter, finger):
    r = smooth_retargeter
    r._retarget_cfg.update(thumb_ip_tracking='direct_free', thumb_ip_bend_scope='free',
                           w_thumb_ip_bend=5)
    r._build_retarget_params()
    kp = _free_thumb_pose(r, 60)
    kp[(8, 12, 16, 20)[finger]] = kp[4]
    r._last_qpos_phys = r._neutral_qpos_phys.copy()
    actual = r._solve(kp)
    r._w_thumb_ip_bend = 0
    r._last_qpos_phys = r._neutral_qpos_phys.copy()
    expected = r._solve(kp)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('disabled', ['pair', 'contact'])
def test_free_scope_does_not_release_thumb_for_disabled_contact(smooth_retargeter, disabled):
    r = smooth_retargeter
    r._retarget_cfg.update(thumb_ip_bend_scope='free', w_thumb_ip_bend=5)
    r._build_retarget_params()
    kp = _free_thumb_pose(r, 40)
    kp[8] = kp[4]
    if disabled == 'pair':
        r._pinch_pair_weights[:] = 0
    else:
        r._w_pinch = 0
    args = (r._compute_tip_vectors(kp), r._compute_tip_dirs(kp),
            r._compute_full_hand_vectors(kp), r._compute_pinch_alpha(kp), None,
            r._compute_pinch_vectors(kp))
    q = r._neutral_qpos_phys.copy()
    without = r._loss_and_grad(q, *args)
    with_bend = r._loss_and_grad(q, *args, thumb_ip_bend=.6)
    error = q[r._thumb_ip_index] + r._thumb_ip_rest_bend - .6
    assert with_bend[0] - without[0] == pytest.approx(5 * error**2)


def test_direct_free_preserves_pinch_solve_given_same_pose(smooth_retargeter):
    r = smooth_retargeter
    r._retarget_cfg.update(thumb_ip_bend_scope='always', w_thumb_ip_bend=5)
    r._build_retarget_params()
    kp = _operator_pinch(r, [.03, 0, 0])
    assert r._compute_pinch_alpha(kp)[1] > 0
    outputs = []
    for mode in ('soft', 'direct_free'):
        r._thumb_ip_tracking = mode
        r._last_qpos_phys = r._neutral_qpos_phys.copy()
        outputs.append(r._solve(kp))
    np.testing.assert_array_equal(*outputs)


def test_direct_free_holds_thumb_on_degenerate_thumb_input(smooth_retargeter):
    r = smooth_retargeter
    r._retarget_cfg.update(thumb_ip_tracking='direct_free', thumb_ip_bend_scope='always')
    r._build_retarget_params()
    kp = _free_thumb_pose(r, 30)
    kp[4] = kp[3]
    before = r._neutral_qpos_phys[r._thumb_ip_index]
    q = r._solve(kp)
    assert q[r._thumb_ip_index] == before


def test_free_thumb_target_reaches_id15_command_conversion(smooth_retargeter, monkeypatch):
    import dataclasses
    import mujoco
    from orca_teleop.am200.canonical import CanonicalJointMap
    from orca_teleop.am200.composer import HoldArmActionComposer

    r = smooth_retargeter
    r._retarget_cfg.update(thumb_ip_tracking='direct_free', thumb_ip_bend_scope='always')
    r._build_retarget_params()
    r._calibration_done = True
    kp = _free_thumb_pose(r, 40)
    monkeypatch.setattr(r, '_target_keypoints_in_urdf_frame', lambda _: kp)
    for n in range(40):
        action = r.retarget(TargetPose(kp, timestamp_ns=1_000_000_000 + n * 33_333_333))
    expected = 40 - np.rad2deg(r._thumb_ip_rest_bend)
    assert action.data['thumb_dip'] == pytest.approx(expected)
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'orca_am200/robot_model/am_arm200_orca_right.xml'))
    mapping = CanonicalJointMap.load(ROOT / 'orca_am200/robot_model/joint_map.json', model)
    canonical = HoldArmActionComposer(mapping)(None, action)
    applied = mapping.hand_to_physical(canonical.position[mapping.hand_slice])
    assert applied['thumb_dip'] == pytest.approx(expected, abs=1e-6)
    # Exercise calibration/ID routing as pure math; never connect a serial bus.
    hand = r._hand
    assert hand.config.joint_to_motor_map['thumb_dip'] == 15
    # Exercise routing with explicit synthetic calibration, independently of
    # whether this real unit is currently awaiting a fresh thumb calibration.
    low, high = hand.config.joint_roms_dict['thumb_dip']
    hand.calibration = dataclasses.replace(
        hand.calibration,
        motor_limits_dict=hand.motor_limits_dict | {15: [-5., -2.5]},
        joint_to_motor_ratios_dict=hand.joint_to_motor_ratios_dict | {15: 2.5 / (high - low)},
    )
    hand._wrap_offsets_dict = {m: 0 for m in hand.config.motor_ids}
    motor = hand._joint_to_motor_pos(applied)
    estimated = hand._motor_to_joint_pos(motor)
    assert estimated['thumb_dip'] == pytest.approx(expected)
    next_motor = hand._joint_to_motor_pos({**applied, 'thumb_dip': expected + 10})
    i = hand.config.motor_id_to_idx_dict[15]
    assert abs(next_motor[i] - motor[i]) > 0.01


@pytest.mark.parametrize('scope,mode', [('always','unknown'), ('pinch','direct_free')])
def test_invalid_direct_tracking_configuration(smooth_retargeter, scope, mode):
    smooth_retargeter._retarget_cfg.update(thumb_ip_tracking=mode, thumb_ip_bend_scope=scope)
    with pytest.raises(ValueError, match='thumb IP tracking|thumb_ip_tracking'):
        smooth_retargeter._build_retarget_params()
