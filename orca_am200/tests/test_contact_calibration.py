"""Contact fit must generalize across poses, not simply close recorded points."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from orca_teleop.retargeting.contact_calibration import (
    FINGERS, fit_offsets, fingerprint, load_offsets,
)
from orca_teleop.retargeting.adaptive_analytical import AdaptiveAnalyticalRetargeter

ROOT = Path(__file__).resolve().parents[2]


def contacts(offsets, seed=42):
    rng = np.random.default_rng(seed)
    samples = []
    for finger in FINGERS[1:]:
        for _ in range(6):
            rotations = []
            for _ in FINGERS:
                q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
                q[:, -1] *= np.linalg.det(q)
                rotations.append(q)
            rotations = np.array(rotations)
            point = rng.normal(size=3) * .03
            positions = point - np.einsum('fij,fj->fi', rotations, offsets)
            samples.append(dict(finger=finger, positions_m=positions, rotations=rotations))
    return samples


def test_fit_recovers_all_five_contact_points_and_validates_unseen_poses():
    offsets = np.array([[.003, -.002, .001], [.002, .004, 0],
                        [-.001, .006, .002], [.004, -.005, .002], [.005, -.006, 0]])
    result = fit_offsets(contacts(offsets))
    np.testing.assert_allclose([result['offsets_m'][f] for f in FINGERS], offsets, atol=1e-12)
    assert result['training_samples'] == 16
    assert result['validation_samples'] == 8
    assert max(e for v in result['validation_errors_m'].values() for e in v) < 1e-12


def test_repeated_poses_cannot_identify_contact_points():
    samples = contacts(np.zeros((5, 3)))
    for sample in samples:
        sample['rotations'] = np.repeat(np.eye(3)[None], 5, axis=0)
    with pytest.raises(ValueError, match='do not identify'):
        fit_offsets(samples)


def test_inconsistent_held_out_contact_rejects_pose_dependent_model_error():
    samples = contacts(np.zeros((5, 3)))
    samples[-1]['positions_m'][4] += [.01, 0, 0]
    with pytest.raises(ValueError, match='Held-out'):
        fit_offsets(samples)


def test_excessive_correction_requires_joint_calibration_not_an_offset():
    offsets = np.zeros((5, 3)); offsets[4, 0] = .025
    with pytest.raises(ValueError, match='15 mm'):
        fit_offsets(contacts(offsets))


def make_retargeter(config=None):
    return AdaptiveAnalyticalRetargeter(
        hand_config_path=str(ROOT/'configs/orcahand-right-v1-feetech/config.yaml'),
        urdf_path=str(ROOT/'orcahand_description/v1/models/urdf/orcahand_right.urdf'),
        config_path=str(config or ROOT/'configs/adaptive_analytical_orca_v1_pinch_refined.yaml'),
    )


def test_profile_is_bound_to_current_model_and_motor_calibration(tmp_path):
    r = make_retargeter()
    profile = fit_offsets(contacts(np.zeros((5, 3))))
    profile['fingerprint'] = fingerprint(r)
    path = tmp_path/'profile.json'; path.write_text(json.dumps(profile))
    assert set(load_offsets(path, r)) == set(FINGERS)
    r._hand.joint_to_motor_ratios_dict[13] += .0001
    with pytest.raises(ValueError, match='stale'):
        load_offsets(path, r)


def test_fitted_points_have_correct_optimizer_jacobians(tmp_path):
    import yaml
    r = make_retargeter()
    offsets = np.array([[.002, .001, 0], [0, .003, 0], [0, .004, 0],
                        [.001, .003, 0], [.002, .004, 0]])
    profile = fit_offsets(contacts(offsets)); profile['fingerprint'] = fingerprint(r)
    path = tmp_path/'profile.json'; path.write_text(json.dumps(profile))
    cfg = r.config; cfg['retarget']['contact_calibration'] = path.name
    config = tmp_path/'calibrated.yaml'; config.write_text(yaml.safe_dump(cfg))
    corrected = make_retargeter(config)
    np.testing.assert_allclose([corrected._contact_offsets[i] for i in corrected._tip_indices], offsets, atol=1e-12)
    np.testing.assert_allclose(corrected._frame_offsets, r._frame_offsets)
    q = corrected._neutral_qpos_phys + .1
    positions, jac = corrected._robot.compute_positions_and_jacobians(
        corrected._full_model_qpos(q), corrected._computed_frame_indices, corrected._contact_offsets)
    analytic = jac[:, :, corrected._model_q_indices_np].copy()
    numeric = np.empty_like(analytic); eps = 1e-6
    for j in range(len(q)):
        d = np.zeros_like(q); d[j] = eps
        plus, _ = corrected._robot.compute_positions_and_jacobians(
            corrected._full_model_qpos(q+d), corrected._computed_frame_indices, corrected._contact_offsets)
        minus, _ = corrected._robot.compute_positions_and_jacobians(
            corrected._full_model_qpos(q-d), corrected._computed_frame_indices, corrected._contact_offsets)
        numeric[:, :, j] = (plus-minus)/(2*eps)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-7)


def test_capture_only_reads_and_releases_fingers(tmp_path, monkeypatch):
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location('contact_cli', ROOT/'scripts/calibrate_hand_contacts.py')
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    r = make_retargeter()
    r._hand.config = __import__('dataclasses').replace(r._hand.config, port=str(tmp_path))
    calls = []
    class Reader:
        def __init__(self, config):
            assert 'wrist' not in config.joint_ids
            assert 17 not in config.motor_ids
            calls.append('created')
        def is_calibrated(self): return True
        def _try_port(self, port, config): calls.append('connect')
        def disable_torque(self): calls.append('release'); return []
        def disconnect(self): calls.append('disconnect')
    monkeypatch.setattr(cli, 'ContactReader', Reader)
    monkeypatch.setattr(cli, 'stable_angles', lambda *a: {})
    monkeypatch.setattr('builtins.input', lambda *a: 'q')
    cli.capture(SimpleNamespace(samples=tmp_path/'samples.json'), r)
    assert calls == ['created', 'connect', 'release', 'disconnect']
