"""Paper equations, analytic derivatives, and the real ORCA ingress contract."""
from pathlib import Path

import numpy as np
import pytest

from orca_teleop.retargeting.key_objectives import KeyObjectivesRetargeter, pinch_targets
from orca_teleop.retargeting.retargeter import Retargeter, TargetPose

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/key_objectives_orca_v1.yaml"
HAND = ROOT / "configs/orcahand-right-v1-feetech/config.yaml"
URDF = ROOT / "orcahand_description/v1/models/urdf/orcahand_right.urdf"


@pytest.fixture
def r():
    return Retargeter.from_paths(str(HAND), str(URDF), backend="key_objectives", config_path=str(CONFIG))


def test_distance_mapping_and_complementary_weights():
    distances = np.array([0, 0.01, 0.055, 0.10, 0.12])
    vectors = np.column_stack([distances, np.zeros((5, 2))])
    target, pinch, shape = pinch_targets(vectors, 0.01, 0.10, 10)
    np.testing.assert_allclose(target[:, 0], [0, 0, 0.05, 0.10, 0.12], atol=1e-12)
    np.testing.assert_allclose(target[:, 1:], 0)
    np.testing.assert_allclose(pinch + shape, 1)
    assert np.all(np.diff(pinch) < 0)
    assert pinch[3] == pytest.approx(0.5)


def test_analytic_gradient_matches_all_terms_in_finite_difference(r):
    rng = np.random.default_rng(24)
    kp = rng.normal(0, 0.03, (21, 3))
    targets = r._prepare_targets(kp)
    q = r._neutral_qpos_phys + 0.10
    previous = q - 0.02
    _, analytic = r._loss_and_grad(q, targets, previous)
    eps = 1e-6
    numeric = np.array([(r._loss_and_grad(q + d, targets, previous)[0]
                         - r._loss_and_grad(q - d, targets, previous)[0]) / (2 * eps)
                        for d in np.eye(len(q)) * eps])
    np.testing.assert_allclose(analytic, numeric, atol=1e-8, rtol=2e-5)
    assert all(value > 0 for value in r.last_terms.values())


def test_independent_backend_ingress_and_reset(r, monkeypatch):
    from orca_teleop.retargeting.adaptive_analytical import AdaptiveAnalyticalRetargeter

    def forbidden(*args, **kwargs):
        raise AssertionError("Paper backend must not invoke adaptive optimization")

    monkeypatch.setattr(AdaptiveAnalyticalRetargeter, "_solve", forbidden)
    kp = np.zeros((21, 3))
    # A nondegenerate synthetic open hand; no user capture/ignored data needed.
    for start, x, y in ((1, 0.04, 0.02), (5, 0.02, 0.06), (9, 0, 0.07),
                        (13, -0.02, 0.06), (17, -0.04, 0.05)):
        for k in range(4):
            kp[start + k] = [x, y + k * 0.02, 0]
    for _ in range(30):
        assert r.retarget(TargetPose(kp)) is None
    command = r.retarget(TargetPose(kp, wrist_angle_degrees=12.0))
    assert command is not None and len(command.data) == 17
    assert command.data["wrist"] == pytest.approx(-12.0)
    q = np.deg2rad([command.data[j] for j in r._finger_joint_ids])
    assert np.isfinite(q).all()
    assert np.all(q >= r._lower_phys) and np.all(q <= r._upper_phys)
    assert all(np.array_equal(p, np.zeros(3)) for p in r._frame_offsets)
    r.reset()
    assert r._last_qpos_phys is None and not r._calibration_done and r._mano_scale == 1


def test_solver_reduces_its_objective_without_exceeding_limits(r):
    kp = np.random.default_rng(7).normal(0, 0.03, (21, 3))
    targets = r._prepare_targets(kp)
    before, _ = r._loss_and_grad(r._neutral_qpos_phys, targets)
    q = r._solve(kp)
    after, _ = r._loss_and_grad(q, targets)
    assert after < before
    assert np.all(q >= r._lower_phys) and np.all(q <= r._upper_phys)


def test_adaptive_profile_is_rejected(r):
    with pytest.raises(ValueError, match="optimizer.type"):
        KeyObjectivesRetargeter.from_paths(str(HAND), str(URDF),
                                          str(ROOT / "configs/adaptive_analytical_orca_v1_pinch.yaml"))


def test_nonfinite_frame_does_not_corrupt_state(r):
    with pytest.raises(ValueError, match="finite"):
        r.retarget(TargetPose(np.full((21, 3), np.nan)))
    assert not r._calibration_mags and r._last_qpos_phys is None


@pytest.mark.parametrize("hz", [15, 30, 60])
def test_execution_limit_bounds_closing_release_and_duplicate_frames(r, hz):
    r._calibration_done = True
    kp = np.zeros((21, 3))
    for start, x, y in ((1, 0.04, 0.02), (5, 0.02, 0.06), (9, 0, 0.07),
                        (13, -0.02, 0.06), (17, -0.04, 0.05)):
        for k in range(4):
            kp[start + k] = [x, y + k * 0.02, 0]
    previous = r._neutral_qpos_phys.copy()
    moved = 0.0
    for i in range(20):
        target = kp.copy()
        if i < 10:
            target[4] = target[12]  # close onto middle, then release
        timestamp = 1_000_000_000 + round(i * 1e9 / hz)
        r.retarget(TargetPose(target, timestamp_ns=timestamp))
        q = r._last_qpos_phys.copy()
        limit = np.deg2rad(60 / (30 if i == 0 else hz))
        assert np.max(np.abs(q - previous)) <= limit + 1e-8
        moved += float(np.linalg.norm(q - previous))
        r.retarget(TargetPose(kp, timestamp_ns=timestamp))
        np.testing.assert_array_equal(r._last_qpos_phys, q)
        previous = q
    assert moved > np.deg2rad(5)
