"""Exercise the real upstream solver and the local pipeline adapter, without USB."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("dex_retargeting")

from dex_retargeting.optimizer import DexPilotOptimizer
from orca_teleop.retargeting.retargeter import Retargeter, TargetPose

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def r():
    return Retargeter.from_paths(
        model_path=str(ROOT / "configs/orcahand-right-v1-feetech/config.yaml"),
        urdf_path=str(ROOT / "orcahand_description/v1/models/urdf/orcahand_right.urdf"),
        backend="dexpilot", config_path=str(ROOT / "configs/dexpilot_orca_v1.yaml"),
    )


def hand_points():
    p = np.zeros((21, 3))
    bases = [[-.035, .025, 0], [-.025, .055, 0], [0, .060, 0],
             [.022, .055, 0], [.040, .045, 0]]
    for i, base in enumerate(bases):
        p[1 + 4*i:5 + 4*i] = np.array(base) + np.array([
            [0, 0, 0], [0, .025, 0], [0, .050, 0], [0, .075, 0]])
    return p


def test_official_five_finger_solver_and_physical_bounds(r):
    assert type(r._optimizer) is DexPilotOptimizer
    assert r._optimizer.num_fingers == 5
    assert r._optimizer.target_link_human_indices.shape == (2, 15)
    assert r._optimizer.fixed_joint_names == ["right_wrist"]
    assert r._optimizer.target_joint_names == [f"right_{j}" for j in r._finger_joint_ids]
    np.testing.assert_allclose(r._optimizer.opt.get_lower_bounds(), r._lower_phys)
    np.testing.assert_allclose(r._optimizer.opt.get_upper_bounds(), r._upper_phys)
    # Preserve package defaults instead of injecting adaptive/refined-pinch settings.
    assert r._optimizer.eta1 == 1e-4
    assert r._optimizer.eta2 == .03
    assert r._optimizer.scaling == 1


def test_input_vectors_are_rotation_and_translation_invariant(r):
    p = hand_points()
    angle = .71
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    expected = r._target_vectors(TargetPose(p))
    actual = r._target_vectors(TargetPose(p @ rotation.T + [1, 2, 3]))
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("finger,tip", enumerate([8, 12, 16, 20]))
def test_each_pinch_activates_correct_upstream_pair_and_resets(r, finger, tip):
    r.reset()
    points = hand_points()
    points[4] = points[tip] + [.001, 0, 0]
    pose = TargetPose(points, wrist_angle_degrees=10)
    vectors = r._target_vectors(pose)
    # The first four vectors must be index/middle/ring/pinky -> thumb, not swapped.
    assert np.linalg.norm(vectors[finger]) == pytest.approx(.001)
    out = r.retarget(pose)
    assert r._optimizer.projected[finger]
    assert np.isfinite(r._last_qpos_phys).all()
    assert np.all(r._last_qpos_phys >= r._lower_phys)
    assert np.all(r._last_qpos_phys <= r._upper_phys)
    assert out.as_array(['wrist'])[0] == -10
    first = r._last_qpos_phys.copy()
    for _ in range(3):
        r.retarget(pose)
    r.reset()
    assert not r._optimizer.projected.any()
    assert r._last_qpos_phys is None
    r.retarget(pose)
    np.testing.assert_allclose(r._last_qpos_phys, first, atol=1e-7)


def test_invalid_landmarks_do_not_enter_optimizer(r):
    calls = r._seq.num_retargeting
    points = hand_points()
    points[4, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        r.retarget(TargetPose(points))
    with pytest.raises(ValueError, match="canonical"):
        r.retarget(TargetPose(hand_points(), source="unsupported"))
    assert r._seq.num_retargeting == calls


def test_elapsed_time_bound_includes_duplicate_and_out_of_order_frames(r):
    r.reset()
    points = hand_points()
    previous = r._neutral.copy()
    times = [1_000_000_000, 1_020_000_000, 1_020_000_000, 1_010_000_000, 1_040_000_000]
    intervals = [1/30, .02, 0, 0, .02]
    for t, dt in zip(times, intervals):
        r.retarget(TargetPose(points, timestamp_ns=t))
        assert np.max(np.abs(r._last_qpos_phys - previous)) <= np.deg2rad(60) * dt + 1e-10
        previous = r._last_qpos_phys.copy()


def test_urdf_rotations_match_command_adapter(r):
    import mujoco
    from orca_core import OrcaJointPositions
    from orca_teleop.am200.canonical import CanonicalJointMap
    from orca_teleop.am200.composer import HoldArmActionComposer

    model = mujoco.MjModel.from_xml_path(str(ROOT / "orca_am200/robot_model/am_arm200_orca_right.xml"))
    data = mujoco.MjData(model)
    mapping = CanonicalJointMap.load(ROOT / "orca_am200/robot_model/joint_map.json", model)
    angles = dict(r._hand.config.neutral_position)
    angles.update(thumb_mcp=15, thumb_abd=10, thumb_pip=40, thumb_dip=30,
                  index_mcp=35, middle_mcp=25, ring_mcp=20, pinky_mcp=15, wrist=0)
    refs = mapping.ref[mapping.hand_slice]
    values = np.clip(np.deg2rad([angles[j] for j in mapping.hand_joint_ids]),
                     mapping.lower[mapping.hand_slice] - refs,
                     mapping.upper[mapping.hand_slice] - refs)
    angles = dict(zip(mapping.hand_joint_ids, np.rad2deg(values)))
    command = HoldArmActionComposer(mapping)(None, OrcaJointPositions(angles))
    data.qpos[mapping.qpos_index] = command.position
    mujoco.mj_forward(model, data)
    q = np.zeros(r._robot.dof)
    q[r._model_q_indices_np] = np.deg2rad([angles[j] for j in r._finger_joint_ids])
    r._robot.compute_forward_kinematics(q)
    palm = r._robot.get_link_pose(r._robot.get_link_index('right_palm')).copy()
    sim_palm = model.body('right_palm').id
    sim_rotation = data.xmat[sim_palm].reshape(3, 3)
    for finger in ('thumb', 'index', 'middle', 'ring', 'pinky'):
        name = f'right_{finger}_ip'
        pose = r._robot.get_link_pose(r._robot.get_link_index(name))
        local = palm[:3, :3].T @ (pose[:3, 3] - palm[:3, 3])
        sim = sim_rotation.T @ (data.xpos[model.body(name).id] - data.xpos[sim_palm])
        np.testing.assert_allclose(local, sim, atol=2e-6, err_msg=name)
