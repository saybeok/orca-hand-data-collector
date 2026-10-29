"""Compare retarget FK with the simulator after the actual command adapter."""
from pathlib import Path

import mujoco
import numpy as np
import pytest
from orca_core import OrcaJointPositions
from orca_teleop.am200.canonical import CanonicalJointMap
from orca_teleop.am200.composer import HoldArmActionComposer
from orca_teleop.retargeting.retargeter import Retargeter

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("pose", ["zero", "neutral", "bent"])
@pytest.mark.parametrize("backend", ["adaptive_analytical", "key_objectives"])
def test_retarget_fk_matches_commanded_mujoco_pose(pose, backend):
    model = mujoco.MjModel.from_xml_path(str(ROOT / "orca_am200/robot_model/am_arm200_orca_right.xml"))
    data = mujoco.MjData(model)
    mapping = CanonicalJointMap.load(ROOT / "orca_am200/robot_model/joint_map.json", model)
    r = Retargeter.from_paths(
        model_path=str(ROOT / "configs/orcahand-right-v1-feetech/config.yaml"),
        urdf_path=str(ROOT / "orcahand_description/v1/models/urdf/orcahand_right.urdf"),
        backend=backend,
        config_path=str(ROOT / ("configs/key_objectives_orca_v1.yaml" if backend == "key_objectives"
                                else "configs/adaptive_analytical_orca_v1.yaml")),
    )
    angles = {j: 0.0 for j in mapping.hand_joint_ids}
    if pose == "neutral":
        angles.update(r._hand.config.neutral_position)
    elif pose == "bent":
        angles.update(thumb_mcp=15, thumb_abd=10, thumb_pip=40, thumb_dip=30,
                      index_mcp=35, middle_mcp=25, ring_mcp=20, pinky_mcp=15)
    angles["wrist"] = 0.0
    # Compare poses within MuJoCo stops. Out-of-range live requests are clipped
    # by the production adapter, not used as evidence of FK agreement.
    refs = mapping.ref[mapping.hand_slice]
    values = np.clip(np.deg2rad([angles[j] for j in mapping.hand_joint_ids]),
                     mapping.lower[mapping.hand_slice] - refs,
                     mapping.upper[mapping.hand_slice] - refs)
    angles = dict(zip(mapping.hand_joint_ids, np.rad2deg(values)))
    adapter = HoldArmActionComposer(mapping)
    command = adapter(None, OrcaJointPositions(angles))
    np.testing.assert_allclose(command.position[mapping.hand_slice], values + refs, atol=1e-12)
    data.qpos[mapping.qpos_index] = command.position
    mujoco.mj_forward(model, data)
    names = ["right_palm"] + [f"right_{finger}_{link}" for finger in
             ("thumb", "index", "middle", "ring", "pinky") for link in ("mp", "ip")]
    names.append("right_thumb_dp")
    indices = [r._robot.model.getFrameId(name) for name in names]
    q = np.deg2rad([angles[j] for j in r._finger_joint_ids])
    xyz, _ = r._robot.compute_positions_and_jacobians(r._full_model_qpos(q), indices)
    palm = r._robot.data.oMf[indices[0]]
    sim_palm = model.body("right_palm").id
    sim_rotation = data.xmat[sim_palm].reshape(3, 3)
    for i, name in enumerate(names[1:], 1):
        sim = sim_rotation.T @ (data.xpos[model.body(name).id] - data.xpos[sim_palm])
        fk = palm.rotation.T @ (xyz[i] - xyz[0])
        np.testing.assert_allclose(fk, sim, atol=2e-6, err_msg=f"{pose}: {name}")
        fk_rotation = palm.rotation.T @ r._robot.data.oMf[indices[i]].rotation
        sim_body_rotation = sim_rotation.T @ data.xmat[model.body(name).id].reshape(3, 3)
        np.testing.assert_allclose(fk_rotation, sim_body_rotation, atol=2e-5,
                                   err_msg=f"{pose}: {name} rotation")
