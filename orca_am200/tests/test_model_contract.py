"""Contract tests for the combined AM-ARM200 + ORCA right hand v1 embodiment.

These guard the properties the teleop and dataset paths depend on: a
deterministic 20-DoF ordering, actuator limits that cannot command past a joint
stop, physically plausible dynamics, and hand joint names that the upstream
orca_teleop simulation sink can still map onto orca_core joint ids.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "robot_model" / "am_arm200_orca_right.xml"
JOINT_MAP_PATH = ROOT / "robot_model" / "joint_map.json"
HAND_CONFIG_PATH = ROOT.parent / "configs" / "orcahand-right-v1-feetech" / "config.yaml"

ARM_DOF = 3
HAND_DOF = 17
TOTAL_DOF = ARM_DOF + HAND_DOF


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(MODEL_PATH))


@pytest.fixture(scope="module")
def joint_map():
    return json.loads(JOINT_MAP_PATH.read_text(encoding="utf-8"))


def test_model_has_twenty_actuated_dof(model):
    assert model.nq == TOTAL_DOF
    assert model.nu == TOTAL_DOF
    assert model.njnt == TOTAL_DOF


def test_joint_map_covers_every_actuator_exactly_once(model, joint_map):
    entries = joint_map["canonical_order"]
    assert [entry["index"] for entry in entries] == list(range(TOTAL_DOF))
    assert sorted(entry["model_actuator_index"] for entry in entries) == list(range(model.nu))

    for entry in entries:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, entry["mujoco_joint"])
        assert joint_id >= 0, f"missing joint {entry['mujoco_joint']}"
        actuator_id = entry["model_actuator_index"]
        assert int(model.actuator_trnid[actuator_id, 0]) == joint_id
        assert (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            == entry["mujoco_actuator"]
        )


def test_arm_occupies_the_first_three_indices(joint_map):
    entries = joint_map["canonical_order"]
    assert [entry["role"] for entry in entries[:ARM_DOF]] == [
        "arm.shoulder_pan",
        "arm.shoulder_lift",
        "arm.elbow_flex",
    ]
    assert all(entry["role"].startswith("hand.") for entry in entries[ARM_DOF:])


def test_hand_order_matches_the_orca_core_hand_config(joint_map):
    """The hardware backend concatenates its hand vector without reordering."""
    config = yaml.safe_load(HAND_CONFIG_PATH.read_text(encoding="utf-8"))
    canonical_hand = [
        entry["orca_core_joint_id"] for entry in joint_map["canonical_order"][ARM_DOF:]
    ]
    assert canonical_hand == list(config["joint_ids"])
    assert len(canonical_hand) == HAND_DOF


def test_hand_joint_names_are_mappable_by_the_upstream_sim_sink(model, joint_map):
    """Upstream ``OrcaHandSimSink`` resolves actuators by joint name.

    It strips the ``right_`` prefix and translates the remainder to an orca_core
    joint id, so a naming drift in the combined model would break the sim path.
    """
    from orca_teleop.sim import _sim_joint_to_config_id

    config = yaml.safe_load(HAND_CONFIG_PATH.read_text(encoding="utf-8"))
    valid_ids = set(config["joint_ids"])

    resolved = {}
    for entry in joint_map["canonical_order"][ARM_DOF:]:
        core_name = entry["mujoco_joint"].removeprefix("right_")
        config_id = _sim_joint_to_config_id(core_name, valid_ids)
        assert config_id is not None, f"{entry['mujoco_joint']} has no orca_core counterpart"
        resolved[config_id] = entry["index"]

    assert set(resolved) == valid_ids, "hand actuators must be a 1:1 cover of the config"


def test_actuators_cannot_command_past_a_joint_stop(model):
    for actuator in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator, 0])
        joint_low, joint_high = model.jnt_range[joint_id]
        ctrl_low, ctrl_high = model.actuator_ctrlrange[actuator]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
        assert ctrl_low >= joint_low - 1e-9, f"{name} can command below its joint limit"
        assert ctrl_high <= joint_high + 1e-9, f"{name} can command above its joint limit"


def test_masses_and_inertias_are_physically_plausible(model):
    """Guards against the mm^2 -> m^2 inertia bug in the handoff model."""
    assert model.body_mass.sum() < 5.0, "combined robot should weigh a few kg, not tens"

    hand_mass = sum(
        model.body_mass[body]
        for body in range(model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "").startswith("right_")
    )
    assert 0.8 < hand_mass < 1.6, f"ORCA hand should weigh ~1.1 kg, got {hand_mass:.3f} kg"
    assert model.body_inertia.max() < 0.01, "no link of a 2 kg robot has inertia above 0.01 kg m^2"


def test_contact_excludes_survive_the_hand_attachment(model):
    assert model.nexclude >= 18, "the v1 hand's adjacent-link excludes must be preserved"


def test_no_self_collision_in_the_rest_pose(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    contacts = [
        (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1]),
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2]),
            float(c.dist),
        )
        for c in data.contact[: data.ncon]
    ]
    assert not contacts, f"rest pose is already in contact: {contacts}"


def test_hand_mounting_face_sits_on_the_flange(model):
    """Locks in the mount pose derived by scripts/derive_mount.py."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    flange = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "orca_flange")
    flange_rotation = data.xmat[flange].reshape(3, 3)
    flange_origin = data.xpos[flange]

    tower = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_tower")
    vertices = []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != tower or model.geom_dataid[geom] < 0:
            continue
        mesh = model.geom_dataid[geom]
        start = model.mesh_vertadr[mesh]
        local = model.mesh_vert[start : start + model.mesh_vertnum[mesh]].reshape(-1, 3)
        world = local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        vertices.append((world - flange_origin) @ flange_rotation)
    tower_vertices = np.vstack(vertices)

    lateral = (tower_vertices[:, :2].min(axis=0) + tower_vertices[:, :2].max(axis=0)) / 2
    assert np.linalg.norm(lateral) < 0.005, (
        f"tower is {np.linalg.norm(lateral) * 1000:.1f} mm off the flange axis"
    )
    face = tower_vertices[:, 2].min()
    assert abs(face) < 0.005, f"tower face is {face * 1000:.1f} mm off the flange plane"
    assert tower_vertices[:, 2].max() > 0.1, (
        "the hand must extend away from the flange, not into it"
    )


def test_palm_faces_down_at_the_rest_pose(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    anchors = {}
    for finger in ("index", "middle", "pinky"):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"right_{finger}_mcp")
        anchors[finger] = data.xanchor[joint].copy()
    tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_middle_ip")

    along = data.xpos[tip] - anchors["middle"]
    across = anchors["index"] - anchors["pinky"]
    normal = np.cross(across / np.linalg.norm(across), along / np.linalg.norm(along))
    normal /= np.linalg.norm(normal)

    tilt_deg = np.degrees(np.arccos(np.clip(np.dot(normal, [0.0, 0.0, -1.0]), -1.0, 1.0)))
    assert tilt_deg < 10.0, f"palm is {tilt_deg:.1f} deg away from facing down"


def test_holding_the_rest_pose_under_gravity_is_stable(model):
    data = mujoco.MjData(model)
    data.ctrl[:] = 0.0
    for _ in range(int(2.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    assert not np.any(data.warning.number), f"MuJoCo warnings: {list(data.warning.number)}"
    assert np.all(np.isfinite(data.qpos))
    arm_error_deg = np.rad2deg(np.abs(data.qpos[:ARM_DOF])).max()
    assert arm_error_deg < 3.0, f"arm sags {arm_error_deg:.1f} deg while holding its rest pose"
    assert np.abs(data.qvel).max() < 0.5, "pose should have settled after 2 s"
