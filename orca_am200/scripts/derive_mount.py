"""Derive the ``orca_flange`` -> ORCA v1 hand mount transform.

The handoff model (``handoff/scene_v4_original.xml``) carries the *v2* hand export,
which matches neither the physical v1 Feetech hand nor the joint names that
orca_teleop can map, so the combined model attaches the v1 hand instead. That
needs a mount pose, and the two exports place their body frames differently, so
the pose is derived from physical features rather than copied.

Two facts drive it:

  * The v1 hand model is authored with its tower base on the ``z = 0`` plane of
    the attach frame, with the hand extending along ``+z``. That plane is the
    face that bolts to the flange.
  * The handoff flange has the hand extending along flange ``+z`` (its v2 subtree
    puts the palm at ``(0, 0, 0.1498)``), and its author's stated intent was a
    neutral palm-down work orientation.

So the mount aligns the tower base face with the flange face, the hand axis with
flange ``+z``, and rolls about that axis until the palm faces world-down with the
arm at its rest pose.

The script also verifies the v1 <-> v2 joint correspondence used elsewhere and
reports the per-joint axis sign relation, which the retargeting path needs.
"""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HANDOFF_SCENE = ROOT / "handoff" / "scene_v4_original.xml"
HAND_MODEL = ROOT / "robot_model" / "orcahand_right_v1.xml"

FLANGE_BODY = "orca_flange"
V1_ROOT_BODY = "right_tower"
PROBE_MOUNT_BODY = "mount_probe_root"

# v1 orca_core joint id -> raw v2 joint name in the handoff scene.
#
# Recovered from the tree topology and joint axes: the handoff hand orders its
# subtrees pinky, ring, middle, index, thumb (decreasing x in the palm frame,
# matching the v1 layout), and the four thumb axes reproduce the v1 thumb chain
# exactly. The v2 export reuses the "M-" (middle) part names for the ring finger.
V1_TO_V2_JOINTS = {
    "wrist": "orcahand_right_R-Carpals_8d1f1041_to_TopTower-Model_4a80d30e",
    "pinky_abd": "orcahand_right_P-AP_f5e42b61_to_R-Carpals_8d1f1041",
    "pinky_mcp": "orcahand_right_P-PP_1d411b9b_to_P-AP_f5e42b61",
    "pinky_pip": "orcahand_right_P-FingerTipAssembly_cd219176_to_P-PP_1d411b9b",
    "ring_abd": "orcahand_right_M-AP_6ec59111_to_R-Carpals_8d1f1041",
    "ring_mcp": "orcahand_right_M-PP_8660a1eb_to_M-AP_6ec59111",
    "ring_pip": "orcahand_right_M-FingerTipAssembly_424a8e75_to_M-PP_8660a1eb",
    "middle_abd": "orcahand_right_M-AP_e04a96f2_to_R-Carpals_8d1f1041",
    "middle_mcp": "orcahand_right_M-PP_08efa608_to_M-AP_e04a96f2",
    "middle_pip": "orcahand_right_M-FingerTipAssembly_34afb748_to_M-PP_08efa608",
    "index_abd": "orcahand_right_I-AP-R_d95d02d1_to_R-Carpals_8d1f1041",
    "index_mcp": "orcahand_right_I-PP_bacbd481_to_I-AP-R_d95d02d1",
    "index_pip": "orcahand_right_I-FingerTipAssembly_ec49c16c_to_I-PP_bacbd481",
    "thumb_mcp": "orcahand_right_T-TP-R_1c2b802d_to_R-Carpals_8d1f1041",
    "thumb_abd": "orcahand_right_R-T-AP_a9723101_to_T-TP-R_1c2b802d",
    "thumb_pip": "orcahand_right_T-PP_68395e98_to_R-T-AP_a9723101",
    "thumb_dip": "orcahand_right_T-DP_b7429e50_to_T-PP_68395e98",
}

PROBE_XML = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="mount_probe">
  <compiler angle="radian" autolimits="true"/>
  <asset>
    <model name="orcahand_right" file="{hand_model}"/>
  </asset>
  <worldbody>
    <body name="{mount_body}">
      <attach model="orcahand_right" body="{root_body}" prefix=""/>
    </body>
  </worldbody>
</mujoco>
"""


def strip_geometry(model_path: Path) -> str:
    """Drop geoms and mesh assets so the frames compile without the v2 STLs.

    Every body in the handoff scene carries an explicit ``<inertial>``, so the
    kinematic tree survives intact.
    """
    root = ET.parse(model_path).getroot()
    for parent in root.iter():
        for child in list(parent):
            if child.tag in ("geom", "mesh", "asset"):
                parent.remove(child)
    return ET.tostring(root, encoding="unicode")


def load(model_path: Path):
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
    except ValueError:
        model = mujoco.MjModel.from_xml_string(strip_geometry(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def body_rotation(model, data, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise KeyError(f"body {body_name!r} not found")
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, data.xquat[body_id])
    return rotation.reshape(3, 3), data.xpos[body_id].copy()


def joint_frames(model, data, rotation, origin, joint_names):
    anchors = np.empty((len(joint_names), 3))
    axes = np.empty((len(joint_names), 3))
    for row, name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise KeyError(f"joint {name!r} not found")
        anchors[row] = rotation.T @ (data.xanchor[joint_id] - origin)
        axes[row] = rotation.T @ data.xaxis[joint_id]
    return anchors, axes


def body_vertices(model, data, body_name: str) -> np.ndarray:
    """World-frame mesh vertices of every geom on ``body_name``."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    chunks = []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != body_id or model.geom_dataid[geom] < 0:
            continue
        mesh = model.geom_dataid[geom]
        start = model.mesh_vertadr[mesh]
        verts = model.mesh_vert[start : start + model.mesh_vertnum[mesh]].reshape(-1, 3)
        chunks.append(verts @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom])
    if not chunks:
        raise ValueError(f"body {body_name!r} has no mesh geoms")
    return np.vstack(chunks)


def palm_normal(model, data, rotation, origin) -> np.ndarray:
    """Palm-facing unit normal, from the finger MCP anchors and the middle finger."""
    anchors = {}
    for finger in ("index", "middle", "pinky"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"right_{finger}_mcp")
        anchors[finger] = rotation.T @ (data.xanchor[joint_id] - origin)
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_middle_ip")
    tip = rotation.T @ (data.xpos[tip_id] - origin)

    along = tip - anchors["middle"]
    across = anchors["index"] - anchors["pinky"]
    normal = np.cross(across / np.linalg.norm(across), along / np.linalg.norm(along))
    return normal / np.linalg.norm(normal)


def v2_palm_normal(model, data, rotation, origin) -> np.ndarray:
    anchors = {}
    for finger in ("index", "middle", "pinky"):
        name = V1_TO_V2_JOINTS[f"{finger}_mcp"]
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        anchors[finger] = rotation.T @ (data.xanchor[joint_id] - origin)
    pip_name = V1_TO_V2_JOINTS["middle_pip"]
    pip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, pip_name)
    tip = rotation.T @ (data.xanchor[pip_id] - origin)

    along = tip - anchors["middle"]
    across = anchors["index"] - anchors["pinky"]
    normal = np.cross(across / np.linalg.norm(across), along / np.linalg.norm(along))
    return normal / np.linalg.norm(normal)


def main() -> None:
    v1_ids = list(V1_TO_V2_JOINTS)

    handoff_model, handoff_data = load(HANDOFF_SCENE)
    flange_rotation, flange_origin = body_rotation(handoff_model, handoff_data, FLANGE_BODY)
    v2_anchors, v2_axes = joint_frames(
        handoff_model,
        handoff_data,
        flange_rotation,
        flange_origin,
        [V1_TO_V2_JOINTS[j] for j in v1_ids],
    )

    with tempfile.TemporaryDirectory() as tmp:
        probe_path = Path(tmp) / "probe.xml"
        probe_path.write_text(
            PROBE_XML.format(
                hand_model=HAND_MODEL, root_body=V1_ROOT_BODY, mount_body=PROBE_MOUNT_BODY
            ),
            encoding="utf-8",
        )
        probe_model, probe_data = load(probe_path)
        mount_rotation, mount_origin = body_rotation(probe_model, probe_data, PROBE_MOUNT_BODY)
        v1_anchors, v1_axes = joint_frames(
            probe_model, probe_data, mount_rotation, mount_origin, [f"right_{j}" for j in v1_ids]
        )
        tower = body_vertices(probe_model, probe_data, V1_ROOT_BODY)
        v1_palm_normal = palm_normal(probe_model, probe_data, mount_rotation, mount_origin)

    # The tower's mounting face. The v1 model puts it on z = 0 of the attach frame.
    base_center = np.array(
        [
            (tower[:, 0].min() + tower[:, 0].max()) / 2,
            (tower[:, 1].min() + tower[:, 1].max()) / 2,
            tower[:, 2].min(),
        ]
    )

    # Roll about the flange axis so the palm faces world-down at the arm's rest pose.
    down_in_flange = flange_rotation.T @ np.array([0.0, 0.0, -1.0])
    target_normal = down_in_flange - down_in_flange[2] * np.array([0.0, 0.0, 1.0])
    off_axis_deg = np.degrees(np.arcsin(abs(down_in_flange[2])))
    target_normal /= np.linalg.norm(target_normal)

    axis = np.array([0.0, 0.0, 1.0])
    rotation = np.column_stack([np.cross(target_normal, axis), target_normal, axis])
    # v1 palm normal is +y in the mount frame, so mapping y -> target_normal needs
    # no extra correction; assert it rather than silently assuming.
    assert np.dot(v1_palm_normal, [0.0, 1.0, 0.0]) > 0.99, v1_palm_normal
    translation = -rotation @ base_center

    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rotation).ravel())
    mujoco.mju_normalize4(quat)

    def fmt(values) -> str:
        return " ".join(f"{v:.9g}" for v in np.asarray(values))

    print("v1 hand in the attach frame")
    print(f"  tower AABB      min {fmt(tower.min(0))}  max {fmt(tower.max(0))}")
    print(f"  mounting face   z = {tower[:, 2].min() * 1000:.2f} mm, center {fmt(base_center)}")
    print(f"  palm normal     {fmt(v1_palm_normal)}")
    print()
    print("flange frame")
    print(f"  world-down in flange coords {fmt(down_in_flange)}")
    print(f"  flange axis is {off_axis_deg:.2f} deg off horizontal at the arm rest pose")
    print(f"  chosen palm normal          {fmt(target_normal)}")
    print()
    print("MOUNT BODY POSE, relative to orca_flange")
    print(f'  pos="{fmt(translation)}"')
    print(f'  quat="{fmt(quat)}"')
    print()

    placed_base = translation + rotation @ base_center
    placed_normal_world = flange_rotation @ rotation @ v1_palm_normal
    v2_normal_world = flange_rotation @ v2_palm_normal(
        handoff_model, handoff_data, flange_rotation, flange_origin
    )
    print("verification")
    print(f"  tower face center lands at flange {fmt(placed_base)} (want 0 0 0)")
    print(f"  palm normal in world              {fmt(placed_normal_world)}")
    print(f"  handoff v2 palm normal in world   {fmt(v2_normal_world)}")
    agreement_deg = np.degrees(
        np.arccos(np.clip(np.dot(placed_normal_world, v2_normal_world), -1, 1))
    )
    print(f"  agreement with the handoff pose   {agreement_deg:.1f} deg")
    print()

    report_correspondence(v1_anchors, v1_axes, v2_anchors, v2_axes, v1_ids)


def report_correspondence(v1_anchors, v1_axes, v2_anchors, v2_axes, v1_ids) -> None:
    """Check the V1_TO_V2_JOINTS table independently of the chosen mount.

    Aligns the two hands by their joint anchors first, so the axis comparison
    measures whether the joints correspond rather than how the hand is mounted.
    A wrong pairing would show near-orthogonal axes and centimetres of anchor
    error; the residual that remains is the genuine v1/v2 design difference.
    """
    source = v1_anchors - v1_anchors.mean(axis=0)
    target = v2_anchors - v2_anchors.mean(axis=0)
    u, _, vt = np.linalg.svd(source.T @ target)
    correction = np.eye(3)
    if np.linalg.det(vt.T @ u.T) < 0:
        correction[2, 2] = -1.0
    alignment = vt.T @ correction @ u.T
    errors = np.linalg.norm(source @ alignment.T - target, axis=1)

    print("v1 <-> v2 joint correspondence check (anchor-aligned, mount-independent)")
    print(f"  anchor residual rms={errors.mean() * 1000:.2f} mm  max={errors.max() * 1000:.2f} mm")
    print(f"  {'joint':12s} {'anchor err':>11s} {'axis dot':>9s}  verdict")
    for row, joint_id in enumerate(v1_ids):
        dot = float(np.dot(alignment @ v1_axes[row], v2_axes[row]))
        verdict = "ok" if abs(dot) > 0.9 else "SUSPECT"
        sign = "same sign" if dot > 0 else "sign flipped"
        print(f"  {joint_id:12s} {errors[row] * 1000:8.2f} mm {dot:9.4f}  {verdict}, {sign}")


if __name__ == "__main__":
    main()
