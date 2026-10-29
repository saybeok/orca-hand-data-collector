"""Inspect the combined AM-ARM200 + ORCA right hand model.

Offscreen renders a few poses by default, or opens the interactive viewer with
``--interactive``. Poses are addressed by canonical joint role so the ordering in
joint_map.json is what gets exercised, not raw ctrl indices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "robot_model" / "am_arm200_orca_right.xml"
JOINT_MAP_PATH = ROOT / "robot_model" / "joint_map.json"
RENDER_DIR = ROOT / "renders"

POSES: dict[str, dict[str, float]] = {
    "home": {},
    "arm_inspect": {
        "arm.shoulder_lift": -0.70,
        "arm.elbow_flex": 1.20,
    },
    "hand_fist": {
        f"hand.{finger}_{segment}": 1.3
        for finger in ("index", "middle", "ring", "pinky")
        for segment in ("mcp", "pip")
    }
    | {"hand.thumb_pip": 1.0, "hand.thumb_dip": 1.0, "hand.thumb_abd": -0.9},
    "arm_reach_and_pinch": {
        "arm.shoulder_pan": 0.5,
        "arm.shoulder_lift": -0.9,
        "arm.elbow_flex": 1.5,
        "hand.index_mcp": 1.0,
        "hand.index_pip": 0.6,
        "hand.thumb_pip": 0.9,
        "hand.thumb_dip": 0.5,
        "hand.thumb_abd": -0.6,
    },
}


def canonical_indices() -> dict[str, int]:
    joint_map = json.loads(JOINT_MAP_PATH.read_text(encoding="utf-8"))
    return {entry["role"]: entry["model_actuator_index"] for entry in joint_map["canonical_order"]}


def apply_pose(model, data, roles: dict[str, int], pose: dict[str, float]) -> None:
    mujoco.mj_resetData(model, data)
    for role, value in pose.items():
        actuator = roles[role]
        joint_id = int(model.actuator_trnid[actuator, 0])
        low, high = model.jnt_range[joint_id]
        clamped = float(np.clip(value, low, high))
        data.ctrl[actuator] = clamped
        data.qpos[model.jnt_qposadr[joint_id]] = clamped
    mujoco.mj_forward(model, data)


def framing_camera(model, data, azimuth: float, elevation: float):
    """Free camera that keeps the whole robot in frame for any pose."""
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    positions = data.xpos[1:]  # skip the world body
    center = (positions.min(axis=0) + positions.max(axis=0)) / 2
    camera.lookat[:] = center
    camera.distance = 2.2 * max(np.ptp(positions, axis=0).max(), 0.3)
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


def render(model, data, width: int, height: int, camera) -> np.ndarray:
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=camera)
        return renderer.render()


def describe_contacts(model, data) -> str:
    def body_of(geom: int) -> str:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom])

    pairs = [
        f"{body_of(contact.geom1)}/{body_of(contact.geom2)} {contact.dist * 1000:.1f}mm"
        for contact in data.contact[: data.ncon]
    ]
    return ", ".join(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactive", action="store_true", help="open the MuJoCo viewer")
    parser.add_argument("--pose", default="home", choices=sorted(POSES), help="pose to view")
    parser.add_argument("--camera", default="free", help='"free" auto-frames, or a camera name')
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-20.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    roles = canonical_indices()

    if args.interactive:
        from mujoco import viewer

        apply_pose(model, data, roles, POSES[args.pose])
        viewer.launch(model, data)
        return

    from PIL import Image

    RENDER_DIR.mkdir(exist_ok=True)
    for name, pose in POSES.items():
        apply_pose(model, data, roles, pose)
        if args.camera == "free":
            camera = framing_camera(model, data, args.azimuth, args.elevation)
        else:
            camera = args.camera
        pixels = render(model, data, args.width, args.height, camera)
        out = RENDER_DIR / f"{name}.png"
        Image.fromarray(pixels).save(out)
        contacts = describe_contacts(model, data)
        print(f"{name:20s} ncon={data.ncon:2d}  {contacts:60s} -> {out.name}")


if __name__ == "__main__":
    main()
