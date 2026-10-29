"""Generate the canonical 20-DoF joint map for the combined embodiment.

Indices 0..2 are the AM-ARM200 arm. Indices 3..19 are the ORCA right-hand joints
in the order the orca_core hand config declares them, so the hardware backend can
concatenate its measured/commanded vector without reordering. The simulation
backend needs a permutation instead, which is emitted here as
``model_actuator_index`` rather than being recomputed at runtime.

Regenerate with:
    python scripts/build_joint_map.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mujoco
import yaml

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "robot_model" / "am_arm200_orca_right.xml"
JOINT_MAP = ROOT / "robot_model" / "joint_map.json"
DEFAULT_HAND_CONFIG = ROOT.parent / "configs" / "orcahand-right-v1-feetech" / "config.yaml"

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex")
HAND_JOINT_PREFIX = "right_"


def load_hand_joint_ids(config_path: Path) -> list[str]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    joint_ids = config.get("joint_ids")
    if not joint_ids:
        raise ValueError(f"{config_path} declares no joint_ids")
    return list(joint_ids)


def build(config_path: Path) -> dict:
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    hand_joint_ids = load_hand_joint_ids(config_path)

    entries = []
    for index, name in enumerate(ARM_JOINTS + tuple(hand_joint_ids)):
        is_arm = index < len(ARM_JOINTS)
        mj_joint = name if is_arm else f"{HAND_JOINT_PREFIX}{name}"
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mj_joint)
        if joint_id < 0:
            raise KeyError(f"model has no joint {mj_joint!r}")

        actuator_id = next(
            (a for a in range(model.nu) if int(model.actuator_trnid[a, 0]) == joint_id), None
        )
        if actuator_id is None:
            raise KeyError(f"joint {mj_joint!r} has no actuator")

        low, high = model.jnt_range[joint_id]
        entries.append(
            {
                "index": index,
                "role": f"arm.{name}" if is_arm else f"hand.{name}",
                "orca_core_joint_id": None if is_arm else name,
                "mujoco_joint": mj_joint,
                "mujoco_actuator": mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
                ),
                "model_actuator_index": actuator_id,
                "model_qpos_index": int(model.jnt_qposadr[joint_id]),
                "range_rad": [round(float(low), 6), round(float(high), 6)],
            }
        )

    if len(entries) != model.nu:
        raise ValueError(f"canonical order has {len(entries)} entries but model has {model.nu}")
    if len({entry["model_actuator_index"] for entry in entries}) != model.nu:
        raise ValueError("canonical order does not cover every model actuator exactly once")

    return {
        "robot": "AM-ARM200 3DoF + ORCA right hand v1",
        "total_dof": len(entries),
        "arm_dof": len(ARM_JOINTS),
        "hand_dof": len(hand_joint_ids),
        "model": MODEL.name,
        "hand_config": str(config_path),
        "hand_order_source": (
            "orca_core hand config joint_ids, so the hardware vector needs no reordering"
        ),
        "canonical_order": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hand-config",
        type=Path,
        default=Path(os.environ.get("ORCA_MODEL_PATH", DEFAULT_HAND_CONFIG)),
        help="orca_core hand config.yaml that defines the canonical hand joint order",
    )
    args = parser.parse_args()

    joint_map = build(args.hand_config)
    JOINT_MAP.write_text(json.dumps(joint_map, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {JOINT_MAP.relative_to(ROOT)}  ({joint_map['total_dof']} DoF)")
    for entry in joint_map["canonical_order"]:
        print(
            f"  {entry['index']:2d} {entry['role']:20s} "
            f"mj={entry['mujoco_joint']:20s} act_idx={entry['model_actuator_index']:2d}"
        )


if __name__ == "__main__":
    main()
