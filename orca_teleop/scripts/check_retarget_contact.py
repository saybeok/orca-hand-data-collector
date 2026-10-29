"""Inspect collision-surface clearance in a saved retarget command trace.

This prescribes joint poses (mj_forward), not a dynamic grasp rollout. Negative
clearance means collision geometry overlap in the requested pose. No object is
attached, moved or lifted, and no grasp-success claim is made.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import mujoco
import numpy as np
from orca_core import OrcaHand, OrcaJointPositions

from orca_teleop.am200.canonical import CanonicalJointMap
from orca_teleop.am200.sim_env import DEFAULT_MODEL_PATH


def distal_geoms(model, body_name, skin_only=False):
    body = model.body(body_name).id
    geoms = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body or not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        mesh = int(model.geom_dataid[g])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh) if mesh >= 0 else ""
        if not skin_only or "skin" in (name or ""):
            geoms.append(g)
    if not geoms:
        raise ValueError(f"No collision {'skin ' if skin_only else ''}geoms on {body_name}")
    return geoms


def clearance(model, data, first, second):
    return min(mujoco.mj_geomDistance(model, data, a, b, 1.0, None) for a in first for b in second)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="NPZ from eval_retarget_pinch --output-dir")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--hand-config", default=os.environ.get("ORCA_MODEL_PATH"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/eval/pinch_contact"))
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.model_path))
    data = mujoco.MjData(model)
    mapping = CanonicalJointMap.load(args.model_path.parent / "joint_map.json", model)
    with np.load(args.trace) as trace:
        q = trace["q_rad"]
        human = trace["human"][:, 0]
        frames = trace["frame_index"]
        if "joint_ids" in trace:
            joints = trace["joint_ids"].tolist()
        else:
            if not args.hand_config:
                parser.error("Old trace lacks joint_ids; provide the original --hand-config")
            # Config parsing only; no hardware connection is made.
            joints = [j for j in OrcaHand(args.hand_config).config.joint_ids if j != "wrist"]
    if q.shape != (len(frames), len(joints)) or not len(q) or not np.isfinite(q).all():
        parser.error("Expected a nonempty finite q_rad trace matching joint_ids/frame_index")
    all_geoms = [distal_geoms(model, b) for b in ("right_thumb_dp", "right_index_ip")]
    pads = [distal_geoms(model, b, True) for b in ("right_thumb_dp", "right_index_ip")]
    rows = []
    clipped_joint_counts = np.zeros(mapping.hand_dof, dtype=np.int64)
    for frame, angles, operator_gap in zip(frames, q, human, strict=True):
        physical = OrcaJointPositions(dict(zip(joints, np.rad2deg(angles), strict=True)) | {"wrist": 0.0})
        raw = mapping.hand_from_physical(physical)
        clipped = np.clip(raw, mapping.lower[mapping.hand_slice], mapping.upper[mapping.hand_slice])
        changed = np.abs(raw - clipped) > 1e-8
        clipped_joint_counts += changed
        data.qpos[mapping.qpos_index[mapping.hand_slice]] = clipped
        mujoco.mj_forward(model, data)
        rows.append({
            "frame_index": int(frame), "operator_gap_cm": float(operator_gap),
            "distal_clearance_mm": 1000 * clearance(model, data, *all_geoms),
            "pad_clearance_mm": 1000 * clearance(model, data, *pads),
            "clipped_joints": int(np.count_nonzero(changed)),
        })
    pinch = [r for r in rows if r["operator_gap_cm"] < 3.0]
    summary = {
        "trace": str(args.trace), "model": str(args.model_path),
        "method": "Prescribed, clipped command poses; signed MuJoCo collision-geometry distance",
        "pinch_frames": len(pinch), "frames_with_joint_clipping": sum(r["clipped_joints"] > 0 for r in rows),
        "clipped_joint_counts": {
            joint: int(count) for joint, count in zip(mapping.hand_joint_ids, clipped_joint_counts, strict=True)
            if count
        },
        "pinch_pad_clearance_min_median_max_mm": (
            np.percentile([r["pad_clearance_mm"] for r in pinch], [0, 50, 100]).tolist() if pinch else None
        ),
        "pinch_pad_overlap_frames": sum(r["pad_clearance_mm"] < 0 for r in pinch),
        "limitations": ["Convex collision meshes approximate the finger pads.",
                        "Overlap is an infeasible requested pose, not evidence of stable contact.",
                        "This does not test object retention, table extraction, or real hardware."],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "clearance.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
