#!/usr/bin/env python3
"""Hardware teleop — same MediaPipe/retarget path as teleop_sim.py.

No flexion/chirality hacks: MuJoCo already proved the retargeter output. If the
physical hand disagrees, fix ``calibration.yaml`` motor limits/signs instead.
"""
from __future__ import annotations

import argparse
import os

from orca_teleop.pipeline import OrcaHandSink, run_local


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default=os.environ.get("ORCA_MODEL_PATH"))
    parser.add_argument("--urdf_path", default=os.environ.get("ORCA_URDF_PATH"))
    parser.add_argument(
        "--retarget-config",
        default=os.environ.get("ORCA_RETARGET_CONFIG"),
    )
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    parser.add_argument("--show-video", action="store_true")
    parser.add_argument("--no-mirror", dest="mirror", action="store_false")
    parser.set_defaults(mirror=True)
    parser.add_argument("--camera-id", type=int, default=None)
    parser.add_argument("--confidence", type=float, default=0.7)
    args = parser.parse_args()

    run_local(
        model_path=args.model_path,
        urdf_path=args.urdf_path,
        handedness=args.hand,
        confidence=args.confidence,
        show_video=args.show_video,
        sink=OrcaHandSink(args.model_path),
        retargeter_config_path=args.retarget_config,
        camera_id=args.camera_id,
        mirror=args.mirror,
    )


if __name__ == "__main__":
    main()
