#!/usr/bin/env python3
"""Quest WebXR -> retargeter -> physical ORCA hand."""
from __future__ import annotations

import argparse
import os

from orca_teleop.pipeline import OrcaHandSink, run_metaquest_local


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default=os.environ.get("ORCA_MODEL_PATH"))
    parser.add_argument("--urdf_path", default=os.environ.get("ORCA_URDF_PATH"))
    parser.add_argument(
        "--retarget-config",
        default=os.environ.get("ORCA_RETARGET_CONFIG"),
    )
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    parser.add_argument("--quest-host", default="0.0.0.0")
    parser.add_argument("--quest-port", type=int, default=8765)
    parser.add_argument("--quest-fps", type=int, default=30)
    args = parser.parse_args()

    run_metaquest_local(
        model_path=args.model_path,
        urdf_path=args.urdf_path,
        handedness=args.hand,
        quest_host=args.quest_host,
        quest_port=args.quest_port,
        quest_fps=args.quest_fps,
        sink=OrcaHandSink(args.model_path),
        retargeter_config_path=args.retarget_config,
    )


if __name__ == "__main__":
    main()
