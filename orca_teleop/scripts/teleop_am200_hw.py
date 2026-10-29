"""Teleoperate the physical AM-ARM200 + ORCA hand (IK arm + retargeted hand).

Same ingress as ``teleop_am200_sim.py``, but drives Feetech buses:
  hand → ``$ORCA_MODEL_PATH`` port (usually /dev/ttyACM0)
  arm  → ``configs/am-arm200-3dof/config.yaml`` port (/dev/ttyACM1)

Quest provides wrist pose for arm IK. Webcam has no wrist pose, so the arm
holds the connect-time pose and only the hand follows.

Examples:
    python scripts/teleop_am200_hw.py --source quest --clutch --arm-scale 0.35 --arm-max-step 0.3
    python scripts/teleop_am200_hw.py --source webcam --show-video --no-arm --no-mirror --camera-id 0
"""

from __future__ import annotations

import argparse
import logging
import math
import os

from orca_teleop.am200 import (
    Am200ActionComposer,
    Am200HardwareSink,
    ArmMappingConfig,
    OperatorCalibration,
)
from orca_teleop.am200.arm_client import default_arm_config_path
from orca_teleop.ingress.server import DEFAULT_PORT
from orca_teleop.pipeline import run, run_local, run_metaquest_local


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="quest",
        choices=["quest", "webcam", "remote"],
    )
    parser.add_argument("--model-path", default=None, help="combined MJCF for IK")
    parser.add_argument("--hand-config", default=os.environ.get("ORCA_MODEL_PATH"))
    parser.add_argument("--arm-config", default=str(default_arm_config_path()))
    parser.add_argument("--urdf_path", default=os.environ.get("ORCA_URDF_PATH"))
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    parser.add_argument("--quest-host", default="0.0.0.0")
    parser.add_argument("--quest-port", type=int, default=8765)
    parser.add_argument("--quest-fps", type=int, default=30)
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--show-video", action="store_true")
    parser.add_argument("--visualize-landmarks", action="store_true",
                        help="show the incoming hand skeleton and thumb bend angles before retargeting")
    parser.add_argument("--camera-id", type=int, default=None)
    parser.add_argument("--no-mirror", dest="mirror", action="store_false")
    parser.set_defaults(mirror=True)
    parser.add_argument(
        "--retargeter",
        default="adaptive_analytical",
        choices=["rmsprop", "adaptive_analytical", "key_objectives", "dexpilot"],
    )
    parser.add_argument("--retarget-config", default=os.environ.get("ORCA_RETARGET_CONFIG"))
    parser.add_argument("--arm-scale", type=float, default=0.35)
    parser.add_argument('--arm-max-velocity-deg-s', type=float, default=None)
    parser.add_argument('--arm-max-acceleration-deg-s2', type=float, default=None)
    parser.add_argument(
        "--arm-max-step",
        type=float,
        default=None,
        help="clip every arm joint to +/- this many rad from the connect-time pose",
    )
    parser.add_argument(
        "--no-arm",
        action="store_true",
        help="do not open the arm bus; hand-only teleop",
    )
    parser.add_argument("--clutch", action="store_true")
    parser.add_argument("--hand-feedback-debug", action="store_true",
                        help="log all finger commands vs motor-encoder joint estimates once per second")
    parser.add_argument("--operator-cal", default=None)
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.hand_config:
        parser.error("--hand-config or $ORCA_MODEL_PATH is required")

    sink = Am200HardwareSink(
        hand_config_path=args.hand_config,
        arm_config_path=args.arm_config,
        model_path=args.model_path,
        enable_arm=not args.no_arm,
        enable_hand=True,
        arm_max_step_rad=args.arm_max_step,
        hand_feedback_debug=args.hand_feedback_debug,
    )

    composer = None
    if not args.no_arm:
        composer = Am200ActionComposer.from_model_path(
            sink.model_path,
            config=ArmMappingConfig(
                scale=args.arm_scale,
                max_velocity_rad_s=(None if args.arm_max_velocity_deg_s is None
                                    else math.radians(args.arm_max_velocity_deg_s)),
                max_acceleration_rad_s2=(None if args.arm_max_acceleration_deg_s2 is None
                                         else math.radians(args.arm_max_acceleration_deg_s2)),
            ),
            calibration=OperatorCalibration.load(args.operator_cal),
        )

    common = dict(
        model_path=args.hand_config,
        urdf_path=args.urdf_path,
        port=args.port,
        sink=sink,
        retargeter_backend=args.retargeter,
        retargeter_config_path=args.retarget_config,
        action_composer=composer,
        visualize_landmarks=args.visualize_landmarks,
    )

    if args.source == "quest":
        run_metaquest_local(
            handedness=args.hand,
            quest_host=args.quest_host,
            quest_port=args.quest_port,
            quest_fps=args.quest_fps,
            clutch_enabled=args.clutch,
            **common,
        )
    elif args.source == "webcam":
        run_local(
            handedness=args.hand,
            confidence=args.confidence,
            show_video=args.show_video,
            camera_id=args.camera_id,
            mirror=args.mirror,
            **common,
        )
    else:
        run(**common)


if __name__ == "__main__":
    main()
