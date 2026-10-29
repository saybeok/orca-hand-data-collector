"""Teleoperate the combined AM-ARM200 + ORCA right hand in simulation.

Same ingress and retargeter stack as the hand-only path. The retargeter solves
the 17 hand joints; the arm's 3 come from position IK on the operator's wrist,
carried by ``HandFrame.wrist_transform``. Publishers without a wrist pose (the
webcam one) drive the hand alone and the arm holds still.

The arm mapping is incremental: wherever your wrist is on the first frame
becomes the anchor, and the flange follows your displacement from it, scaled by
``--arm-scale``. Three joints can only track position, not wrist rotation.

Examples:
    # Quest 3S, one command. Open the printed URL in the Quest browser.
    python scripts/teleop_am200_sim.py --source quest

    # Webcam instead of the headset.
    python scripts/teleop_am200_sim.py --source webcam --show-video

    # Wait for a publisher on another machine.
    python scripts/teleop_am200_sim.py --source remote
"""

import argparse
import logging
import os

from orca_teleop.am200 import (
    Am200ActionComposer,
    Am200SimSink,
    ArmMappingConfig,
    OperatorCalibration,
)
from orca_teleop.ingress.server import DEFAULT_PORT
from orca_teleop.am200.tasks import TASKS
from orca_teleop.pipeline import run, run_local, run_metaquest_local


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="quest",
        choices=["quest", "webcam", "remote"],
        help="where hand landmarks come from (default: quest)",
    )
    parser.add_argument("--model-path", default=None, help="combined MJCF; defaults to orca_am200")
    parser.add_argument(
        "--task", choices=sorted(TASKS), default=None,
        help="load an interactive manipulation scene (hwatu_mat: card on soft mat)",
    )
    parser.add_argument(
        "--respawn-after", type=float, default=None, metavar="SECONDS",
        help="respawn the task object after success/drop; omit to keep the result",
    )
    parser.add_argument(
        "--hand-config",
        default=os.environ.get("ORCA_MODEL_PATH"),
        help="orca_core hand config.yaml (default: $ORCA_MODEL_PATH)",
    )
    parser.add_argument("--urdf_path", default=None, help="hand URDF for the retargeter")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="gRPC ingress port")
    parser.add_argument(
        "--render-mode",
        default="human",
        choices=["human", "rgb_array"],
        help="'human' opens the MuJoCo viewer (default)",
    )
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    parser.add_argument("--quest-host", default="0.0.0.0")
    parser.add_argument("--quest-port", type=int, default=8765)
    parser.add_argument("--quest-fps", type=int, default=30)
    parser.add_argument("--confidence", type=float, default=0.7, help="MediaPipe confidence")
    parser.add_argument("--show-video", action="store_true", help="show the webcam feed")
    parser.add_argument(
        "--retargeter",
        default="adaptive_analytical",
        choices=["rmsprop", "adaptive_analytical", "key_objectives", "dexpilot"],
    )
    parser.add_argument("--retarget-config", default=os.environ.get("ORCA_RETARGET_CONFIG"))
    parser.add_argument(
        "--arm-scale",
        type=float,
        default=0.35,
        help="metres of flange motion per metre of wrist motion (default: 0.35, same as hardware)",
    )
    parser.add_argument(
        "--arm-max-step", type=float, default=None,
        help="same as hardware: limit arm position to +/- RAD from start, not speed",
    )
    parser.add_argument(
        "--no-arm",
        action="store_true",
        help="ignore the wrist pose and hold the arm still, driving the hand only",
    )
    parser.add_argument(
        "--clutch",
        action="store_true",
        help="pinch the OTHER hand to let the arm follow; release to park it (Quest only)",
    )
    parser.add_argument(
        "--operator-cal",
        default=None,
        help="operator calibration yaml (default: $ORCA_OPERATOR_CALIBRATION or configs/)",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()
    if args.arm_max_step is not None and not (0 < args.arm_max_step < float("inf")):
        parser.error("--arm-max-step must be finite and positive")
    if args.respawn_after is not None:
        if args.task is None or not (0 < args.respawn_after < float("inf")):
            parser.error("--respawn-after requires --task and a finite positive duration")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    sink = Am200SimSink(
        model_path=args.model_path,
        hand_config_path=args.hand_config,
        render_mode=args.render_mode,
        task=args.task,
        respawn_after_s=args.respawn_after,
        arm_max_step_rad=args.arm_max_step,
        arm_damping_ratio=1.0,
    )

    composer = None
    if not args.no_arm:
        composer = Am200ActionComposer.from_model_path(
            sink.model_path,
            config=ArmMappingConfig(scale=args.arm_scale),
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
            **common,
        )
    else:
        run(**common)


if __name__ == "__main__":
    main()
