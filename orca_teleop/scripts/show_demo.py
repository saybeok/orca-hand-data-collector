"""Run the AM-ARM200 audience show: greeting and dexterous hand flourishes.

The arm starts from its connect-time pose.  Before hardware execution, place it
in a repeatable neutral pose with enough clearance in every direction.  The
script raises the arm with its palm toward the audience, waves left/right,
plays the official ORCA open/grasp/pinch/abduction demo poses plus a travelling
finger wave, then returns to the starting pose before torque is disabled.

Preview without hardware:

    python scripts/show_demo.py --backend sim --fast --video ../media/show_demo.mp4

Run the physical robot (explicit opt-in required):

    python scripts/show_demo.py --backend hardware --execute --cycles 1
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from orca_teleop.am200 import Am200HardwareSink, Am200SimSink, RobotJointCommand, SimCameraConfig
from orca_teleop.am200.arm_client import default_arm_config_path
from orca_teleop.am200.canonical import CanonicalJointMap
from orca_teleop.am200.show_demo import DemoFrame, build_show_demo

logger = logging.getLogger(__name__)


def _joint_map(model_path: Path) -> CanonicalJointMap:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    return CanonicalJointMap.load(model_path.parent / "joint_map.json", model)


def _count_in() -> None:
    print("카운트 인:", end="", flush=True)
    for count in (3, 2, 1):
        print(f" {count}", end="", flush=True)
        time.sleep(1.0)
    print()


def _run(
    sink: Any,
    frames: list[DemoFrame],
    *,
    realtime: bool,
    video_path: Path | None,
) -> None:
    joint_ids = tuple(sink.joint_ids)
    pixels: list[np.ndarray] | None = [] if video_path is not None else None
    preview_camera = None
    if pixels is not None and isinstance(sink, Am200SimSink):
        preview_camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(sink._env.model, preview_camera)
        preview_camera.lookat[:] = [0.0, -0.12, 0.32]
        preview_camera.distance = 0.85
        preview_camera.azimuth = 90.0  # audience in world -y, facing the palm
        preview_camera.elevation = -8.0
    t0 = time.monotonic()
    phase = ""
    for frame in frames[1:]:
        if realtime:
            delay = t0 + frame.time_s - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
        if frame.phase != phase:
            phase = frame.phase
            logger.info("phase: %s", phase)
        sink.dispatch_action(RobotJointCommand.now(joint_ids, frame.position))
        if pixels is not None:
            pixels.append(sink._render(preview_camera))

    if pixels is not None:
        import imageio.v2 as imageio

        video_path.parent.mkdir(parents=True, exist_ok=True)
        dt = np.median(np.diff([f.time_s for f in frames]))
        imageio.mimsave(video_path, pixels, fps=round(1.0 / dt))
        logger.info("saved preview: %s", video_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["sim", "hardware"], default="sim")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="required acknowledgement before physical hardware can move",
    )
    parser.add_argument("--cycles", type=int, default=1, help="official flourish sequence repeats")
    parser.add_argument("--speed", type=float, default=1.2, help="tempo multiplier, 0.5-1.5")
    parser.add_argument("--hz", type=float, default=30.0, help="command rate")
    parser.add_argument(
        "--motion-scale",
        type=float,
        default=0.7,
        help="joint-motion amplitude, 0.1-0.7 (start lower on hardware)",
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--hand-config", default=None, help="default: $ORCA_MODEL_PATH")
    parser.add_argument("--arm-config", type=Path, default=default_arm_config_path())
    parser.add_argument("--fast", action="store_true", help="simulation only: run without wall-clock waits")
    parser.add_argument("--video", type=Path, default=None, help="simulation only: save preview mp4")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.backend == "hardware":
        if not args.execute:
            parser.error("hardware movement requires --execute")
        if args.fast:
            parser.error("--fast is simulation-only")
        if args.video is not None:
            parser.error("--video is simulation-only")
        print(
            "팔을 주변 사람·물체와 충분히 떨어진 중립 자세로 잡아 주세요.\n"
            "연결 순간 자세가 기준점이 됩니다. 비상 시 Ctrl-C를 누르세요."
        )
        input("준비되면 Enter: ")
        sink: Any = Am200HardwareSink(
            hand_config_path=args.hand_config,
            arm_config_path=args.arm_config,
            model_path=args.model_path,
            enable_arm=True,
            enable_hand=True,
        )
    else:
        sink = Am200SimSink(
            model_path=args.model_path,
            hand_config_path=args.hand_config,
            render_mode="rgb_array",
            camera_config=SimCameraConfig(name="frontal", width=640, height=480),
        )

    completed = False
    try:
        sink.connect()
        if isinstance(sink, Am200SimSink):
            # The sim starts at MJCF qpos0, while hardware init_joints() starts
            # at the configured ORCA neutral pose. Let the sim reach that same
            # neutral before treating its measured state as the choreography
            # origin.
            for _ in range(round(args.hz * 2.0)):
                sink.go_home()
        model_path = Path(sink.model_path)
        mapping = _joint_map(model_path)
        start = np.asarray(sink.get_observation().joint_state, dtype=np.float64)
        frames = build_show_demo(
            sink.joint_ids,
            start,
            mapping.lower,
            mapping.upper,
            cycles=args.cycles,
            speed=args.speed,
            hz=args.hz,
            motion_scale=args.motion_scale,
            return_to_start=True,
        )
        logger.info(
            "show duration %.1fs, %d commands, cycles=%d, speed=%.2f, scale=%.2f",
            frames[-1].time_s,
            len(frames) - 1,
            args.cycles,
            args.speed,
            args.motion_scale,
        )
        if args.backend == "hardware":
            _count_in()
        _run(sink, frames, realtime=not args.fast, video_path=args.video)
        completed = True
        return 0
    except KeyboardInterrupt:
        logger.warning("interrupted; stopping commands and disabling torque")
        return 130
    finally:
        if completed:
            # The generated trajectory has already returned to connect-time
            # neutral; this makes the intended shutdown state explicit.
            sink.go_home()
            time.sleep(0.3)
        sink.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
