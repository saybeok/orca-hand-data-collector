"""Record scripted-expert demonstrations of an AM-ARM200 sim task into a LeRobotDataset.

The privileged scripted expert (:mod:`orca_teleop.am200.scripted`) drives the
sim closed-loop at the sink's control rate; only successful episodes are kept.
The layout matches ``record_dataset.py --backend am200_sim`` and what
``eval_am200_policy.py`` feeds a checkpoint: ``observation.state`` / ``action``
are the 20-DoF canonical vector in model radians (the action is the expert's
joint target), images are ``observation.images.<key>``.

    python scripts/record_am200_scripted.py --repo-id $HF_USERNAME/am200-hwatu-place \\
        --sim-task hwatu_place --task-text "put the hwatu card on the yellow mark" \\
        --episodes 100 --randomize --sim-camera wrist:wrist_cam

Demo seeds start at 0; ``eval_am200_policy.py`` evaluates from seed 1000. Each
kept episode's seed, time to success and dynamics sample go to
``meta/scripted_episodes.jsonl`` in the dataset root.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from orca_teleop.am200 import Am200SimSink, RobotJointCommand, SimCameraConfig, parse_sim_camera_spec
from orca_teleop.am200.scripted import make_expert
from orca_teleop.am200.tasks import TASKS

logger = logging.getLogger(__name__)


def run_demo(
    sink: Am200SimSink, dataset: Any, seed: int, max_steps: int, tail_steps: int, task_text: str
) -> dict[str, Any] | None:
    """Record one expert episode into the dataset buffer; returns its record if it succeeded."""
    joint_ids = sink.joint_ids
    sink.reset_scene(seed=seed)
    expert = make_expert(sink)
    expert.reset()
    t_success: float | None = None
    steps = 0
    while steps < max_steps:
        obs = sink.get_observation()
        target = np.asarray(expert.select_action({}), dtype=np.float64)
        sink.dispatch_action(RobotJointCommand.now(tuple(joint_ids), target))
        dataset.add_frame({
            "observation.state": np.asarray(obs.joint_state, dtype=np.float32),
            "action": target.astype(np.float32),
            "task": task_text,
            **{f"observation.images.{k}": v for k, v in obs.images.items()},
        })
        steps += 1
        info = sink.task_info or {}
        if t_success is None and info.get("success"):
            t_success = steps * sink._env.control_dt
            max_steps = min(max_steps, steps + tail_steps)
        if info.get("dropped"):
            break
    if t_success is None:
        return None
    return {
        "seed": seed,
        "time_to_success_s": t_success,
        "frames": steps,
        "dynamics": dict(sink._env.dynamics_sample),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, default=None, help="default: the LeRobot cache")
    parser.add_argument("--overwrite", action="store_true", help="delete an existing dataset at --root")
    parser.add_argument("--sim-task", default="hwatu_place", choices=sorted(TASKS))
    parser.add_argument("--task-text", required=True)
    parser.add_argument("--episodes", type=int, default=50, help="successful episodes to keep")
    parser.add_argument("--max-attempts", type=int, default=None, help="default: 2 x --episodes")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=14.0)
    parser.add_argument(
        "--tail-seconds", type=float, default=0.5,
        help="keep recording this long after success (the release/retreat)",
    )
    parser.add_argument(
        "--randomize", action=argparse.BooleanOptionalAction, default=None,
        help="domain randomization (default: robot_model/sim_dynamics.yaml)",
    )
    parser.add_argument("--hand-config", default=None, help="default: $ORCA_MODEL_PATH")
    parser.add_argument("--camera-name", default="frontal", help="image key of the task camera")
    parser.add_argument("--camera-width", type=int, default=320)
    parser.add_argument("--camera-height", type=int, default=240)
    parser.add_argument(
        "--sim-camera", action="append", default=[], metavar="KEY:MJCF_CAMERA",
        help="extra fixed MJCF camera observed under KEY, repeatable (e.g. wrist:wrist_cam)",
    )
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s | %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("orca_teleop.am200.sim_sink").setLevel(logging.WARNING)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    sink = Am200SimSink(
        hand_config_path=args.hand_config,
        render_mode="rgb_array",
        camera_config=SimCameraConfig(
            name=args.camera_name, width=args.camera_width, height=args.camera_height
        ),
        task=args.sim_task,
        randomize=args.randomize,
        extra_cameras=dict(parse_sim_camera_spec(s) for s in args.sim_camera),
    )
    sink.connect()
    dt = sink._env.control_dt
    joint_ids = list(sink.joint_ids)
    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": (len(joint_ids),), "names": joint_ids},
        "action": {"dtype": "float32", "shape": (len(joint_ids),), "names": joint_ids},
    }
    for key, shape in sink.camera_shapes.items():
        features[f"observation.images.{key}"] = {
            "dtype": "video", "shape": shape, "names": ["height", "width", "channels"],
        }
    if args.overwrite and args.root is not None and args.root.exists():
        shutil.rmtree(args.root)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=round(1.0 / dt), features=features, root=args.root, use_videos=True
    )
    logger.info(
        "task=%s control_dt=%.3fs (dataset fps %d) randomize=%s -> %s",
        args.sim_task, dt, dataset.fps, sink._env.dynamics.randomization.enabled, dataset.root,
    )

    max_steps = max(1, round(args.max_seconds / dt))
    tail_steps = round(args.tail_seconds / dt)
    max_attempts = args.max_attempts or 2 * args.episodes
    sidecar = Path(dataset.root) / "meta" / "scripted_episodes.jsonl"
    kept = attempts = 0
    try:
        while kept < args.episodes and attempts < max_attempts:
            seed = args.seed_start + attempts
            attempts += 1
            rec = run_demo(sink, dataset, seed, max_steps, tail_steps, args.task_text)
            if rec is None:
                dataset.clear_episode_buffer()
                logger.info("seed %d failed; discarded (%d/%d kept)", seed, kept, attempts)
                continue
            dataset.save_episode()
            kept += 1
            with sidecar.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"episode_index": dataset.num_episodes - 1, **rec}) + "\n")
            logger.info(
                "seed %d kept: success at %.1fs, %d frames (%d/%d kept)",
                seed, rec["time_to_success_s"], rec["frames"], kept, attempts,
            )
    except KeyboardInterrupt:
        logger.info("interrupted")
        dataset.clear_episode_buffer()
    finally:
        sink.close()

    dataset.finalize()
    logger.info("kept %d / %d attempts; dataset has %d episodes", kept, attempts, dataset.num_episodes)
    if args.push_to_hub:
        dataset.push_to_hub()
    return 0 if kept == args.episodes else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
