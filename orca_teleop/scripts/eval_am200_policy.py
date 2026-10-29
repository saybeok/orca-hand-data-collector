"""Evaluate a policy on an AM-ARM200 + ORCA hand sim task over many seeds.

Each episode resets the task scene with its own seed (object pose and, with
``--randomize``, dynamics), runs the policy closed-loop at the sink's control
rate, and records success / time-to-success / lift / drop. The summary reports
the success rate with a Wilson 95% interval; per-episode records go to JSON.

    python scripts/eval_am200_policy.py \\
        --policy-path outputs/train/am200-pinch-act/checkpoints/last/pretrained_model \\
        --dataset-repo-id $HF_USERNAME/am200-pinch --task-text "pinch and lift the cylinder" \\
        --episodes 50 --randomize --out eval/pinch_act.json

``--policy-path hold`` runs a do-nothing baseline (robot holds home) to check
the pipeline without a checkpoint; ``--policy-path scripted`` runs the
privileged scripted pinch expert (:mod:`orca_teleop.am200.scripted`).

Observations and actions follow ``record_dataset.py --backend am200_sim``:
``observation.state`` / ``action`` are the 20-DoF canonical vector in model
radians and the image key is ``observation.images.<camera-name>``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from orca_teleop.am200 import (
    Am200SimSink,
    RobotJointCommand,
    SimCameraConfig,
    parse_sim_camera_spec,
)
from orca_teleop.am200.tasks import TASKS
from orca_teleop.policies import GenericPolicy, reset_policy

logger = logging.getLogger(__name__)


class HoldPolicy:
    """Baseline that commands the current joint state forever."""

    def __init__(self) -> None:
        self._target: np.ndarray | None = None

    def reset(self) -> None:
        self._target = None

    def select_action(self, observation: dict[str, Any]) -> np.ndarray:
        if self._target is None:
            self._target = np.asarray(observation["observation.state"], dtype=np.float64)
        return self._target


def action_vector(action: Any, joint_ids: list[str]) -> np.ndarray:
    """Policy output (named dict, tensor or array) -> canonical 20-vector."""
    if isinstance(action, dict):
        missing = [j for j in joint_ids if j not in action]
        if missing:
            raise ValueError(f"policy action is missing joints: {missing[:4]}...")
        return np.array([float(action[j]) for j in joint_ids])
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    arr = np.asarray(action, dtype=np.float64)
    while arr.ndim > 1:
        arr = arr[0]
    if arr.shape != (len(joint_ids),):
        raise ValueError(f"policy action shape {arr.shape}, expected ({len(joint_ids)},)")
    return arr


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def run_episode(
    sink: Am200SimSink,
    policy: GenericPolicy,
    seed: int,
    max_steps: int,
    task_text: str,
    video_keys: list[str],
    frames: list[np.ndarray] | None,
) -> dict[str, Any]:
    joint_ids = sink.joint_ids
    sink.reset_scene(seed=seed)
    reset_policy(policy)
    dt = sink._env.control_dt
    max_lift, t_success, steps = 0.0, None, 0
    info: dict[str, Any] = sink.task_info or {}
    start_pose = np.asarray(info.get("object_pose", np.zeros(7)))[:3].tolist()

    for steps in range(1, max_steps + 1):
        obs = sink.get_observation()
        if frames is not None:
            frames.append(np.concatenate([obs.images[k] for k in video_keys], axis=1))
        frame = {
            "observation.state": obs.joint_state,
            "observation.state.names": joint_ids,
            "task": task_text,
            **{f"observation.images.{k}": v for k, v in obs.images.items()},
        }
        target = action_vector(policy.select_action(frame), joint_ids)
        sink.dispatch_action(RobotJointCommand.now(tuple(joint_ids), target))

        info = sink.task_info or {}
        max_lift = max(max_lift, float(info.get("lift_m", 0.0)))
        if info.get("success"):
            t_success = steps * dt
            break
        if info.get("dropped"):
            break

    return {
        "seed": seed,
        "success": t_success is not None,
        "time_to_success_s": t_success,
        "steps": steps,
        "max_lift_m": max_lift,
        "dropped": bool(info.get("dropped", False)),
        "object_start_xyz": start_pose,
        "dynamics": dict(sink._env.dynamics_sample),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy-path", required=True, help="LeRobot checkpoint dir, or 'hold'")
    parser.add_argument("--dataset-repo-id", default=None, help="dataset the policy was trained on")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sim-task", default="pinch", choices=sorted(TASKS))
    parser.add_argument("--task-text", default="pinch and lift the cylinder")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=1000, help="keep away from demo seeds")
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument(
        "--randomize", action=argparse.BooleanOptionalAction, default=None,
        help="domain randomization (default: robot_model/sim_dynamics.yaml)",
    )
    parser.add_argument("--am200-model", default=None, help="override the task scene MJCF")
    parser.add_argument("--hand-config", default=None, help="default: $ORCA_MODEL_PATH")
    parser.add_argument("--render-mode", default="rgb_array", choices=["rgb_array", "human"])
    parser.add_argument("--camera-name", default="frontal", help="image key used in the dataset")
    parser.add_argument("--camera-width", type=int, default=320)
    parser.add_argument("--camera-height", type=int, default=240)
    parser.add_argument(
        "--sim-camera", action="append", default=[], metavar="KEY:MJCF_CAMERA",
        help="extra fixed MJCF camera observed under KEY, repeatable (e.g. wrist:wrist_cam)",
    )
    parser.add_argument("--video-dir", type=Path, default=None, help="save one mp4 per episode")
    parser.add_argument("--out", type=Path, default=None, help="results json")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s | %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("orca_teleop.am200.sim_sink").setLevel(logging.WARNING)

    policy: GenericPolicy | None = None
    if args.policy_path == "hold":
        policy = HoldPolicy()
    elif args.policy_path == "scripted":
        pass  # needs the connected sink; built below
    else:
        if not args.dataset_repo_id:
            parser.error("--dataset-repo-id is required for a LeRobot checkpoint")
        from orca_teleop.policies import LeRobotPolicyAdapter

        policy = LeRobotPolicyAdapter.from_pretrained(
            args.policy_path,
            args.dataset_repo_id,
            dataset_root=str(args.dataset_root) if args.dataset_root else None,
            device=args.device,
        )

    extra_cameras = dict(parse_sim_camera_spec(s) for s in args.sim_camera)
    sink = Am200SimSink(
        model_path=args.am200_model,
        hand_config_path=args.hand_config,
        render_mode=args.render_mode,
        camera_config=SimCameraConfig(
            name=args.camera_name, width=args.camera_width, height=args.camera_height
        ),
        task=args.sim_task,
        randomize=args.randomize,
        extra_cameras=extra_cameras,
    )
    sink.connect()
    if policy is None:
        from orca_teleop.am200.scripted import make_expert

        policy = make_expert(sink)
    randomized = sink._env.dynamics.randomization.enabled
    max_steps = max(1, round(args.max_seconds / sink._env.control_dt))
    logger.info(
        "task=%s episodes=%d control_dt=%.3fs max_steps=%d randomize=%s",
        args.sim_task, args.episodes, sink._env.control_dt, max_steps, randomized,
    )

    video_keys = [args.camera_name, *extra_cameras]
    records: list[dict[str, Any]] = []
    t0 = time.time()
    try:
        for i in range(args.episodes):
            frames: list[np.ndarray] | None = [] if args.video_dir else None
            rec = run_episode(
                sink, policy, args.seed_start + i, max_steps, args.task_text, video_keys, frames
            )
            records.append(rec)
            n_ok = sum(r["success"] for r in records)
            logger.info(
                "ep %3d seed=%d %s lift=%.3fm steps=%d | running %d/%d",
                i + 1, rec["seed"], "SUCCESS" if rec["success"] else ("drop" if rec["dropped"] else "fail"),
                rec["max_lift_m"], rec["steps"], n_ok, len(records),
            )
            if frames:
                _save_video(args.video_dir / f"ep{i:03d}_seed{rec['seed']}.mp4", frames, sink._env.control_dt)
    except KeyboardInterrupt:
        logger.info("interrupted after %d episodes", len(records))
    finally:
        sink.close()

    n = len(records)
    k = sum(r["success"] for r in records)
    lo, hi = wilson_interval(k, n)
    times = [r["time_to_success_s"] for r in records if r["success"]]
    summary = {
        "policy": args.policy_path,
        "sim_task": args.sim_task,
        "episodes": n,
        "successes": k,
        "success_rate": k / n if n else 0.0,
        "success_rate_ci95": [lo, hi],
        "mean_time_to_success_s": float(np.mean(times)) if times else None,
        "drop_rate": sum(r["dropped"] for r in records) / n if n else 0.0,
        "mean_max_lift_m": float(np.mean([r["max_lift_m"] for r in records])) if n else 0.0,
        "randomize": randomized,
        "seed_start": args.seed_start,
        "max_seconds": args.max_seconds,
        "wall_s": time.time() - t0,
    }
    print(
        f"\nsuccess {k}/{n} = {summary['success_rate']:.1%} (95% CI {lo:.1%}-{hi:.1%})"
        + (f", mean time {summary['mean_time_to_success_s']:.1f}s" if times else "")
        + f", drops {summary['drop_rate']:.1%}"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "episodes": records}, indent=2), encoding="utf-8")
        print(f"saved {args.out}")
    return 0


def _save_video(path: Path, frames: list[np.ndarray], dt: float) -> None:
    try:
        import imageio.v2 as imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(path, frames, fps=round(1.0 / dt))
    except Exception:
        logger.exception("could not write %s", path)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
