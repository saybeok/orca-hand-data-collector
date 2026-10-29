"""Record teleoperated episodes into a LeRobotDataset.

Record — per episode:

  - teleop consumer thread: owns ``actions_q``, mirrors the latest command, and
    dispatches to the hand at teleop rate (independent of recording)
  - main thread: sample the sink on a fixed ``--fps`` clock (default 30 Hz).
    Each tick snapshots the mirrored teleop command and current observations
    into one dataset row. Recording timing does not drive robot dispatch.
  - recorder thread: blocks on ``rec_q``, calls ``dataset.add_frame(...)`` for
    every dict, and ``dataset.save_episode()`` when it sees the SAVE sentinel
  - LeRobot image writer: four threads per camera write PNGs asynchronously;
    saving an episode waits for and validates the images before video encoding
  - episodes after the first begin with ``go_home()`` (reset to the recording home
    pose), rest (default 5 s) with no teleop dispatch, then sensor warmup and
    recording; episode 1 starts without homing; SPACE saves and advances; at
    the end: optional push to Hub

Add cameras:

  Cameras are opened with OpenCV. On macOS a plugged-in / paired iPhone shows up
  as a regular video device (through Continuity Camera) via the AVFoundation
  backend, so it can be recorded like any other camera. First discover its
  index::

      python scripts/record_dataset.py --list-cameras

  then pass ``--camera NAME:INDEX[:WIDTHxHEIGHT][@FPS]`` (repeatable). Omitted
  resolution/fps default to 640x480@30. To validate the camera pipeline
  *without* the physical hand, add ``--stub`` (state/action are dummy, but real
  video is recorded).

Example usage:

.. code-block:: bash

    HF_USERNAME=your_username python scripts/record_dataset.py --repo-id $HF_USERNAME/orca-teleop \\
        --task "pick up the block" \\
        --num-episodes 5 --episode-seconds 20 \\
        --camera front:0 --camera wrist:1 \\
        --push-to-hub

    # Discover which index the iPhone (Continuity Camera) landed on
    python scripts/record_dataset.py --list-cameras

    # Camera-only smoke test with the iPhone, no hand required
    python scripts/record_dataset.py --repo-id $HF_USERNAME/orca-iphone-test \\
        --task "wave at the camera" --stub \\
        --camera iphone:1:1280x720@30 --num-episodes 1 --episode-seconds 5

    # Record using a simulated hand
    python scripts/record_dataset.py --backend sim --repo-id $HF_USERNAME/orca-sim \\
        --task "pick up the block" --num-episodes 5

    # Teleop the REAL hand with Manus gloves, ending each episode by pressing SPACE
    # (prerequisite: 'manus-client run' streaming glove data)
    python scripts/record_dataset.py --repo-id $HF_USERNAME/orca-manus \\
        --task "pick up the block" --backend hardware \\
        --local --source manus --hand right \\
        --episode-end space --num-episodes 20 --camera front:0

    # Teleop and record the REAL hand from the repository-owned Quest WebXR page
    python scripts/record_dataset.py --repo-id $HF_USERNAME/orca-quest-hardware \\
        --task "pick up the block" --backend hardware \\
        --local --source metaquest --hand right \\
        --episode-end space --num-episodes 20 --camera iphone:1

    # Combined AM-ARM200 + ORCA hand v1 in MuJoCo (20-DoF state/action)
    python scripts/record_dataset.py --backend am200_sim --repo-id $HF_USERNAME/orca-am200 \\
        --task "reach and grasp" --local --source metaquest --hand right --clutch \\
        --episode-end space --num-episodes 20

    # Physical AM-ARM200 + ORCA hand v1 from Quest (same 20-DoF schema as am200_sim)
    python scripts/record_dataset.py --backend am200_hw --repo-id $HF_USERNAME/orca-am200-real \\
        --task "pick up the card" --local --source metaquest --hand right --clutch \\
        --arm-scale 0.35 --arm-max-step 0.3 \\
        --episode-end space --num-episodes 20 --camera front:0
"""

import argparse
import logging
import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
from orca_core import OrcaJointPositions

from orca_teleop.cameras import (
    OpenCVCameraConfig,
    parse_camera_spec,
    print_available_cameras,
    with_recording_defaults,
)
from orca_teleop.constants import (
    DEFAULT_CONFIDENCE,
    DEFAULT_HAND,
    DEFAULT_PORT,
    JOIN_TIMEOUT,
    QUEUES_MAXSIZE,
    SIM_RENDER_MODE,
)
from orca_teleop.ingress.server import IngressServer
from orca_teleop.pipeline import (
    _SHUTDOWN,
    OrcaHandSink,
    RecordableSink,
    TeleopQueues,
    _manus_publisher,
    _mediapipe_publisher,
    _metaquest_publisher,
    retargeter_worker,
)
from orca_teleop.recording import (
    TeleopActionMirror,
    action_to_feature,
    drain_actions_queue,
    teleop_consumer_loop,
    wait_for_teleop_mirror_ready,
)
from orca_teleop.utils import RateTicker

logger = logging.getLogger(__name__)


MODEL_PATH = (
    "/Users/fracapuano/Documents/orca_core/orca_core/models/v2/orcahand_touch_right/config.yaml"
)
_DEFAULT_FPS = 30
_DEFAULT_NUM_EPISODES = 1
_DEFAULT_EPISODE_SECONDS = 30.0
_DEFAULT_REST_SECONDS = 5.0
_IMAGE_WRITER_THREADS_PER_CAMERA = 4
# Limit pending raw images as well as rec_q: LeRobot's writer queue is unbounded.
_IMAGE_WRITER_BACKLOG_PER_CAMERA = 64

# Sentinels pushed onto rec_q between episodes — the recorder either flushes
# (save) or drops (discard) the in-progress episode buffer, then keeps draining.
_SAVE_EPISODE = object()
_DISCARD_EPISODE = object()


class _KeyboardController:
    """Read single keypresses from an interactive terminal on a background thread.

    Puts the TTY in cbreak mode (so keys are delivered without waiting for Enter,
    while Ctrl+C still raises ``KeyboardInterrupt``) and maps:

      - ``space`` -> ``advance_event`` (stop recording, save episode, then start
        the next episode with go_home + rest)
      - ``q`` / ``Q`` / ``Esc`` -> ``stop_event`` (end the whole recording session)

    The terminal state is always restored on ``stop()`` or thread exit.
    """

    def __init__(self, stop_event: threading.Event, advance_event: threading.Event) -> None:
        self._stop_event = stop_event
        self._advance_event = advance_event
        self._thread: threading.Thread | None = None
        self._old_attrs = None
        self._fd: int | None = None

    def start(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError(
                "--episode-end space/both requires an interactive terminal, but stdin "
                "is not a TTY. Run in a real terminal or use --episode-end timer."
            )
        import termios
        import tty

        self._fd = sys.stdin.fileno()
        self._old_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)  # cbreak keeps ISIG on, so Ctrl+C still works
        self._thread = threading.Thread(target=self._loop, name="keyboard-controller", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        import select

        try:
            while not self._stop_event.is_set():
                ready, _, _ = select.select([self._fd], [], [], 0.1)
                if not ready:
                    continue
                ch = os.read(self._fd, 1)
                if not ch:
                    continue
                if ch == b" ":
                    self._advance_event.set()
                elif ch in (b"q", b"Q", b"\x1b"):
                    self._stop_event.set()
                    self._advance_event.set()  # unblock the episode loop immediately
        finally:
            self._restore()

    def _restore(self) -> None:
        if self._old_attrs is not None and self._fd is not None:
            import termios

            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
            except Exception:
                pass
            self._old_attrs = None

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._restore()


def _default_lerobot_root(repo_id: str) -> Path:
    """Mirror lerobot's default cache location."""
    base = os.environ.get("HF_LEROBOT_HOME")
    if base:
        return Path(base) / repo_id
    return Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id


def _parse_camera_configs(specs: list[str]) -> list[OpenCVCameraConfig]:
    return [with_recording_defaults(parse_camera_spec(spec)) for spec in specs]


def _preview_observation(sink, control):
    observation = sink.get_observation()
    if control:
        control.observe(observation)
    return observation


def _stub_action_publisher(
    actions_q: "queue.Queue",
    stop_event: threading.Event,
    joint_ids: list[str],
    fps: float,
    *,
    as_robot_command: bool = False,
) -> None:
    """Push small random-walk actions onto actions_q (dev only).

    These actions are *recorded* by the main loop but, in stub mode, are
    deliberately not dispatched to the device — see ``main()``.
    """
    from orca_teleop.am200 import RobotJointCommand

    rng = np.random.default_rng()
    n = len(joint_ids)
    pose = np.zeros(n, dtype=float)
    period = 1.0 / float(fps)
    roles = tuple(joint_ids)
    while not stop_event.is_set():
        pose += rng.uniform(-0.2, 0.2, size=n)
        pose = np.clip(pose, -3.0, 3.0)
        if as_robot_command:
            action = RobotJointCommand.now(roles, pose.copy())
        else:
            action = OrcaJointPositions.from_dict(dict(zip(joint_ids, pose, strict=True)))
        try:
            actions_q.put_nowait(action)
        except queue.Full:
            try:
                actions_q.get_nowait()
            except queue.Empty:
                pass
            try:
                actions_q.put_nowait(action)
            except queue.Full:
                pass
        time.sleep(period)


def _create_recording_dataset(dataset_class, *, repo_id, fps, features, root):
    camera_count = sum(ft['dtype'] in ('image', 'video') for ft in features.values())
    threads = _IMAGE_WRITER_THREADS_PER_CAMERA * camera_count
    dataset = dataset_class.create(
        repo_id=repo_id, fps=fps, features=features, root=root, use_videos=True,
        image_writer_processes=0, image_writer_threads=threads,
    )
    logger.info('Image storage: %d cameras, %d asynchronous writer threads.', camera_count, threads)
    return dataset


def _check_image_writer_backlog(dataset):
    writer = getattr(dataset, 'image_writer', None)
    if writer is None:
        return
    camera_count = sum(ft['dtype'] in ('image', 'video') for ft in dataset.features.values())
    if writer.queue.qsize() >= _IMAGE_WRITER_BACKLOG_PER_CAMERA * max(camera_count, 1):
        raise RuntimeError('이미지 저장 대기열이 계속 밀려 현재 시도를 폐기합니다. CPU·디스크 부하를 확인해 주세요.')


def _validate_recorded_images(dataset):
    """Catch swallowed LeRobot PNG write failures before committing episode metadata."""
    writer = getattr(dataset, 'image_writer', None)
    if writer is not None:
        writer.wait_until_done()
    image_features = {
        key: ft for key, ft in getattr(dataset, 'features', {}).items()
        if ft['dtype'] in ('image', 'video')
    }
    if not image_features:
        return
    from PIL import Image

    buffer = dataset.episode_buffer
    for key, ft in image_features.items():
        paths = buffer[key]
        if len(paths) != buffer['size']:
            raise RuntimeError(f'{key}: 이미지 수와 데이터 프레임 수가 다릅니다.')
        if paths and len(list(Path(paths[0]).parent.glob('frame-*.png'))) != buffer['size']:
            raise RuntimeError(f'{key}: 이전 시도의 이미지가 남아 있거나 이미지가 누락되었습니다.')
        for path in paths:
            with Image.open(path) as image:
                if image.size != (ft['shape'][1], ft['shape'][0]):
                    raise RuntimeError(f'{key}: 저장된 이미지 해상도가 다릅니다: {path}')
                image.verify()


def _discard_recording_episode(dataset):
    writer = getattr(dataset, 'image_writer', None)
    if writer is not None:
        writer.wait_until_done()
    directories = []
    for key, ft in getattr(dataset, 'features', {}).items():
        if ft['dtype'] == 'video':
            index = int(np.asarray(dataset.episode_buffer['episode_index']).reshape(-1)[0])
            directory = dataset._get_image_file_dir(index, key)
            if not directory.resolve().is_relative_to((dataset.root / 'images').resolve()):
                raise RuntimeError(f'Unexpected temporary image directory: {directory}')
            directories.append(directory)
    dataset.clear_episode_buffer()
    # LeRobot 0.4.4 clears dtype=image files, but leaves temporary dtype=video
    # PNGs on discard. A shorter retry must not encode the previous trial's tail.
    for directory in directories:
        if directory.is_dir():
            shutil.rmtree(directory)


def _recorder_loop(
    dataset,
    rec_q: "queue.Queue",
    episode_finalized_q: "queue.Queue[bool]",
    failure_event: threading.Event | None = None,
) -> None:
    """Drain rec_q. Frames are dicts; save/discard sentinels finalize; ``None`` exits."""
    episode_failed = False
    while True:
        item = rec_q.get()
        if item is None:
            break
        if item is _SAVE_EPISODE:
            succeeded = False
            try:
                if episode_failed:
                    _discard_recording_episode(dataset)
                    logger.error('Refusing to save an episode after a frame write failure.')
                else:
                    _validate_recorded_images(dataset)
                    dataset.save_episode()
                    logger.info("Episode saved.")
                    succeeded = True
            except Exception:
                logger.exception("Failed to save episode")
                if failure_event is not None:
                    failure_event.set()
                try:
                    _discard_recording_episode(dataset)
                except Exception:
                    logger.exception('Failed to discard the unsuccessful episode')
            finally:
                episode_failed = False
                episode_finalized_q.put(succeeded)
            continue
        if item is _DISCARD_EPISODE:
            succeeded = False
            try:
                _discard_recording_episode(dataset)
                logger.warning("Episode discarded (incomplete / sensor failure).")
                succeeded = True
            except Exception:
                logger.exception("Failed to discard episode buffer")
            finally:
                episode_failed = False
                episode_finalized_q.put(succeeded)
            continue
        if episode_failed:
            continue
        try:
            _check_image_writer_backlog(dataset)
            dataset.add_frame(item)
        except Exception:
            logger.exception("Failed to add frame")
            episode_failed = True
            if failure_event is not None:
                failure_event.set()


def _main_record(argv: list[str]) -> None:
    # Short-circuit camera discovery before argparse enforces required flags
    if "--list-cameras" in argv:
        print_available_cameras()
        return

    parser = argparse.ArgumentParser(
        description="Record teleoperated episodes into a LeRobotDataset.",
    )
    parser.add_argument("--repo-id", required=True, help="LeRobotDataset repo id")
    parser.add_argument("--task", required=True, help="Task description for every episode")
    parser.add_argument('--control-dir', type=Path, default=None,
                        help='Local desktop collector control/preview directory (no TTY needed).')
    parser.add_argument('--auto-home', action=argparse.BooleanOptionalAction, default=True,
                        help='Return home between attempts; desktop UI disables this for manual resets.')
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=_DEFAULT_NUM_EPISODES,
        help=f"Number of episodes to record (default: {_DEFAULT_NUM_EPISODES})",
    )
    parser.add_argument(
        "--episode-seconds",
        type=float,
        default=_DEFAULT_EPISODE_SECONDS,
        help=(
            "Length of each episode in seconds. Only used for --episode-end timer/both "
            f"(default: {_DEFAULT_EPISODE_SECONDS})"
        ),
    )
    parser.add_argument(
        "--episode-end",
        choices=["timer", "space", "both"],
        default="timer",
        help=(
            "How each episode ends: 'timer' after --episode-seconds (default), "
            "'space' when you press the SPACE key (no time limit), or 'both' "
            "(whichever comes first). For space/both, press 'q'/Esc to end the "
            "whole session. Requires an interactive terminal."
        ),
    )
    parser.add_argument(
        "--rest-seconds",
        type=float,
        default=_DEFAULT_REST_SECONDS,
        help="Pause after go_home() at the start of each episode after the first "
        "episode (default: {_DEFAULT_REST_SECONDS})",
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        help=(
            "Camera spec NAME[:INDEX|usb=USB_PATH][:WIDTHxHEIGHT][@FPS], repeatable "
            "(e.g. front:0, iphone:1 — defaults to 640x480@30 when omitted). "
            "Cameras become part of the sink's observation; with --backend sim "
            "they're recorded alongside the sim's MuJoCo render. "
            "Use --list-cameras to discover indices."
        ),
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available camera indices (with resolution) and exit.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="After recording, push the dataset to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Wipe any existing dataset at the same root before creating.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Local dataset root (defaults to lerobot's HF cache).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=_DEFAULT_FPS,
        help=f"Fixed dataset sampling rate in Hz (default: {_DEFAULT_FPS})",
    )
    parser.add_argument("--model-path", default=MODEL_PATH, help="OrcaHand model directory")
    parser.add_argument("--urdf-path", default=None, help="Hand URDF file")
    parser.add_argument(
        "--backend",
        choices=["hardware", "sim", "am200_sim", "am200_hw"],
        default="hardware",
        help=(
            "Backend to record against (default: hardware). am200_sim and am200_hw "
            "(physical AM-ARM200 + ORCA hand) are 20-DoF."
        ),
    )
    parser.add_argument(
        "--am200-model",
        default=None,
        help="Combined AM+hand MJCF for the am200 backends (default: orca_am200 model).",
    )
    parser.add_argument(
        "--hand-config",
        default=os.environ.get("ORCA_MODEL_PATH"),
        help="orca_core hand config.yaml for am200 retargeting (default: $ORCA_MODEL_PATH).",
    )
    parser.add_argument(
        "--arm-config",
        default=None,
        help="am200_hw: AM-ARM200 config.yaml (default: configs/am-arm200-3dof).",
    )
    parser.add_argument(
        "--arm-max-step",
        type=float,
        default=None,
        help=(
            "am200_hw: clip every arm joint to +/- this many rad from the connect-time "
            "pose (overrides max_step_from_home_rad in the arm config)."
        ),
    )
    parser.add_argument(
        "--arm-scale",
        type=float,
        default=0.5,
        help="Flange metres per wrist metre for the am200 backends (default: 0.5).",
    )
    parser.add_argument(
        "--operator-cal",
        default=None,
        help="am200 backends: operator calibration file for reproducible episode starts.",
    )
    parser.add_argument('--arm-max-velocity-deg-s', type=float, default=None,
                        help='Limit each arm command to this many degrees/s; pair with acceleration limit.')
    parser.add_argument('--arm-max-acceleration-deg-s2', type=float, default=None,
                        help='Limit arm command acceleration in degrees/s^2; pair with velocity limit.')
    parser.add_argument(
        "--no-arm",
        action="store_true",
        help="am200 backends: hold the arm still; record hand motion only in the 20-D vector.",
    )
    parser.add_argument(
        "--clutch",
        action="store_true",
        help="Quest: off-hand pinch clutch for the arm (metaquest + am200 backends).",
    )
    parser.add_argument(
        "--sim-task",
        default=None,
        choices=["pinch", "hwatu", "hwatu_mat", "hwatu_place"],
        help="am200_sim: load a task scene (object respawned + dynamics resampled per episode).",
    )
    parser.add_argument(
        "--randomize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="am200_sim: per-episode domain randomization "
        "(default: robot_model/sim_dynamics.yaml).",
    )
    parser.add_argument(
        "--sim-camera",
        action="append",
        default=[],
        metavar="KEY:MJCF_CAMERA",
        help="am200_sim: record another fixed MJCF camera under KEY, repeatable "
        "(e.g. wrist:wrist_cam).",
    )
    parser.add_argument(
        "--retargeter", default="adaptive_analytical",
        choices=["rmsprop", "adaptive_analytical", "key_objectives", "dexpilot"],
        help="Retarget backend; key_objectives requires its own --retarget-config",
    )
    parser.add_argument(
        "--retarget-config",
        default=os.environ.get("ORCA_RETARGET_CONFIG"),
        help="Selected backend's YAML (default: $ORCA_RETARGET_CONFIG).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"gRPC ingress port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Also launch a local publisher (see --source) for one-command recording.",
    )
    parser.add_argument(
        "--source",
        choices=["mediapipe", "manus", "metaquest"],
        default="mediapipe",
        help=(
            "Input publisher and landmark convention: 'mediapipe' webcam (default), "
            "'manus' gloves (requires 'manus-client run'), or 'metaquest' WebXR in "
            "Quest Browser. With --local the matching publisher is launched here; "
            "without --local, launch it separately and point it at this ingress."
        ),
    )
    parser.add_argument(
        "--hand",
        choices=["left", "right"],
        default=DEFAULT_HAND,
        help=f"Which hand the local publisher streams (default: {DEFAULT_HAND}).",
    )
    parser.add_argument(
        "--zmq-address",
        default="tcp://127.0.0.1:2044",
        help="ZMQ address of the Manus C++ client for --source manus (default: tcp://127.0.0.1:2044).",
    )
    parser.add_argument(
        "--quest-host",
        default="0.0.0.0",
        help="Quest WebXR HTTP/WebSocket bind host (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--quest-port",
        type=int,
        default=8765,
        help="Quest WebXR HTTP/WebSocket port (default: 8765).",
    )
    parser.add_argument(
        "--quest-fps",
        type=int,
        default=_DEFAULT_FPS,
        help=f"Meta Quest gRPC publish-rate cap (default: {_DEFAULT_FPS}).",
    )
    parser.add_argument(
        "--quest-wrist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drive relative wrist flexion from the WebXR wrist pose (default: enabled).",
    )
    parser.add_argument(
        "--quest-wrist-scale",
        type=float,
        default=1.0,
        help="Scale applied to relative WebXR wrist flexion (default: 1.0).",
    )
    parser.add_argument(
        "--show-video",
        action="store_true",
        help="Show the local MediaPipe webcam feed with landmarks.",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help=(
            "Dev only: no hardware required. Skip ingress + retargeter, record random "
            "(undispatched) actions. Combine with --camera to test camera recording alone."
        ),
    )
    args = parser.parse_args(argv)
    if args.local and args.stub:
        parser.error("--local cannot be combined with --stub.")
    if args.source == "manus" and not args.local:
        logger.warning(
            "--source manus has no effect without --local; run the Manus publisher "
            "yourself (python -m orca_teleop.ingress.manus.publisher) or add --local."
        )
    if args.episode_end in ("space", "both") and not args.control_dir and not sys.stdin.isatty():
        parser.error("--episode-end space/both needs an interactive terminal.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    # TODO: Use constants key names from lerobot for frame key names

    queues = TeleopQueues(
        landmarks_q=queue.Queue(maxsize=QUEUES_MAXSIZE),
        actions_q=queue.Queue(maxsize=QUEUES_MAXSIZE),
    )
    stop_event = threading.Event()
    advance_event = threading.Event()
    control = None
    if args.control_dir:
        from orca_teleop.collection_control import CollectionControl
        control = CollectionControl(args.control_dir, stop_event, advance_event)

    # Sinks own and compose the full observation
    camera_configs = _parse_camera_configs(args.camera)
    action_composer = None
    retarget_model_path = args.model_path

    if args.backend in ("am200_sim", "am200_hw"):
        import mujoco

        from orca_teleop.am200 import (
            Am200ActionComposer,
            Am200HardwareSink,
            Am200SimSink,
            ArmMappingConfig,
            CanonicalJointMap,
            HoldArmActionComposer,
            OperatorCalibration,
            parse_sim_camera_spec,
        )

        if not args.hand_config:
            parser.error(f"--backend {args.backend} needs --hand-config or $ORCA_MODEL_PATH")
        if args.backend == "am200_sim":
            sink = Am200SimSink(
                model_path=args.am200_model,
                hand_config_path=args.hand_config,
                render_mode=SIM_RENDER_MODE,
                camera_configs=camera_configs,
                task=args.sim_task,
                randomize=args.randomize,
                extra_cameras=dict(parse_sim_camera_spec(s) for s in args.sim_camera),
            )
        else:
            if args.stub:
                parser.error("--backend am200_hw cannot be combined with --stub.")
            sink = Am200HardwareSink(
                hand_config_path=args.hand_config,
                arm_config_path=args.arm_config,
                model_path=args.am200_model,
                camera_configs=camera_configs,
                enable_arm=not args.no_arm,
                arm_max_step_rad=args.arm_max_step,
            )
        retarget_model_path = args.hand_config
        if not args.stub:
            if args.no_arm:
                model = mujoco.MjModel.from_xml_path(str(sink.model_path))
                joint_map = CanonicalJointMap.load(
                    sink.model_path.parent / "joint_map.json", model
                )
                action_composer = HoldArmActionComposer(joint_map)
            else:
                action_composer = Am200ActionComposer.from_model_path(
                    sink.model_path,
                    config=ArmMappingConfig(
                        scale=args.arm_scale,
                        max_velocity_rad_s=(None if args.arm_max_velocity_deg_s is None
                                            else np.deg2rad(args.arm_max_velocity_deg_s)),
                        max_acceleration_rad_s2=(None if args.arm_max_acceleration_deg_s2 is None
                                                 else np.deg2rad(args.arm_max_acceleration_deg_s2)),
                    ),
                    calibration=OperatorCalibration.load(args.operator_cal),
                )
    elif args.backend == "sim":
        from orca_teleop.sim import OrcaHandSimSink

        sink = OrcaHandSimSink(
            render_mode=SIM_RENDER_MODE,
            camera_configs=camera_configs,
        )
    else:
        sink = OrcaHandSink(
            model_path=args.model_path,
            camera_configs=camera_configs,
            connect_hardware=not args.stub,
        )
        if sink.handedness != args.hand:
            parser.error(
                f"--hand {args.hand!r} does not match the loaded physical OrcaHand "
                f"model ({sink.handedness!r}). Pass the matching --model-path or --hand."
            )
    sink.connect()
    assert isinstance(sink, RecordableSink), "Sinks must be recordable to collect data"

    joint_ids = list(sink.joint_ids)
    n_joints = len(joint_ids)

    # LeRobotDataset features schema
    features = {
        "observation.state": {"dtype": "float32", "shape": (n_joints,), "names": joint_ids},
        "action": {"dtype": "float32", "shape": (n_joints,), "names": joint_ids},
    }
    for cam_name, shape in sink.camera_shapes.items():
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }

    target_root = args.root if args.root is not None else _default_lerobot_root(args.repo_id)
    if args.overwrite and target_root.exists():
        logger.info("--overwrite: removing existing dataset at %s", target_root)
        shutil.rmtree(target_root)

    dataset = _create_recording_dataset(
        LeRobotDataset,
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=args.root,
    )
    logger.info("Dataset root: %s", dataset.root)
    if control:
        import hashlib
        from datetime import datetime, timezone
        from orca_teleop.collection_control import atomic_json

        configuration = {}
        paths = {
            'hand_config': args.hand_config,
            'retarget_config': args.retarget_config,
            'urdf': args.urdf_path,
            'arm_config': str(sink._arm_config_path) if hasattr(sink, '_arm_config_path') else None,
        }
        if args.hand_config:
            paths['calibration'] = str(Path(args.hand_config).with_name('calibration.yaml'))
        for key, value in paths.items():
            if value and Path(value).is_file():
                content = Path(value).read_bytes()
                configuration[key] = {'sha256': hashlib.sha256(content).hexdigest(), 'path': value}
                shutil.copyfile(value, dataset.root / f'collection_{key}{Path(value).suffix}')
        arm = getattr(sink, '_arm', None)
        arm_reference = None if arm is None else {
            'motor_home_rad': arm._motor_home.tolist(),
            'joint_signs': arm._signs.tolist(),
            'motor_ids': arm.motor_ids,
        }
        atomic_json(dataset.root / 'collection_session.json', {
            'created_at': datetime.now(timezone.utc).isoformat(),
            'arguments': {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
            'joint_ids': joint_ids, 'angle_units': 'canonical_model_radians',
            'arm_reference': arm_reference, 'configuration': configuration,
        })
        control.update(dataset_root=str(dataset.root), joint_ids=joint_ids, target=args.num_episodes)

    rec_q: queue.Queue = queue.Queue(maxsize=64)
    episode_finalized_q: queue.Queue[bool] = queue.Queue(maxsize=1)

    ingress_server: IngressServer | None = None
    retargeter_thread: threading.Thread | None = None
    stub_thread: threading.Thread | None = None
    publisher_process = None
    quest_reset_event = None

    if args.stub:
        logger.info(
            "STUB mode — no hardware required; ingress + retargeter bypassed, random "
            "actions recorded but NOT dispatched. Cameras still record real video."
        )
        stub_thread = threading.Thread(
            target=_stub_action_publisher,
            args=(queues.actions_q, stop_event, joint_ids, args.fps),
            kwargs={"as_robot_command": args.backend == "am200_sim"},
            name="stub-action-publisher",
        )
        stub_thread.start()
    else:
        if args.local:
            import multiprocessing

            ctx = multiprocessing.get_context("spawn")
            if args.source == "manus":
                publisher_process = ctx.Process(
                    target=_manus_publisher,
                    args=(args.port, args.hand, args.zmq_address),
                    name="manus-publisher",
                    daemon=True,
                )
            elif args.source == "metaquest":
                quest_reset_event = ctx.Event()
                publisher_process = ctx.Process(
                    target=_metaquest_publisher,
                    args=(
                        args.port,
                        args.hand,
                        args.quest_host,
                        args.quest_port,
                        args.quest_fps,
                        args.quest_wrist,
                        args.quest_wrist_scale,
                        quest_reset_event,
                        args.clutch,
                    ),
                    name="metaquest-webxr-publisher",
                    daemon=True,
                )
            else:
                publisher_process = ctx.Process(
                    target=_mediapipe_publisher,
                    args=(args.port, args.hand, DEFAULT_CONFIDENCE, args.show_video),
                    name="mediapipe-publisher",
                    daemon=True,
                )
            publisher_process.start()
            logger.info(
                "Local %s publisher started (pid=%d, hand=%s)",
                args.source,
                publisher_process.pid,
                args.hand,
            )
        ingress_server = IngressServer(queues.landmarks_q, stop_event, port=args.port)
        ingress_server.start()
        retargeter_thread = threading.Thread(
            target=retargeter_worker,
            args=(queues, stop_event, retarget_model_path, args.urdf_path),
            kwargs={
                "landmark_source": ("webxr" if args.source == "metaquest" else "mediapipe"),
                "retargeter_config_path": args.retarget_config,
                "retargeter_backend": args.retargeter,
                "action_composer": action_composer,
            },
            name="retargeter",
        )
        retargeter_thread.start()

    recorder_failed = threading.Event()
    recorder_thread = threading.Thread(
        target=_recorder_loop,
        args=(dataset, rec_q, episode_finalized_q),
        kwargs={'failure_event': recorder_failed},
        name="dataset-recorder",
    )
    recorder_thread.start()

    action_mirror = TeleopActionMirror()
    dispatch_enabled = threading.Event()
    teleop_thread = threading.Thread(
        target=teleop_consumer_loop,
        args=(queues.actions_q,),
        kwargs={
            "mirror": action_mirror,
            "stop_event": stop_event,
            "shutdown_sentinel": _SHUTDOWN,
            "dispatch_action": None if args.stub else sink.dispatch_action,
            "dispatch_enabled": dispatch_enabled,
        },
        name="teleop-consumer",
        daemon=True,
    )
    teleop_thread.start()

    use_keyboard = args.episode_end in ("space", "both")
    use_timer = args.episode_end in ("timer", "both")
    keyboard: _KeyboardController | None = None
    if use_keyboard and control is None:
        keyboard = _KeyboardController(stop_event, advance_event)
        keyboard.start()

    if control:
        logger.info('Desktop collector: Record starts an episode; Save keeps it; '
                    'Discard/Stop drops the unfinished attempt. auto_home=%s', args.auto_home)
    elif use_keyboard:
        logger.info(
            "Recording up to %d episode(s). From episode 2 onward: go_home()"
            "%s before recording. Press SPACE to save and advance, 'q'/Esc to finish.",
            args.num_episodes,
            f", then rest {args.rest_seconds:.1f}s" if args.rest_seconds > 0 else "",
        )
    else:
        logger.info(
            "Recording %d episode(s) of ~%.1fs each — each episode waits until teleop + "
            "cameras are healthy before capturing. Ctrl+C to abort.",
            args.num_episodes,
            args.episode_seconds,
        )

    try:
        ep_idx = 0
        attempt_idx = 0
        while ep_idx < args.num_episodes:
            if stop_event.is_set():
                break
            attempt_idx += 1
            logger.info("=== Episode %d / %d ===", ep_idx + 1, args.num_episodes)
            advance_event.clear()
            if control:
                control.start_event.clear()
                control.discard_event.clear()
                control.update(phase='warming_up', saved=ep_idx, attempt=attempt_idx,
                               frames=0, seconds=0)
            if quest_reset_event is not None:
                quest_reset_event.set()
            if action_composer is not None:
                action_composer.reset()

            if attempt_idx > 1 and args.auto_home and not args.stub:
                logger.info("Resetting robot to home position...")
                try:
                    sink.go_home()
                except Exception:
                    logger.exception("Failed to move to home position; continuing anyway.")

            if attempt_idx > 1 and args.rest_seconds > 0 and not stop_event.is_set():
                logger.info("Resting %.1fs before recording...", args.rest_seconds)
                stop_event.wait(args.rest_seconds)

            dispatch_enabled.clear()
            action_mirror.reset()
            drained = drain_actions_queue(
                queues.actions_q,
                stop_event=stop_event,
                shutdown_sentinel=_SHUTDOWN,
            )
            if drained:
                logger.debug(
                    "Discarded %d stale teleop command(s) before episode %d.",
                    drained,
                    ep_idx + 1,
                )
            if stop_event.is_set():
                break

            ready = wait_for_teleop_mirror_ready(
                get_observation=(lambda: _preview_observation(sink, control)),
                mirror=action_mirror,
                stop_event=stop_event,
            )
            if not ready:
                logger.info(
                    "Stopped before episode %d — waiting for sensors ended without readiness.",
                    ep_idx + 1,
                )
                break

            if control:
                # Teleop can be positioned between trials without collecting
                # reset motion. Only the Record button starts a new episode.
                dispatch_enabled.set()
                control.update(phase='ready')
                while not stop_event.is_set() and not control.start_event.wait(0.1):
                    _preview_observation(sink, control)
                if stop_event.is_set():
                    break
                control.update(phase='recording')

            logger.info("Sensors ready — recording episode %d at %d Hz.", ep_idx + 1, args.fps)
            dispatch_enabled.set()
            ep_end = time.perf_counter() + args.episode_seconds
            n_frames = 0
            episode_ok = True
            episode_start = time.perf_counter()
            last_ui_update = 0.0
            ticker = RateTicker(dt=1.0 / float(args.fps))

            while not stop_event.is_set():
                if recorder_failed.is_set():
                    raise RuntimeError('데이터 프레임 저장 실패. 현재 시도를 폐기합니다.')
                if use_keyboard and advance_event.is_set():
                    break
                if use_timer and time.perf_counter() >= ep_end:
                    break

                action = action_mirror.snapshot()
                if action is None:
                    ticker.tick()
                    continue

                action_arr = action_to_feature(action, joint_ids)

                try:
                    observation = sink.get_observation()
                    if control:
                        control.observe(observation)
                except Exception:
                    logger.exception(
                        "Observation capture failed; discarding episode and aborting session."
                    )
                    episode_ok = False
                    if control:
                        control.update(phase='error', error='센서 수신이 중단되어 현재 시도를 폐기했습니다.')
                    stop_event.set()
                    break

                frame = {
                    "observation.state": observation.joint_state,
                    "action": action_arr,
                    "task": args.task,
                }
                for cam_name, img in observation.images.items():
                    frame[f"observation.images.{cam_name}"] = img

                try:
                    rec_q.put_nowait(frame)
                    n_frames += 1
                except queue.Full:
                    raise RuntimeError('디스크·영상 저장 속도가 부족해 녹화가 밀렸습니다. 현재 시도를 폐기합니다.')

                ticker.tick()
                if control and time.perf_counter() - last_ui_update >= 0.25:
                    control.update(frames=n_frames, seconds=round(time.perf_counter() - episode_start, 1))
                    last_ui_update = time.perf_counter()

            dispatch_enabled.clear()
            if control:
                episode_ok = episode_ok and not control.discard_event.is_set()
                if episode_ok or not stop_event.is_set():
                    control.update(phase='finalizing')
            logger.info("Episode %d captured %d frames.", ep_idx + 1, n_frames)
            if episode_ok and n_frames > 0:
                logger.info("Finalizing episode %d before continuing...", ep_idx + 1)
                rec_q.put(_SAVE_EPISODE)
                if not episode_finalized_q.get():
                    logger.error("Stopping because episode %d could not be saved.", ep_idx + 1)
                    if control:
                        control.update(phase='error', error='에피소드 저장 실패. 로그와 저장 공간을 확인해 주세요.')
                    break
                ep_idx += 1
                if control:
                    control.update(saved=ep_idx)
            elif n_frames > 0:
                logger.info("Discarding episode %d before continuing...", ep_idx + 1)
                rec_q.put(_DISCARD_EPISODE)
                if not episode_finalized_q.get():
                    logger.error("Stopping because episode %d could not be discarded.", ep_idx + 1)
                    break
            elif not episode_ok:
                logger.error("Episode %d produced no usable frames; not saving.", ep_idx + 1)

    except KeyboardInterrupt:
        logger.info("Interrupted — finalizing dataset.")
        if control:
            rec_q.put(_DISCARD_EPISODE)
            episode_finalized_q.get()

    except Exception as exc:
        if control:
            control.update(phase='error', error=str(exc))
            rec_q.put(_DISCARD_EPISODE)
            episode_finalized_q.get()
        raise

    finally:
        stop_event.set()
        if keyboard is not None:
            keyboard.stop()  # restore terminal mode early
        rec_q.put(None)  # drain sentinel — recorder finishes pending items first

        if ingress_server is not None:
            ingress_server.stop()
        if publisher_process is not None and publisher_process.is_alive():
            publisher_process.terminate()
        if publisher_process is not None:
            publisher_process.join(timeout=3.0)
        if retargeter_thread is not None:
            retargeter_thread.join(timeout=JOIN_TIMEOUT)
        if stub_thread is not None:
            stub_thread.join(timeout=JOIN_TIMEOUT)
        teleop_thread.join(timeout=JOIN_TIMEOUT)

        recorder_thread.join()
        try:
            dataset.stop_image_writer()
        except Exception:
            logger.exception('Failed to stop the image writer')
        try:
            dataset.finalize()  # writes the parquet footers; unloadable without it
        except Exception:
            logger.exception("Failed to finalize the dataset")
            if control:
                control.update(phase='error', error='데이터셋 마무리 실패. 로그와 저장 공간을 확인해 주세요.')

        try:
            num_eps = dataset.num_episodes
        except Exception:
            num_eps = "?"
        logger.info("Dataset now contains %s episode(s) at %s", num_eps, dataset.root)

        if args.push_to_hub:
            try:
                logger.info("Pushing %s to the Hugging Face Hub...", args.repo_id)
                dataset.push_to_hub()
                logger.info("Push complete.")
            except Exception:
                logger.exception("Failed to push to hub")

        sink.close()
        if control:
            control.finish(saved=dataset.num_episodes)


def main() -> None:
    _main_record(sys.argv[1:])


if __name__ == "__main__":
    main()
