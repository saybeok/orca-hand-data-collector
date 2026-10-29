"""Hardware sink for AM-ARM200 (3 DoF) + ORCA hand on separate Feetech buses."""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from orca_core import OrcaHand, OrcaJointPositions

from orca_teleop.am200.arm_client import AmArmFeetech, default_arm_config_path, load_arm_config
from orca_teleop.am200.canonical import ARM_ROLES, CanonicalJointMap, RobotJointCommand
from orca_teleop.am200.sim_env import resolve_model_path
from orca_teleop.cameras import CameraManager, OpenCVCameraConfig
from orca_teleop.constants import MOTION_NUM_STEPS
from orca_teleop.pipeline import _SHUTDOWN, RecordableSink, SinkObservation, connect_orca_hand

logger = logging.getLogger(__name__)

STALE_COMMAND_WARN_S = 2.0


def _check_bus_ports(hand_port: str | None, arm_port: str | None) -> None:
    """Refuse to start unless each enabled bus has its own, present adapter.

    Both buses use motor IDs 1-3, so a shared port would drive index-finger
    servos as arm joints. A missing hand port is fatal too: ``OrcaHand.connect``
    would otherwise auto-detect a port (possibly the arm's) and rewrite the config.
    """
    for name, port in (("hand", hand_port), ("arm", arm_port)):
        if port is not None and not os.path.exists(port):
            raise RuntimeError(f"{name} serial port {port} is not connected")
    if hand_port is not None and arm_port is not None:
        if os.path.realpath(hand_port) == os.path.realpath(arm_port):
            raise RuntimeError(
                f"hand and arm both resolve to {os.path.realpath(hand_port)}; "
                "plug in both USB adapters and fix the port in each config"
            )


def _enable_sync_hand_writes(hand: Any) -> None:
    """Use one broadcast packet per hand target on legacy ORCA Core releases."""
    client = getattr(hand, "_motor_client", None)
    sync_write = getattr(client, "write_positions_sync", None)
    if not callable(sync_write):
        raise RuntimeError(
            "This ORCA Core version lacks Feetech sync-write support; "
            "refusing high-rate full-hand commands"
        )

    def write_desired_pos(
        motor_ids: Any,
        positions: Any,
        speed: int | None = None,
        torque: int | None = None,
    ) -> None:
        sync_write(motor_ids, positions, speed=speed, torque=torque)

    client.write_desired_pos = write_desired_pos


class Am200HardwareSink(RecordableSink):
    """Drives the physical arm + hand from the canonical 20-DoF command.

    Arm joints use connect-time motor angles as hardware zero; model radians
    from IK are applied as signed offsets. Hand joints use ``orca_core`` physical
    degrees via :class:`OrcaHand`.
    """

    def __init__(
        self,
        hand_config_path: str | Path | None = None,
        arm_config_path: str | Path | None = None,
        model_path: str | Path | None = None,
        camera_configs: list[OpenCVCameraConfig] | None = None,
        enable_arm: bool = True,
        enable_hand: bool = True,
        arm_max_step_rad: float | None = None,
        hand_feedback_debug: bool = False,
    ) -> None:
        self._hand_config_path = hand_config_path
        self._arm_config_path = Path(arm_config_path) if arm_config_path else default_arm_config_path()
        self._arm_max_step_rad = arm_max_step_rad
        self._model_path = resolve_model_path(model_path)
        self._enable_arm = enable_arm
        self._enable_hand = enable_hand
        self._cameras = CameraManager(camera_configs or [])

        self._hand: OrcaHand | None = None
        self._arm: AmArmFeetech | None = None
        self._map: CanonicalJointMap | None = None
        self._target: np.ndarray | None = None
        self._last_command_ns: int | None = None
        self._stale_logged = False
        self._hand_feedback_debug = hand_feedback_debug
        self._last_hand_feedback_s = -float("inf")

    def connect(self) -> None:
        import mujoco

        if self._hand_config_path is None:
            env_path = os.environ.get("ORCA_MODEL_PATH")
            if not env_path:
                raise RuntimeError("Pass hand_config_path or set $ORCA_MODEL_PATH")
            self._hand_config_path = env_path

        hand_port = None
        if self._enable_hand:
            with Path(self._hand_config_path).open(encoding="utf-8") as f:
                hand_port = str(yaml.safe_load(f)["port"])
        arm_port = str(load_arm_config(self._arm_config_path)["port"]) if self._enable_arm else None
        _check_bus_ports(hand_port, arm_port)

        model = mujoco.MjModel.from_xml_path(str(self._model_path))
        joint_map_path = self._model_path.parent / "joint_map.json"
        self._map = CanonicalJointMap.load(joint_map_path, model)
        self._target = np.zeros(self._map.dof, dtype=np.float64)

        if self._enable_hand:
            hand = OrcaHand(str(self._hand_config_path))
            ok, msg = connect_orca_hand(hand)
            if not ok:
                raise RuntimeError(f"hand connect failed: {msg}")
            _enable_sync_hand_writes(hand)
            hand.init_joints()
            self._hand = hand
            self._target[self._map.hand_slice] = self._map.hand_from_physical(self.home_position())

        if self._enable_arm:
            arm = AmArmFeetech(self._arm_config_path, max_step_rad=self._arm_max_step_rad)
            arm.connect()
            self._arm = arm

        self._cameras.open()
        self._cameras.ensure_live()
        logger.info(
            "Am200HardwareSink connected: arm=%s hand=%s dof=%d",
            self._enable_arm,
            self._enable_hand,
            self._map.dof,
        )

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def joint_ids(self) -> list[str]:
        assert self._map is not None
        return list(self._map.roles)

    @property
    def retarget_model_path(self) -> str | None:
        return None if self._hand_config_path is None else str(self._hand_config_path)

    @property
    def camera_shapes(self) -> dict[str, tuple[int, int, int]]:
        return self._cameras.shapes

    def get_observation(self) -> SinkObservation:
        return SinkObservation(
            joint_state=self._read_joint_state(),
            images=self._cameras.capture(),
        )

    def _read_joint_state(self) -> np.ndarray:
        assert self._map is not None
        state = np.zeros(self._map.dof, dtype=np.float32)
        if self._arm is not None:
            state[self._map.arm_slice] = self._arm.read_model_rad().astype(np.float32)
        if self._hand is not None:
            phys = self._hand.get_joint_position()
            state[self._map.hand_slice] = self._map.hand_from_physical(phys).astype(np.float32)
        return state

    def set_target(self, action: RobotJointCommand | OrcaJointPositions) -> None:
        assert self._map is not None and self._target is not None
        if isinstance(action, RobotJointCommand):
            if tuple(action.joint_ids) != self._map.roles:
                raise ValueError("RobotJointCommand joint_ids mismatch")
            self._target = self._map.clip(action.position)
        elif isinstance(action, OrcaJointPositions):
            target = self._target.copy()
            target[self._map.hand_slice] = self._map.hand_from_physical(action)
            self._target = self._map.clip(target)
        else:
            raise TypeError(f"unsupported action type {type(action).__name__}")
        self._last_command_ns = time.monotonic_ns()
        self._stale_logged = False

    def dispatch_action(self, action: RobotJointCommand | OrcaJointPositions) -> None:
        self.set_target(action)
        self._apply()

    def _apply(self) -> None:
        assert self._map is not None and self._target is not None
        if self._arm is not None:
            self._arm.write_model_rad(self._target[self._map.arm_slice])
        if self._hand is not None:
            phys = self._map.hand_to_physical(self._target[self._map.hand_slice])
            self._hand.set_joint_positions(phys, num_steps=MOTION_NUM_STEPS)
            if self._hand_feedback_debug:
                self._log_hand_feedback(phys)

    def _log_hand_feedback(self, command: dict[str, float]) -> None:
        """Read on the existing writer thread/bus, at most once per second.

        These are motor-encoder-derived joint estimates, not independent
        measurements of the tendon-driven link angle. Reject cached failures.
        """
        now = time.monotonic()
        if now - self._last_hand_feedback_s < 1.0:
            return
        self._last_hand_feedback_s = now
        try:
            measured = self._hand.get_joint_position()
            if not getattr(self._hand._motor_client, "last_read_ok", False):
                raise RuntimeError("motor read failed; cached values rejected")
            target = float(command["thumb_dip"])
            estimate = measured.data.get("thumb_dip")
            if estimate is None or not np.isfinite(estimate):
                raise RuntimeError("thumb_dip encoder estimate unavailable")
            motor = self._hand.config.joint_to_motor_map["thumb_dip"]
            logger.info(
                "Hand feedback | thumb_dip(ID %s) | cmd_deg=%.2f | "
                "encoder_joint_deg=%.2f | error_deg=%.2f",
                motor, target, estimate, target - estimate,
            )
            # get_joint_position() already reads the full hand. Reuse that
            # sample so finger comparisons add no serial transactions.
            for finger in ("thumb", "index", "middle", "ring", "pinky"):
                suffixes = ("mcp", "abd", "pip", "dip") if finger == "thumb" else ("abd", "mcp", "pip")
                joints = [f"{finger}_{suffix}" for suffix in suffixes]
                if not all(joint in command for joint in joints):
                    continue
                targets = np.array([command[joint] for joint in joints], dtype=float)
                estimates = np.array([measured.data.get(joint, np.nan) for joint in joints], dtype=float)
                if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(estimates)):
                    logger.warning("Hand finger feedback unavailable: %s has missing/nonfinite angles", finger)
                    continue
                motors = [self._hand.config.joint_to_motor_map[joint] for joint in joints]
                logger.info(
                    "Hand finger feedback | finger=%s | joints=%s | motor_ids=%s | "
                    "cmd_deg=%s | encoder_joint_deg=%s | error_deg=%s",
                    finger, ",".join(suffixes), motors,
                    np.round(targets, 2).tolist(), np.round(estimates, 2).tolist(),
                    np.round(targets - estimates, 2).tolist(),
                )
        except Exception as exc:
            logger.warning("Hand feedback unavailable: %s", exc)

    def home_position(self) -> OrcaJointPositions:
        assert self._hand is not None or self._hand_config_path is not None
        if self._hand is not None:
            cfg = self._hand.config
        else:
            cfg = OrcaHand(str(self._hand_config_path)).config
        positions = dict(cfg.neutral_position)
        positions["wrist"] = 0.0
        return OrcaJointPositions(positions)

    def go_home(self, duration_s: float = 2.5, hz: float = 30.0) -> None:
        """Glide to the recording home; a single command would snap the arm at full servo speed."""
        assert self._map is not None and self._target is not None
        home = np.zeros(self._map.dof, dtype=np.float64)
        if self._hand is not None:
            home[self._map.hand_slice] = self._map.hand_from_physical(self.home_position())
        start = self._target.copy()
        steps = max(1, round(duration_s * hz))
        for k in range(1, steps + 1):
            u = k / steps
            blend = u * u * (3.0 - 2.0 * u)
            self._target = start + blend * (home - start)
            self._apply()
            time.sleep(1.0 / hz)

    def run_loop(
        self,
        actions_q: queue.Queue[Any],
        stop_event: threading.Event,
    ) -> None:
        while not stop_event.is_set():
            try:
                item = actions_q.get(timeout=0.05)
            except queue.Empty:
                self._warn_if_stale()
                continue
            if item is _SHUTDOWN:
                break
            try:
                self.dispatch_action(item)
            except Exception:
                logger.exception("Am200HardwareSink rejected/failed an action")

    def _warn_if_stale(self) -> None:
        if self._stale_logged or self._last_command_ns is None:
            return
        idle_s = (time.monotonic_ns() - self._last_command_ns) / 1e9
        if idle_s >= STALE_COMMAND_WARN_S:
            logger.warning("No action for %.1fs — holding last target", idle_s)
            self._stale_logged = True

    def close(self) -> None:
        try:
            self._cameras.close()
            if self._hand is not None:
                self._hand.disable_torque()
                self._hand.disconnect()
            if self._arm is not None:
                self._arm.disconnect(disable_torque=True)
        except Exception:
            logger.exception("Am200HardwareSink.close() error")
        finally:
            self._hand = None
            self._arm = None
