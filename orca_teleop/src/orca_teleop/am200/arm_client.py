"""Feetech client for the AM-ARM200's first three joints."""

from __future__ import annotations

import logging
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from orca_core.hardware.feetech_client import FeetechClient

from orca_teleop.am200.canonical import ARM_ROLES

logger = logging.getLogger(__name__)

ARM_JOINT_NAMES: tuple[str, ...] = tuple(r.removeprefix("arm.") for r in ARM_ROLES)


def _read_positions(client: Any) -> np.ndarray:
    """Read motor positions across old and new ORCA Core client APIs."""
    if hasattr(client, "read_position_velocity_current"):
        return np.asarray(client.read_position_velocity_current().position, dtype=np.float64)
    if hasattr(client, "read_pos_vel_cur_sync"):
        positions, _, _ = client.read_pos_vel_cur_sync()
        return np.asarray(positions, dtype=np.float64)
    positions, _, _ = client.read_pos_vel_cur()
    return np.asarray(positions, dtype=np.float64)


def default_arm_config_path() -> Path:
    return Path(__file__).resolve().parents[4] / "configs" / "am-arm200-3dof" / "config.yaml"


def load_arm_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else default_arm_config_path()
    with cfg_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"arm config must be a mapping: {cfg_path}")
    return data


class AmArmFeetech:
    """Position-mode drive for shoulder_pan / shoulder_lift / elbow_flex.

    A taught initial pose becomes the hardware home at the next connection.
    Until one is taught, the measured connect-time pose is used. Model radians
    from IK are applied as offsets from that home::

        motor = motor_home + sign * model_q
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        max_step_rad: float | None = None,
    ) -> None:
        self._cfg = load_arm_config(config_path)
        self._config_path = Path(config_path or default_arm_config_path()).resolve()
        self._joint_names = list(ARM_JOINT_NAMES)
        jmap = self._cfg["joint_to_motor_map"]
        signs = self._cfg.get("joint_signs", {})
        self._motor_ids = [int(jmap[name]) for name in self._joint_names]
        self._signs = np.array(
            [float(signs.get(name, 1)) for name in self._joint_names],
            dtype=np.float64,
        )
        self._max_torque = int(self._cfg.get("max_torque", 500))
        self._speed = int(self._cfg.get("default_speed", 200))
        raw_step = self._cfg.get("max_step_from_home_rad") if max_step_rad is None else max_step_rad
        self._max_step = None if raw_step in (None, False) else float(raw_step)
        self._client: FeetechClient | None = None
        self._motor_home: np.ndarray | None = None
        raw_home = self._cfg.get('home_pose_file')
        self._home_pose_path = (None if not raw_home else
                               (self._config_path.parent / raw_home).resolve())
        self._park_motor: np.ndarray | None = None
        self._feedback_diagnostics = bool(self._cfg.get('feedback_diagnostics', False))
        self._last_feedback_s = -float('inf')
        self._last_motor_target: np.ndarray | None = None

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names)

    @property
    def motor_ids(self) -> list[int]:
        return list(self._motor_ids)

    def connect(self, *, move_to_home: bool = True) -> None:
        park = self._load_home_pose()
        port = str(self._cfg["port"])
        baud = int(self._cfg.get("baudrate", 1_000_000))
        client = FeetechClient(self._motor_ids, port=port, baudrate=baud)
        client.connect()
        self._client = client
        try:
            # FeetechClient.connect already selects servo mode. Re-selecting it
            # disables torque, dropping an arm that was holding its parked pose.
            home = self.read_motor_rad().copy()
            self._motor_home = home
            # Stage the measured pose before enabling torque, never a stale goal.
            self._write_motor_rad(home)
            failed = client.set_torque_enabled(self._motor_ids, True)
            if failed:
                raise RuntimeError(f'Arm torque enable failed: motor IDs {failed}')
            self._park_motor = park
            if move_to_home and park is not None:
                self._move_motor_pose(park)
                self._motor_home = park.copy()
            elif park is None:
                logger.info('No taught initial pose yet; holding the measured pose. '
                            'Save an initial pose from the collector before the next session.')
        except BaseException:
            try:
                # Read/enable/move failures must not release a loaded arm that
                # was already held before this connection.
                self.disconnect(disable_torque=False)
            finally:
                self._client = None
            raise
        logger.info(
            "AmArmFeetech connected on %s ids=%s home=%s",
            port,
            self._motor_ids,
            np.round(self._motor_home, 4).tolist(),
        )

    def disconnect(self, disable_torque: bool = True) -> None:
        if self._client is None:
            return
        try:
            if disable_torque:
                self._client.set_torque_enabled(self._motor_ids, False)
            if disable_torque:
                self._client.disconnect()
            else:
                self._client.disconnect(disable_torque=False)
        finally:
            self._client = None
            self._motor_home = None

    def read_motor_rad(self) -> np.ndarray:
        assert self._client is not None
        positions = _read_positions(self._client)
        if positions.shape != (3,) or not np.isfinite(positions).all():
            raise RuntimeError('Arm motor positions are missing or nonfinite')
        if not getattr(self._client, 'last_read_ok', True):
            raise RuntimeError('Arm motor read failed; cached positions rejected')
        return positions

    def read_model_rad(self) -> np.ndarray:
        """Motor positions expressed as model radians relative to connect-home."""
        assert self._motor_home is not None
        motor = self.read_motor_rad()
        self._log_control_feedback()
        return (motor - self._motor_home) / self._signs

    def _log_control_feedback(self) -> None:
        if not self._feedback_diagnostics or time.monotonic() - self._last_feedback_s < 1.0:
            return
        self._last_feedback_s = time.monotonic()
        try:
            status = self._client.read_sts_control_status()
            for index, motor_id in enumerate(self._motor_ids):
                item = status[motor_id]
                requested = (None if self._last_motor_target is None else
                             round(math.degrees(self._last_motor_target[index]), 1))
                logger.info('Arm feedback | ID=%d %s | requested_deg=%s | servo_goal_deg=%.1f | '
                            'actual_deg=%.1f | torque_on=%s | limit_raw=%s | mode=%s | '
                            'voltage=%.1fV | current=%.0fmA | temp=%sC | status=0x%02x',
                            motor_id, self._joint_names[index], requested,
                            math.degrees(item['goal_rad']), math.degrees(item['position_rad']),
                            item['torque_enabled'], item['torque_limit_raw'], item['mode'],
                            item['voltage_v'], item['current_ma'], item['temperature_c'],
                            item['error_status'])
        except Exception:
            logger.warning('Arm control feedback unavailable; servo settings were not changed', exc_info=True)

    def write_model_rad(
        self,
        model_q: np.ndarray,
        torque: int | None = None,
        speed: int | None = None,
    ) -> None:
        """Command model radians (length 3); clipped relative to connect-home."""
        assert self._client is not None and self._motor_home is not None
        q = np.asarray(model_q, dtype=np.float64).reshape(3)
        if self._max_step is not None:
            q = np.clip(q, -self._max_step, self._max_step)
        motor = self._motor_home + self._signs * q
        self._write_motor_rad(motor, torque=torque, speed=speed)

    def _write_motor_rad(self, motor, torque=None, speed=None) -> None:
        assert self._client is not None
        motor = np.asarray(motor, dtype=np.float64).reshape(3)
        if not np.isfinite(motor).all():
            raise ValueError('Arm command must contain three finite motor angles')
        write = getattr(self._client, "write_positions_sync", None)
        if not callable(write):
            write = self._client.write_desired_pos
        write(
            self._motor_ids,
            motor.astype(np.float32),
            speed=int(self._speed if speed is None else speed),
            torque=int(self._max_torque if torque is None else torque),
        )
        self._last_motor_target = motor.copy()

    def hold_current(self, torque: int | None = None, speed: int | None = None) -> None:
        assert self._motor_home is not None
        # Re-capture soft hold without changing the IK zero.
        motor = self.read_motor_rad()
        self._write_motor_rad(motor, torque=torque, speed=speed)

    def _load_home_pose(self):
        if self._home_pose_path is None or not self._home_pose_path.exists():
            return None
        value = json.loads(self._home_pose_path.read_text())
        if (value.get('version') != 1 or value.get('motor_ids') != self._motor_ids
                or value.get('joint_names') != self._joint_names
                or value.get('joint_signs') != self._signs.tolist()
                or value.get('port') != str(self._cfg['port'])):
            raise ValueError('Saved arm home does not match this arm/configuration; capture it again')
        pose = np.asarray(value.get('motor_rad'), dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError('Saved arm home must contain three finite motor angles')
        return pose

    def save_home_pose(self):
        """Teach the current measured arm pose without changing this session's IK zero."""
        from orca_teleop.collection_control import atomic_json

        if self._home_pose_path is None:
            raise ValueError('Set home_pose_file in the arm config first')
        pose = self.read_motor_rad().copy()
        value = dict(version=1, motor_ids=self._motor_ids, joint_names=self._joint_names,
                     joint_signs=self._signs.tolist(), port=str(self._cfg['port']),
                     motor_rad=pose.tolist(), captured_at_unix_s=time.time())
        self._home_pose_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(self._home_pose_path, value)
        self._park_motor = pose
        logger.info('Saved arm initial/return pose: %s', self._home_pose_path)
        return value

    def _move_motor_pose(self, target) -> None:
        """Cubic joint move, bounded at 15 deg/s and 45 deg/s² by default."""
        start = self.read_motor_rad().copy()
        target = np.asarray(target, dtype=float).reshape(3)
        if not np.isfinite(target).all():
            raise ValueError('Arm return pose must be finite')
        velocity = math.radians(float(self._cfg.get('home_velocity_deg_s', 15)))
        acceleration = math.radians(float(self._cfg.get('home_acceleration_deg_s2', 45)))
        if not (np.isfinite([velocity, acceleration]).all() and velocity > 0 and acceleration > 0):
            raise ValueError('Arm home velocity and acceleration must be positive and finite')
        distance = float(np.max(np.abs(target - start)))
        duration = max(2.5, 1.5 * distance / velocity, math.sqrt(6 * distance / acceleration))
        steps = math.ceil(duration * 30)
        period = duration / steps
        logger.info('Moving arm to initial pose over %.1f s', duration)
        for step in range(1, steps + 1):
            u = step / steps
            self._write_motor_rad(start + u*u*(3 - 2*u) * (target-start))
            time.sleep(period)
        deadline = time.monotonic() + 2.0
        while True:
            error = float(np.max(np.abs(self.read_motor_rad() - target)))
            if error <= math.radians(3):
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f'Arm did not reach initial pose (max error {math.degrees(error):.1f} deg)')
            time.sleep(0.05)

    def return_home(self) -> None:
        assert self._motor_home is not None
        self._move_motor_pose(self._park_motor if self._park_motor is not None else self._motor_home)
