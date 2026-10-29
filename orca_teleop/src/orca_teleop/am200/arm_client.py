"""Feetech client for the AM-ARM200's first three joints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from orca_core.constants import CURRENT_BASED_POSITION
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

    On :meth:`connect`, current motor angles become the hardware home. Model
    radians from IK are applied as offsets from that home::

        motor = motor_home + sign * model_q
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        max_step_rad: float | None = None,
    ) -> None:
        self._cfg = load_arm_config(config_path)
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

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names)

    @property
    def motor_ids(self) -> list[int]:
        return list(self._motor_ids)

    def connect(self) -> None:
        port = str(self._cfg["port"])
        baud = int(self._cfg.get("baudrate", 1_000_000))
        client = FeetechClient(self._motor_ids, port=port, baudrate=baud)
        client.connect()
        try:
            client.set_operating_mode(self._motor_ids, CURRENT_BASED_POSITION)
            client.set_torque_enabled(self._motor_ids, True)
            home = _read_positions(client).copy()
        except Exception:
            try:
                client.set_torque_enabled(self._motor_ids, False)
            finally:
                client.disconnect()
            raise
        self._client = client
        self._motor_home = home
        logger.info(
            "AmArmFeetech connected on %s ids=%s home=%s",
            port,
            self._motor_ids,
            np.round(home, 4).tolist(),
        )

    def disconnect(self, disable_torque: bool = True) -> None:
        if self._client is None:
            return
        try:
            if disable_torque:
                self._client.set_torque_enabled(self._motor_ids, False)
            self._client.disconnect()
        finally:
            self._client = None
            self._motor_home = None

    def read_motor_rad(self) -> np.ndarray:
        assert self._client is not None
        return _read_positions(self._client)

    def read_model_rad(self) -> np.ndarray:
        """Motor positions expressed as model radians relative to connect-home."""
        assert self._motor_home is not None
        motor = self.read_motor_rad()
        return (motor - self._motor_home) / self._signs

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
        write = getattr(self._client, "write_positions_sync", None)
        if not callable(write):
            write = self._client.write_desired_pos
        write(
            self._motor_ids,
            motor.astype(np.float32),
            speed=int(self._speed if speed is None else speed),
            torque=int(self._max_torque if torque is None else torque),
        )

    def hold_current(self, torque: int | None = None, speed: int | None = None) -> None:
        assert self._motor_home is not None
        # Re-capture soft hold without changing the IK zero.
        motor = self.read_motor_rad()
        model = (motor - self._motor_home) / self._signs
        self.write_model_rad(model, torque=torque, speed=speed)
