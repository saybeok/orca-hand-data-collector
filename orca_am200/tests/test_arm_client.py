"""Compatibility tests for the AM-ARM200 Feetech client wrapper."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from orca_teleop.am200.arm_client import AmArmFeetech, _read_positions


def test_read_positions_uses_new_orca_core_api():
    class Client:
        def read_position_velocity_current(self):
            return SimpleNamespace(position=np.array([1.0, 2.0, 3.0]))

    np.testing.assert_array_equal(_read_positions(Client()), [1.0, 2.0, 3.0])


def test_read_positions_uses_legacy_sync_api():
    class Client:
        def read_pos_vel_cur_sync(self):
            return np.array([1.0, 2.0, 3.0]), np.zeros(3), np.zeros(3)

    np.testing.assert_array_equal(_read_positions(Client()), [1.0, 2.0, 3.0])


def test_read_positions_falls_back_to_legacy_individual_api():
    class Client:
        def read_pos_vel_cur(self):
            return np.array([1.0, 2.0, 3.0]), np.zeros(3), np.zeros(3)

    np.testing.assert_array_equal(_read_positions(Client()), [1.0, 2.0, 3.0])


def test_write_model_rad_prefers_single_sync_packet():
    calls = []

    class Client:
        def write_positions_sync(self, motor_ids, positions, **kwargs):
            calls.append((list(motor_ids), np.asarray(positions), kwargs))

        def write_desired_pos(self, *_args, **_kwargs):
            raise AssertionError("individual writer must not be used")

    arm = AmArmFeetech.__new__(AmArmFeetech)
    arm._client = Client()
    arm._motor_home = np.array([1.0, 2.0, 3.0])
    arm._motor_ids = [1, 2, 3]
    arm._signs = np.array([-1.0, 1.0, 1.0])
    arm._max_step = None
    arm._speed = 120
    arm._max_torque = 350

    arm.write_model_rad(np.array([0.1, 0.2, -0.3]))

    assert calls[0][0] == [1, 2, 3]
    np.testing.assert_allclose(calls[0][1], [0.9, 2.2, 2.7])
    assert calls[0][2] == {"speed": 120, "torque": 350}
