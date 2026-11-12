import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'act_hw_runner', Path(__file__).parents[1] / 'scripts' / 'run_act_hw.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_reference_limiter_respects_velocity_acceleration_and_joint_stops():
    start = np.zeros(20)
    limiter = runner.CommandLimiter(start, (np.full(3, -0.1), np.full(3, 0.1)), 30, 90)
    previous_v = np.zeros(3)
    for _ in range(120):
        result = limiter.advance(np.ones(20), 1 / 30)
        assert np.all(np.abs(limiter.v) <= np.deg2rad(30) + 1e-9)
        if np.all(np.abs(result[:3]) < 0.1):
            assert np.max(np.abs(limiter.v - previous_v)) <= np.deg2rad(90) / 30 + 1e-9
        assert np.all(np.abs(result[:3]) <= 0.1)
        previous_v = limiter.v.copy()
    np.testing.assert_array_equal(start, np.zeros(20))


def test_nonfinite_policy_action_rejected_before_command():
    limiter = runner.CommandLimiter(np.zeros(20), (np.full(3, -1), np.full(3, 1)), 30, 90)
    target = np.zeros(20)
    target[5] = np.nan
    with pytest.raises(ValueError, match='command not sent'):
        limiter.advance(target, 1 / 30)
    np.testing.assert_array_equal(limiter.q, np.zeros(20))


def test_control_interval_rejects_stale_targets():
    limiter = runner.CommandLimiter(np.zeros(20), (np.full(3, -1), np.full(3, 1)), 30, 90)
    with pytest.raises(ValueError, match='command not sent'):
        limiter.advance(np.ones(20), 0.5)
