"""Helpers of orca_teleop/scripts/eval_am200_policy.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "orca_teleop" / "scripts" / "eval_am200_policy.py"
_spec = importlib.util.spec_from_file_location("eval_am200_policy", SCRIPT)
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

IDS = ["arm.a", "arm.b", "hand.c"]


def test_wilson_interval_brackets_the_rate():
    lo, hi = ev.wilson_interval(20, 50)
    assert lo < 0.4 < hi
    assert ev.wilson_interval(0, 10)[0] == 0.0
    assert ev.wilson_interval(10, 10)[1] == 1.0


def test_action_vector_accepts_named_and_chunked_outputs():
    np.testing.assert_allclose(ev.action_vector({"hand.c": 3, "arm.a": 1, "arm.b": 2}, IDS), [1, 2, 3])
    np.testing.assert_allclose(ev.action_vector(np.ones((1, 5, 3)), IDS), [1, 1, 1])
    with pytest.raises(ValueError):
        ev.action_vector({"arm.a": 1}, IDS)
    with pytest.raises(ValueError):
        ev.action_vector(np.ones(4), IDS)


def test_hold_policy_freezes_the_first_state():
    p = ev.HoldPolicy()
    first = p.select_action({"observation.state": np.array([1.0, 2.0])})
    np.testing.assert_allclose(p.select_action({"observation.state": np.array([9.0, 9.0])}), first)
    p.reset()
    np.testing.assert_allclose(p.select_action({"observation.state": np.array([9.0, 9.0])}), [9, 9])
