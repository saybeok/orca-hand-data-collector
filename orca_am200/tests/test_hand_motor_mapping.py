"""Physical thumb axis correspondence confirmed with isolated probes, 2026-10-27.

These check command routing and calibration units, not real contact accuracy.
"""
from pathlib import Path

import numpy as np
import pytest
import yaml
from orca_core import OrcaHand, OrcaJointPositions

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/orcahand-right-v1-feetech/config.yaml"


@pytest.fixture
def hand(tmp_path):
    ordering = OrcaJointPositions._default_joint_ids
    config = tmp_path / "config.yaml"
    config.write_bytes(CONFIG.read_bytes())
    cfg = yaml.safe_load(config.read_text())
    cal = {"motor_limits": {}, "joint_to_motor_ratios": {}, "calibrated": True}
    for joint, signed_motor in cfg["joint_to_motor_map"].items():
        motor = abs(signed_motor)
        low, high = cfg["joint_roms"][joint]
        cal["motor_limits"][motor] = [-5., -2.5]
        cal["joint_to_motor_ratios"][motor] = 2.5 / (high - low)
    config.with_name("calibration.yaml").write_text(yaml.safe_dump(cal))
    files = (config, config.with_name("calibration.yaml"))
    before = [path.read_bytes() for path in files]
    h = OrcaHand(str(config))
    h._wrap_offsets_dict = dict.fromkeys(h.config.motor_ids, 0.0)
    try:
        yield h
    finally:
        OrcaJointPositions._default_joint_ids = ordering
        assert [path.read_bytes() for path in files] == before


@pytest.mark.parametrize("joint,motor", [
    ("thumb_mcp", 13),  # Model's palm-root axis; user calls this CMC.
    ("thumb_pip", 16),  # Model's next hinge; user calls this MCP.
    ("thumb_abd", 14),
    ("thumb_dip", 15),
])
def test_confirmed_thumb_axes_route_only_the_selected_motor(hand, joint, motor):
    assert hand.config.joint_to_motor_map[joint] == motor
    low, high = hand.config.joint_roms_dict[joint]
    for angle in (low, (low + high) / 2, high):
        target = hand._joint_to_motor_pos({joint: angle})
        written = {mid: value for mid, value in zip(hand.config.motor_ids, target)
                   if value is not None}
        assert list(written) == [motor]
        motor_lo, motor_hi = hand.motor_limits_dict[motor]
        fraction = (angle - low) / (high - low)
        assert written[motor] == pytest.approx(motor_lo + fraction * (motor_hi - motor_lo))


@pytest.mark.parametrize("joint,motor", [("thumb_mcp", 13), ("thumb_pip", 16)])
def test_corrected_motor_ratios_cover_the_new_joint_rom_and_roundtrip(hand, joint, motor):
    low, high = hand.config.joint_roms_dict[joint]
    motor_lo, motor_hi = hand.motor_limits_dict[motor]
    assert hand.joint_to_motor_ratios_dict[motor] == pytest.approx(
        (motor_hi - motor_lo) / (high - low), rel=1e-12)
    for fraction in (0, .25, .5, .75, 1):
        feedback = np.array([np.mean(hand.motor_limits_dict[mid]) for mid in hand.config.motor_ids])
        feedback[hand.config.motor_id_to_idx_dict[motor]] = motor_lo + fraction * (motor_hi - motor_lo)
        decoded = hand._motor_to_joint_pos(feedback)
        assert decoded[joint] == pytest.approx(low + fraction * (high - low))
        command = hand._joint_to_motor_pos({joint: decoded[joint]})
        assert command[hand.config.motor_id_to_idx_dict[motor]] == pytest.approx(
            feedback[hand.config.motor_id_to_idx_dict[motor]])
