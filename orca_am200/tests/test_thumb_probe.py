"""Exercise the real core joint/calibration path with a single-motor fake bus."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from orca_core.joint_position import OrcaJointPositions

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("check_thumb_ip", ROOT / "scripts/check_thumb_ip.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.fixture
def isolated_probe(monkeypatch, request, tmp_path):
    ordering = OrcaJointPositions._default_joint_ids
    source = ROOT / "configs/orcahand-right-v1-feetech/config.yaml"
    cfg = tmp_path / "config.yaml"
    cfg.write_bytes(source.read_bytes())
    cal = cfg.with_name("calibration.yaml")
    joint = getattr(request, "param", "thumb_dip")
    # Diagnostic tests must not depend on this machine having a valid live
    # calibration (a user may be repairing exactly that missing calibration).
    doc = yaml.safe_load(source.with_name("calibration.yaml").read_text())
    config_doc = yaml.safe_load(cfg.read_text())
    mid = abs(config_doc["joint_to_motor_map"][joint])
    rom = config_doc["joint_roms"][joint]
    doc["motor_limits"][mid] = [-5., -2.5]
    doc["joint_to_motor_ratios"][mid] = 2.5 / (rom[1] - rom[0])
    cal.write_text(yaml.safe_dump(doc))
    before = cfg.read_bytes(), cal.read_bytes()
    hand = probe.make_probe(str(cfg), joint=joint)
    motor = hand.config.joint_to_motor_map[joint]
    assert hand.config.motor_ids == [motor]
    assert hand.config.joint_ids == [joint]
    low, high = hand.config.joint_roms_dict[joint]
    offset = high - 19 if hand.config.joint_inversion_dict[joint] else 19 - low
    initial_motor = hand.motor_limits_dict[motor][0] + offset * hand.joint_to_motor_ratios_dict[motor]

    class Client:
        def __init__(self):
            self.is_connected = False
            self.last_read_ok = True
            self.position = initial_motor
            self.writes = []
            self.torque_calls = []
            self.current_calls = []
            self.fail_after_writes = None

        def connect(self):
            self.is_connected = True

        def disconnect(self):
            self.is_connected = False

        def read_position_velocity_current(self):
            self.last_read_ok = (self.fail_after_writes is None
                                 or len(self.writes) < self.fail_after_writes)
            return SimpleNamespace(position=np.array([self.position]))

        def write_desired_pos(self, ids, positions):
            assert list(ids) == [motor]
            self.writes.append(float(positions[0]))
            self.position = float(positions[0])

        def write_desired_current(self, ids, currents):
            assert list(ids) == [motor]
            self.current_calls.append(list(currents))

        def set_torque_enabled(self, ids, enabled):
            assert list(ids) == [motor]
            self.torque_calls.append(enabled)
            return []

    client = Client()
    monkeypatch.setattr(hand, "_resolve_motor_driver", lambda port: True)
    monkeypatch.setattr(hand, "_create_motor_client", lambda: client)
    monkeypatch.setattr(probe.time, "sleep", lambda seconds: None)
    assert hand.connect(interactive=False)[0]
    try:
        yield hand, client
    finally:
        hand.disconnect()
        OrcaJointPositions._default_joint_ids = ordering
        assert (cfg.read_bytes(), cal.read_bytes()) == before


@pytest.mark.parametrize("delta", [40, 90])
def test_only_thumb_motor_moves_slowly_and_returns_without_recalibration(isolated_probe, delta):
    hand, client = isolated_probe
    probe.run_probe(hand, delta=delta, speed=15, hold=3)
    ratio = hand.joint_to_motor_ratios_dict[15]
    angles = ((np.array(client.writes) - hand.motor_limits_dict[15][0]) / ratio
              + hand.config.joint_roms_dict["thumb_dip"][0])
    assert angles[0] == pytest.approx(19)
    assert max(angles) == pytest.approx(19 + delta)
    assert angles[-1] == pytest.approx(19)
    assert np.max(np.abs(np.diff(angles))) <= 15 * .05 + 1e-9
    assert min(angles) >= hand.config.joint_roms_dict['thumb_dip'][0]
    assert max(angles) <= hand.config.joint_roms_dict['thumb_dip'][1]
    assert client.torque_calls == [True]
    assert hand._wrap_offsets_dict == {15: 0}


def test_stale_start_read_never_enables_or_moves_motor(isolated_probe):
    hand, client = isolated_probe
    client.fail_after_writes = 0
    with pytest.raises(RuntimeError, match="motor position read failed"):
        probe.run_probe(hand, delta=40, speed=15, hold=3)
    assert not client.writes
    assert not client.torque_calls


def test_failed_read_during_motion_stops_without_blind_return(isolated_probe):
    hand, client = isolated_probe
    client.fail_after_writes = 2
    with pytest.raises(RuntimeError, match="캐시"):
        probe.run_probe(hand, delta=40, speed=15, hold=3)
    assert len(client.writes) == 2  # Measured hold, then the first small step.


@pytest.mark.parametrize("delta,speed,hold", [(float("nan"),15,3), (133,15,3), (40,0,3), (40,100,3), (40,15,-1)])
def test_invalid_motion_settings_do_not_move(isolated_probe, delta, speed, hold):
    hand, client = isolated_probe
    with pytest.raises(ValueError):
        probe.run_probe(hand, delta, speed, hold)
    assert not client.writes
    assert not client.torque_calls


@pytest.mark.parametrize("isolated_probe", ["thumb_pip", "thumb_mcp", "middle_abd"], indirect=True)
def test_selected_joint_probe_routes_one_motor_preserves_polarity_and_returns(isolated_probe):
    hand, client = isolated_probe
    joint = hand.config.joint_ids[0]
    motor = hand.config.motor_ids[0]
    probe.run_probe(hand, delta=20, speed=5, hold=3)
    angles = np.array([hand._motor_to_joint_pos(np.array([p]))[joint] for p in client.writes])
    low, high = hand.config.joint_roms_dict[joint]
    assert angles[0] == pytest.approx(19)
    assert angles.max() == pytest.approx(min(39, high))
    assert angles[-1] == pytest.approx(19)
    assert np.max(np.abs(np.diff(angles))) <= 5 * .05 + 1e-8
    assert client.torque_calls == [True]
    assert hand._wrap_offsets_dict == {motor: 0}


@pytest.mark.parametrize("joint", ["wrist", "not_a_joint"])
def test_joint_probe_rejects_nonfinger_names(joint):
    cfg = ROOT / "configs/orcahand-right-v1-feetech/config.yaml"
    with pytest.raises(ValueError, match="손가락 관절"):
        probe.make_probe(str(cfg), joint=joint)
