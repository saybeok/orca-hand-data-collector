"""Selected-joint tensioning must isolate the motor bus, not just the sweep."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

import orca_core
from orca_core.hand_config import OrcaHandConfig
from orca_core.joint_position import OrcaJointPositions
from orca_core.maintenance import tensioning

spec = importlib.util.spec_from_file_location(
    "tension_cli", Path(__file__).resolve().parents[1] / "scripts/tension.py",
)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


@pytest.fixture
def config_path(tmp_path):
    source = Path(orca_core.__file__).parent / "models/v1/orcahand-right/config.yaml"
    data = yaml.safe_load(source.read_text())
    # Match the tested hand's IP motor ID, while keeping a valid full chain.
    data["joint_to_motor_map"]["thumb_dip"] = 15
    data["joint_to_motor_map"]["index_mcp"] = 2
    data.update(port="mock", motor_type="feetech", baudrate=1_000_000)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    cal = tmp_path / "calibration.yaml"
    cal.write_text("{}\n")
    before = path.read_bytes(), cal.read_bytes()
    ordering = OrcaJointPositions._default_joint_ids
    yield path
    OrcaJointPositions._default_joint_ids = ordering
    assert (path.read_bytes(), cal.read_bytes()) == before


class Clock:
    now = 0.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def install_bus_audit(monkeypatch, expected_ids, fail_read=False):
    calls, clients = [], []
    connect = cli.connect_hand

    def audited_connect(hand):
        connect(hand)
        client = hand.motor_client
        assert set(client.motor_ids) == expected_ids
        clients.append(client)
        for mid in client.motor_ids:
            client._pos[mid] = .18
        for name in ("set_torque_enabled", "set_operating_mode",
                     "write_desired_pos", "write_desired_current"):
            original = getattr(client, name)

            def audited(ids, *args, _name=name, _original=original, **kwargs):
                assert set(ids) <= expected_ids, f"Unselected motor reached by {_name}: {ids}"
                calls.append((_name, list(ids), np.asarray(args[0]).copy()))
                return _original(ids, *args, **kwargs)

            monkeypatch.setattr(client, name, audited)
        if fail_read:
            def failed_read():
                raise RuntimeError("fresh motor read failed")
            monkeypatch.setattr(hand, "_read_motor_pos_for_offsets", failed_read)

    monkeypatch.setattr(cli, "connect_hand", audited_connect)

    def interrupt_on_hold(event):
        if event.get("phase") == "holding":
            raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_print_progress", interrupt_on_hold)
    monkeypatch.setattr(tensioning, "time", Clock())
    return calls, clients


@pytest.mark.parametrize("wind", [False, True])
def test_selected_tension_only_ever_controls_id15(config_path, monkeypatch, wind):
    args = ["tension.py", str(config_path), "--mock", "--joints", "thumb_dip"]
    if not wind:
        args.append("--no-move-motors")
    monkeypatch.setattr("sys.argv", args)
    calls, clients = install_bus_audit(monkeypatch, {15})
    assert cli.main() == 0
    writes = [values for name, ids, values in calls if name == "write_desired_pos"]
    assert writes
    if not wind:
        # Preload the measured pose before any mode change enables torque.
        assert calls[0][0] == "write_desired_pos"
        assert len(writes) == 1
        np.testing.assert_allclose(writes[0], [.18])
    else:
        values = np.array(writes).ravel()
        assert values.min() < .18
        assert values.max() > values.min()
    assert not clients[0].is_connected
    assert not any(clients[0]._torque_enabled.values())


@pytest.mark.parametrize("wind", [False, True])
def test_failed_hold_read_never_enables_or_commands_selected_motor(config_path, monkeypatch, wind):
    args = ["tension.py", str(config_path), "--mock", "--joints", "thumb_dip"]
    if not wind:
        args.append("--no-move-motors")
    monkeypatch.setattr("sys.argv", args)
    calls, clients = install_bus_audit(monkeypatch, {15}, fail_read=True)
    with pytest.raises(RuntimeError, match="fresh motor read failed"):
        cli.main()
    assert not any(name in ("write_desired_pos", "set_operating_mode") for name, _, _ in calls)
    assert not any(bool(values) for name, _, values in calls if name == "set_torque_enabled")
    assert not clients[0].is_connected


def test_invalid_encoder_frame_is_not_preloaded_or_held(config_path, monkeypatch):
    from types import SimpleNamespace
    calls = []

    def reject(ids, positions):
        calls.append(list(ids))
        raise ValueError("outside command frame")

    hand = SimpleNamespace(
        config=SimpleNamespace(motor_ids=[15]),
        motor_client=SimpleNamespace(validate_position_frame=reject),
        _read_motor_pos_for_offsets=lambda: np.array([6.08683586]),
        _set_motor_pos=lambda positions: pytest.fail("must not preload an uncommandable pose"),
    )
    with pytest.raises(ValueError, match="command frame"):
        cli._preload_current_position(hand)
    assert calls == [[15]]


def test_no_selection_keeps_the_full_chain(config_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["tension.py", str(config_path), "--mock", "--no-move-motors"])
    calls, clients = install_bus_audit(monkeypatch, set(range(1,18)))
    assert cli.main() == 0
    touched = {mid for _, ids, _ in calls for mid in ids}
    assert touched == set(range(1,18))
    assert not clients[0].is_connected


def test_selected_config_preserves_mapping_inversion_and_original_config(config_path):
    original = OrcaHandConfig.from_config_path(str(config_path))
    selected = cli._selected_joint_config(original, ["thumb_dip", "index_abd", "thumb_dip"])
    assert selected.joint_ids == ["thumb_dip", "index_abd"]
    assert selected.motor_ids == [14, 15]
    for joint in selected.joint_ids:
        assert selected.joint_to_motor_map[joint] == original.joint_to_motor_map[joint]
        assert selected.joint_inversion_dict[joint] == original.joint_inversion_dict[joint]
        assert selected.joint_roms_dict[joint] == original.joint_roms_dict[joint]
    assert len(original.joint_ids) == 17
    assert len(original.motor_ids) == 17


@pytest.mark.parametrize("joints", [[], ["thumb_ip"]])
def test_invalid_selection_is_rejected(config_path, joints):
    with pytest.raises(ValueError):
        cli._selected_joint_config(OrcaHandConfig.from_config_path(str(config_path)), joints)


def test_wrist_winding_is_rejected_before_connect(config_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["tension.py", str(config_path), "--mock", "--joints", "wrist"])
    monkeypatch.setattr(cli, "connect_hand", lambda hand: pytest.fail("must not connect"))
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
