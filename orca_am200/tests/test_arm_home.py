"""Loaded-arm startup, taught initial pose, and torque-preserving shutdown."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from orca_teleop.am200 import arm_client
from orca_teleop.am200.arm_client import AmArmFeetech
from orca_teleop.am200.hw_sink import Am200HardwareSink


@pytest.fixture
def setup_arm(tmp_path, monkeypatch):
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({
        'port': '/dev/test-arm',
        'joint_to_motor_map': {'shoulder_pan': 1, 'shoulder_lift': 2, 'elbow_flex': 3},
        'joint_signs': {'shoulder_pan': -1, 'shoulder_lift': 1, 'elbow_flex': 1},
        'home_pose_file': 'initial_pose.json',
        'home_velocity_deg_s': 15, 'home_acceleration_deg_s2': 45,
    }))
    clients = []

    class Client:
        def __init__(self, ids, **kwargs):
            self.positions = np.array([-1., -2., -3.])
            self.calls = []
            self.last_read_ok = True
            self.enable_failed = []
            clients.append(self)

        def connect(self):
            self.calls.append('connect')

        def read_position_velocity_current(self):
            self.calls.append('read')
            return SimpleNamespace(position=self.positions.copy())

        def write_positions_sync(self, ids, positions, **kwargs):
            self.calls.append(('write', np.asarray(positions).copy()))
            self.positions = np.asarray(positions, dtype=float).copy()

        def set_torque_enabled(self, ids, enabled):
            self.calls.append(('torque', enabled))
            return self.enable_failed

        def set_operating_mode(self, *args):
            raise AssertionError('Mode reset would briefly release the loaded arm')

        def disconnect(self, *, disable_torque=True):
            self.calls.append(('disconnect', disable_torque))

    monkeypatch.setattr(arm_client, 'FeetechClient', Client)
    monkeypatch.setattr(arm_client.time, 'sleep', lambda seconds: None)
    return config, clients


def test_startup_stages_measured_goal_before_enabling_torque(setup_arm):
    config, clients = setup_arm
    arm = AmArmFeetech(config)
    arm.connect()
    calls = clients[0].calls
    assert calls[:2] == ['connect', 'read']
    assert calls[2][0] == 'write'
    np.testing.assert_allclose(calls[2][1], [-1, -2, -3])
    assert calls[3] == ('torque', True)
    arm.disconnect(disable_torque=False)
    assert calls[-1] == ('disconnect', False)
    assert ('torque', False) not in calls


def test_teach_then_restart_moves_slowly_to_the_same_initial_pose(setup_arm):
    config, clients = setup_arm
    arm = AmArmFeetech(config)
    arm.connect()
    original_zero = arm._motor_home.copy()
    clients[0].positions = np.array([-1.4, -1.2, -2.8])
    taught = arm.save_home_pose()
    # Teaching during a session must not shift the data coordinate frame.
    np.testing.assert_allclose(arm._motor_home, original_zero)
    assert json.loads((config.parent / 'initial_pose.json').read_text()) == taught
    arm.disconnect(disable_torque=False)

    restarted = AmArmFeetech(config)
    restarted.connect()
    writes = np.stack([c[1] for c in clients[1].calls if isinstance(c, tuple) and c[0] == 'write'])
    np.testing.assert_allclose(writes[-1], taught['motor_rad'], atol=1e-6)
    assert np.max(np.abs(np.diff(writes, axis=0))) <= np.deg2rad(15) / 30 + 1e-6
    np.testing.assert_allclose(restarted.read_model_rad(), 0, atol=1e-6)
    restarted.disconnect(disable_torque=False)


@pytest.mark.parametrize('failure', ['read', 'enable'])
def test_startup_failure_does_not_release_previously_held_motors(setup_arm, monkeypatch, failure):
    config, clients = setup_arm
    real_factory = arm_client.FeetechClient

    def factory(*args, **kwargs):
        client = real_factory(*args, **kwargs)
        if failure == 'read':
            client.last_read_ok = False
        else:
            client.enable_failed = [2]
        return client

    monkeypatch.setattr(arm_client, 'FeetechClient', factory)
    with pytest.raises(RuntimeError):
        AmArmFeetech(config).connect()
    assert clients[0].calls[-1] == ('disconnect', False)
    assert ('torque', False) not in clients[0].calls


def test_changed_motor_mapping_rejects_taught_pose_before_connect(setup_arm):
    config, clients = setup_arm
    arm = AmArmFeetech(config)
    arm.connect()
    arm.save_home_pose()
    arm.disconnect(disable_torque=False)
    settings = yaml.safe_load(config.read_text())
    settings['joint_signs']['shoulder_pan'] = 1
    config.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match='does not match'):
        AmArmFeetech(config).connect()
    assert len(clients) == 1


def test_return_reports_saved_pose_in_existing_dataset_coordinates(setup_arm):
    config, clients = setup_arm
    arm = AmArmFeetech(config)
    arm.connect()
    zero = arm._motor_home.copy()
    clients[0].positions = zero + np.array([-.4, .8, .2])
    arm.save_home_pose()
    clients[0].positions = zero + .1
    np.testing.assert_allclose(arm.return_home(), [.4, .8, .2], atol=1e-6)
    np.testing.assert_array_equal(arm._motor_home, zero)
    arm.disconnect(disable_torque=False)


def test_return_does_not_repeat_an_already_completed_trajectory(setup_arm):
    config, clients = setup_arm
    arm = AmArmFeetech(config)
    arm.connect()
    writes_before = len([c for c in clients[0].calls if isinstance(c, tuple) and c[0] == 'write'])
    arm.return_home()
    writes_after = len([c for c in clients[0].calls if isinstance(c, tuple) and c[0] == 'write'])
    assert writes_after == writes_before
    # Measured drift beyond the existing arrival tolerance still needs a return.
    clients[0].positions += .1
    arm.return_home()
    writes_final = len([c for c in clients[0].calls if isinstance(c, tuple) and c[0] == 'write'])
    assert writes_final > writes_after
    arm.disconnect(disable_torque=False)


@pytest.mark.parametrize('return_fails', [False, True])
def test_shutdown_returns_before_disconnect_and_preserves_torque(return_fails):
    calls = []

    def park():
        calls.append('return')
        if return_fails:
            raise RuntimeError('pose not reached')

    sink = Am200HardwareSink.__new__(Am200HardwareSink)
    sink._return_home_on_close = True
    sink._hold_on_disconnect = not return_fails
    sink._arm = SimpleNamespace(
        return_home=park, hold_current=lambda: calls.append('hold'),
        disconnect=lambda **kw: calls.append(('disconnect', kw['disable_torque'])),
    )
    sink._hand = None
    sink._cameras = SimpleNamespace(close=lambda: calls.append('cameras'))
    if return_fails:
        with pytest.raises(RuntimeError, match='팔 복귀'):
            sink.close()
        assert 'hold' in calls
    else:
        sink.close()
    assert calls[0] == 'return' and calls[-1] == ('disconnect', False)
    assert sink._arm is None


def test_connection_holds_arm_before_waiting_for_cameras(setup_arm, monkeypatch):
    config, clients = setup_arm
    from orca_teleop.am200 import hw_sink

    monkeypatch.setattr(hw_sink, '_check_bus_ports', lambda *args: None)
    sink = Am200HardwareSink(arm_config_path=config, enable_hand=False)
    events = []

    def open_cameras():
        assert clients and ('torque', True) in clients[0].calls
        events.append('cameras')

    sink._cameras = SimpleNamespace(open=open_cameras, ensure_live=lambda: None, close=lambda: None)
    sink.connect()
    assert events == ['cameras']
    sink.close(return_home=False)


def test_diagnostic_feedback_is_throttled_without_torque_or_goal_changes(setup_arm, monkeypatch, caplog):
    config, clients = setup_arm
    arm = AmArmFeetech(config)
    arm.connect()
    arm._feedback_diagnostics = True
    samples = []

    def status():
        samples.append(True)
        return {motor: dict(mode=0, torque_enabled=1, torque_limit_raw=700,
                            goal_rad=-1., position_rad=-1.5, voltage_v=10.8,
                            current_ma=2200, temperature_c=42, error_status=32)
                for motor in (1, 2, 3)}

    clients[0].read_sts_control_status = status
    before = [entry for entry in clients[0].calls if isinstance(entry, tuple)]
    with caplog.at_level('INFO', logger=arm_client.__name__):
        for now in (1., 1.1, 2.2):
            monkeypatch.setattr(arm_client.time, 'monotonic', lambda: now)
            arm.read_model_rad()
    after = [entry for entry in clients[0].calls if isinstance(entry, tuple)]
    assert len(before) == len(after)
    assert len(samples) == 2
    assert 'voltage=10.8V' in caplog.text and 'status=0x20' in caplog.text
    arm.disconnect(disable_torque=False)
