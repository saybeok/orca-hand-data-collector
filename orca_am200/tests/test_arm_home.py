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


@pytest.mark.parametrize('bad_limits', [False, True])
def test_startup_reads_firmware_travel_before_enabling_or_moving(setup_arm, monkeypatch, bad_limits):
    config, clients = setup_arm
    factory = arm_client.FeetechClient
    def create(*args, **kwargs):
        client = factory(*args, **kwargs)
        def limits():
            client.calls.append('limits')
            if bad_limits:
                raise OSError('limits unavailable')
            return {1: (-1.5, -.5), 2: (-3., -1.), 3: (-4., -2.)}
        client.read_sts_position_limits = limits
        return client
    monkeypatch.setattr(arm_client, 'FeetechClient', create)
    arm = AmArmFeetech(config)
    if bad_limits:
        with pytest.raises(OSError):
            arm.connect()
        assert clients[0].calls == ['connect', 'read', 'limits', ('disconnect', False)]
    else:
        arm.connect()
        assert clients[0].calls[2] == 'limits'
        lower, upper = arm.model_limits
        np.testing.assert_allclose(lower, [-.5, -1, -1])
        np.testing.assert_allclose(upper, [.5, 1, 1])
        arm.disconnect(disable_torque=False)


def test_out_of_range_passive_pose_reports_joints_without_clipping_into_a_large_move(
        setup_arm, monkeypatch):
    config, clients = setup_arm
    factory = arm_client.FeetechClient
    scale = -2*np.pi/4096

    def create(*args, **kwargs):
        client = factory(*args, **kwargs)
        # Read-only snapshot from the failing startup: motors were torque-off.
        client.positions = np.array([2067, 224, 3137])*scale
        client.read_sts_position_limits = lambda: {
            mid: tuple(sorted(np.array(raw)*scale))
            for mid, raw in {1: [1335, 2693], 2: [691, 2677], 3: [1151, 3032]}.items()
        }
        return client

    monkeypatch.setattr(arm_client, 'FeetechClient', create)
    with pytest.raises(ValueError) as raised:
        AmArmFeetech(config).connect()
    message = str(raised.value)
    assert 'ID 2 (shoulder_lift)' in message and '41.0° 벗어남' in message
    assert 'ID 3 (elbow_flex)' in message and '9.2° 벗어남' in message
    assert 'ID 1 (' not in message
    assert '시작 자세' in message
    # Neither a goal write nor torque enable is allowed for a 41-degree jump.
    assert clients[0].calls == ['connect', 'read', ('disconnect', False)]
    assert not (config.parent/'initial_pose.json').exists()


def test_small_endpoint_feedback_error_can_start_without_changing_firmware_limits(
        setup_arm, monkeypatch):
    config, clients = setup_arm
    factory = arm_client.FeetechClient
    scale = -2*np.pi/4096
    limits = {mid: tuple(sorted(np.array(raw)*scale))
              for mid, raw in {1: [1335, 2693], 2: [691, 2677], 3: [1151, 3032]}.items()}

    def create(*args, **kwargs):
        client = factory(*args, **kwargs)
        client.positions = np.array([2108, 687, 2802])*scale
        client.read_sts_position_limits = lambda: limits
        return client

    monkeypatch.setattr(arm_client, 'FeetechClient', create)
    arm = AmArmFeetech(config)
    arm.connect()
    writes = [c[1] for c in clients[0].calls if isinstance(c, tuple) and c[0] == 'write']
    np.testing.assert_allclose(writes[0], np.array([2108, 691, 2802])*scale)
    lower, upper = arm.model_limits
    assert lower[1] < 0 and upper[1] == pytest.approx(0)
    np.testing.assert_allclose(arm._motor_limits, [limits[i] for i in [1, 2, 3]])
    arm.disconnect(disable_torque=False)


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


@pytest.mark.parametrize('old_sign,new_sign', [(-1, 1), (1, -1)])
def test_explicit_pan_fix_reverses_only_pan_and_keeps_taught_motor_home(setup_arm, old_sign, new_sign):
    import importlib.util
    from pathlib import Path
    script = Path(__file__).resolve().parents[2]/'scripts/migrate_arm_pan_home.py'
    spec = importlib.util.spec_from_file_location('pan_home_migration', script)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    config, clients = setup_arm
    settings = yaml.safe_load(config.read_text())
    settings['joint_signs']['shoulder_pan'] = old_sign
    config.write_text(yaml.safe_dump(settings))
    arm = AmArmFeetech(config)
    arm.connect()
    taught = arm.save_home_pose()
    q = np.array([.1, .2, -.3])
    arm.write_model_rad(q)
    old_target = clients[0].positions.copy()
    arm.disconnect(disable_torque=False)
    settings = yaml.safe_load(config.read_text())
    settings['joint_signs']['shoulder_pan'] = new_sign
    config.write_text(yaml.safe_dump(settings))
    path = config.parent/'initial_pose.json'
    before = path.read_bytes()
    assert migration.migrate(config)['status'] == 'proposed'
    assert path.read_bytes() == before
    result = migration.migrate(config, apply=True)
    assert Path(result['backup']).read_bytes() == before
    assert json.loads(path.read_text())['motor_rad'] == taught['motor_rad']
    assert json.loads(path.read_text())['joint_signs'] == [new_sign, 1, 1]
    restarted = AmArmFeetech(config)
    restarted.connect()
    np.testing.assert_allclose(clients[1].positions, taught['motor_rad'])
    restarted.write_model_rad(q)
    delta = clients[1].positions-np.array(taught['motor_rad'])
    old_delta = old_target-np.array(taught['motor_rad'])
    assert delta[0] == pytest.approx(-old_delta[0])
    np.testing.assert_allclose(delta[1:], old_delta[1:])
    np.testing.assert_allclose(restarted.read_model_rad(), q, atol=1e-6)
    assert migration.migrate(config, apply=True)['status'] == 'already_matches'
    restarted.disconnect(disable_torque=False)


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


@pytest.mark.parametrize('tolerance,error_deg,succeeds', [(5, 3.7, True), (5, 6, False), (3, 3.7, False)])
def test_loaded_home_error_uses_configured_tolerance_without_changing_goal(
        setup_arm, monkeypatch, tolerance, error_deg, succeeds):
    config, clients = setup_arm
    arm = AmArmFeetech(config)
    arm.connect()
    zero = arm._motor_home.copy()
    arm._cfg['home_tolerance_deg'] = tolerance
    write = clients[0].write_positions_sync

    def loaded_write(*args, **kwargs):
        write(*args, **kwargs)
        clients[0].positions[2] -= np.deg2rad(error_deg)

    clients[0].write_positions_sync = loaded_write
    ticks = iter(np.arange(0, 20, .2))
    monkeypatch.setattr(arm_client.time, 'monotonic', lambda: next(ticks))
    target = zero + .2
    if succeeds:
        arm._move_motor_pose(target)
    else:
        with pytest.raises(RuntimeError, match='elbow_flex='):
            arm._move_motor_pose(target)
    np.testing.assert_allclose(arm._last_motor_target, target)
    np.testing.assert_array_equal(arm._motor_home, zero)
    arm.disconnect(disable_torque=False)


@pytest.mark.parametrize('tolerance', [0, -1, float('nan')])
def test_invalid_home_tolerance_is_rejected_before_moving(setup_arm, tolerance):
    config, clients = setup_arm
    arm = AmArmFeetech(config)
    arm.connect()
    arm._cfg['home_tolerance_deg'] = tolerance
    before = len(clients[0].calls)
    with pytest.raises(ValueError, match='tolerance'):
        arm._move_motor_pose(arm._motor_home + .1)
    assert not any(isinstance(c, tuple) and c[0] == 'write' for c in clients[0].calls[before:])
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
