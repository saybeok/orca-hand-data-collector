"""Start/save returns must not compete with teleop or change dataset zero."""
import importlib.util
from pathlib import Path
import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from orca_core import OrcaJointPositions

from orca_teleop.am200 import Am200ActionComposer, ArmMappingConfig
from orca_teleop.am200.episode_return import ArmEpisodeReturn
from orca_teleop.am200.hw_sink import Am200HardwareSink
from orca_teleop.hand_modes import FoldThreeFingersComposer
from orca_teleop.ingress.server import HandLandmarks
from orca_teleop.recording import TeleopActionMirror

ROOT = Path(__file__).resolve().parents[2]
HAND = ROOT / 'configs/orcahand-right-v1-feetech/config.yaml'
MODEL = ROOT / 'orca_am200/robot_model/am_arm200_orca_right.xml'


def frame(n, engaged=True, x=0):
    wrist = np.eye(4)
    wrist[0, 3] = x
    # Deliberately unrelated to PC wall time, as a headset clock can be.
    return HandLandmarks(np.zeros((21, 3)), 'right', 1_000_000_000 + n*33_333_333,
                         wrist_transform=wrist, clutch_supported=True, clutch_engaged=engaged)


@pytest.fixture(params=[False, True])
def setup_return(request):
    composer = Am200ActionComposer.from_model_path(
        MODEL, hand_config_path=HAND,
        config=ArmMappingConfig(max_velocity_rad_s=np.deg2rad(15),
                                max_acceleration_rad_s2=np.deg2rad(45)))
    outer = FoldThreeFingersComposer(HAND, composer) if request.param else composer
    writes = []
    park = np.array([.15, -.2, .25])
    sink = SimpleNamespace(return_arm_home=lambda: park.copy(), dispatch_action=writes.append)
    enabled = threading.Event()
    returner = ArmEpisodeReturn(sink, outer, enabled)
    action = OrcaJointPositions(dict(yaml.safe_load(HAND.read_text())['neutral_position']))
    return returner, composer, enabled, action, park, writes


def test_return_syncs_ik_rejects_old_commands_and_waits_for_clutch_release(setup_return):
    r, composer, enabled, hand, park, writes = setup_return
    enabled.set()
    old = r.compose(frame(0), hand)
    r.return_home()
    assert not enabled.is_set()
    np.testing.assert_allclose(composer.arm_position, park)
    # Buffered release frames during readiness must not unlatch the clutch.
    r.compose(frame(1, engaged=False), hand)
    enabled.set()
    r.dispatch(old)
    assert not writes
    for n in range(2, 6):
        command = r.compose(frame(n, x=.2), hand)
        np.testing.assert_allclose(command.position[:3], park)
    r.compose(frame(6, engaged=False, x=.2), hand)
    command = r.compose(frame(7, x=.2), hand)
    np.testing.assert_allclose(command.position[:3], park, atol=1e-10)
    command = r.compose(frame(8, x=.205), hand)
    assert np.max(abs(command.position[:3]-park)) <= np.deg2rad(15)/30 + 1e-8
    r.dispatch(command)
    assert writes == [command]


def test_disabled_dispatch_does_not_advance_virtual_arm(setup_return):
    r, composer, enabled, hand, park, _ = setup_return
    r.return_home()
    for n in range(20):
        command = r.compose(frame(n, engaged=True, x=.01*n), hand)
        np.testing.assert_allclose(command.position[:3], park)


def test_return_waits_for_an_inflight_motor_write():
    entered, release, returned = (threading.Event() for _ in range(3))
    calls = []
    def write(action):
        calls.append('write_start')
        entered.set()
        assert release.wait(3)
        calls.append('write_end')
    def park():
        calls.append('return')
        returned.set()
        return np.zeros(3)
    enabled = threading.Event(); enabled.set()
    sink = SimpleNamespace(dispatch_action=write, return_arm_home=park)
    r = ArmEpisodeReturn(sink, SimpleNamespace(reset_arm_position=lambda q: None), enabled)
    writer = threading.Thread(target=r.dispatch, args=(object(),))
    parking = threading.Thread(target=r.return_home)
    try:
        writer.start()
        assert entered.wait(2)
        parking.start()
        assert not returned.wait(.05)
    finally:
        release.set()
        writer.join(3)
        if parking.ident is not None:
            parking.join(3)
    assert calls == ['write_start', 'write_end', 'return']


def test_hardware_button_return_only_changes_arm_target():
    calls = []
    position = np.array([.1, -.2, .3])
    sink = Am200HardwareSink.__new__(Am200HardwareSink)
    sink._arm = SimpleNamespace(return_home=lambda: calls.append('same_return_as_close') or position.copy())
    sink._map = SimpleNamespace(arm_slice=slice(0, 3))
    sink._target = np.arange(20, dtype=float)
    before = sink._target.copy()
    np.testing.assert_allclose(sink.return_arm_home(), position)
    np.testing.assert_array_equal(sink._target[3:], before[3:])
    assert calls == ['same_return_as_close']


@pytest.mark.parametrize('reason', ['record', 'save'])
def test_button_return_finishes_before_freshness_gate_and_never_captures_frames(monkeypatch, reason):
    spec = importlib.util.spec_from_file_location('return_recorder', ROOT/'orca_teleop/scripts/record_dataset.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    calls = []
    mirror = TeleopActionMirror(); mirror.update(OrcaJointPositions({'wrist': 10}))
    queues = SimpleNamespace(actions_q=queue.Queue())
    queues.actions_q.put(OrcaJointPositions({'wrist': 20}))
    control = SimpleNamespace(update=lambda **kw: calls.append(('phase', kw['phase'])))
    returner = SimpleNamespace(return_home=lambda: calls.append('return'), sink=object())
    def fresh(**kwargs):
        assert mirror.snapshot() is None and queues.actions_q.empty()
        calls.append('fresh_commands')
        return True
    monkeypatch.setattr(module, 'wait_for_teleop_mirror_ready', fresh)
    assert module._return_arm_for_collection(returner, control, mirror, queues, threading.Event(), reason)
    assert calls == ([('phase', 'returning'), 'return', 'fresh_commands'] if reason == 'record'
                     else [('phase', 'returning'), 'return'])


def test_failed_return_cannot_advance_to_recording(monkeypatch):
    spec = importlib.util.spec_from_file_location('failed_return_recorder', ROOT/'orca_teleop/scripts/record_dataset.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    def failure():
        raise RuntimeError('pose not reached')
    def must_not_start(**kwargs):
        raise AssertionError('Failed return must not start data capture')
    monkeypatch.setattr(module, 'wait_for_teleop_mirror_ready', must_not_start)
    with pytest.raises(RuntimeError, match='pose not reached'):
        module._return_arm_for_collection(
            SimpleNamespace(return_home=failure), SimpleNamespace(update=lambda **kw: None),
            TeleopActionMirror(), SimpleNamespace(actions_q=queue.Queue()), threading.Event(), 'record')
