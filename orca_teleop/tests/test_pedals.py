"""Pedals share the GUI's phase-gated collection commands."""
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from orca_teleop import pedals


def test_three_pedals_repeat_hold_bounce_and_release():
    keys = pedals.PedalPressFilter()
    for code, name, command in ((38, 'a', 'record'), (56, 'b', 'save'), (54, 'c', 'discard')):
        assert keys.press(code, name, now=1) == command
        assert keys.press(code, name, now=2) is None  # held, not another command
        assert keys.press(code, name, repeat=True, now=3) is None
        keys.release(code)
        assert keys.press(code, name, repeat=True, now=3.1) is None
        assert keys.press(code, name, now=3.2) == command
        keys.release(code)
        assert keys.press(code, name, now=3.3) is None  # mechanical bounce
        keys.release(code)
        assert keys.press(code, name, now=4) == command
        keys.release(code)
    assert keys.press(99, 'unmapped', now=5) is None


@pytest.fixture
def routing():
    state = {'phase': 'ready', 'active': True, 'attempt': 1}
    commands = []
    collector = SimpleNamespace(control_dir=Path('/test/session'), status=lambda: dict(state),
                                command=commands.append, logs=deque())
    return pedals.PedalCommands(collector), collector, state, commands


@pytest.mark.parametrize('phase,expected', [
    ('ready', ['record']), ('recording', ['save', 'discard']), ('returning', []),
    ('starting', []), ('warming_up', []), ('finalizing', []), ('converting', []),
    ('parking', []), ('error', []), ('finished', []),
])
def test_only_commands_valid_for_the_current_phase_are_dispatched(routing, phase, expected):
    router, collector, state, commands = routing
    state['phase'] = phase
    for cmd in ('record', 'save', 'discard', 'unknown'):
        router.receive(cmd)
    router.dispatch_pending()
    assert commands == expected


@pytest.mark.parametrize('change', ['phase', 'session', 'attempt', 'inactive'])
def test_delayed_press_cannot_act_on_a_different_trial_or_connection(routing, change):
    router, collector, state, commands = routing
    router.receive('record')
    if change == 'phase': state['phase'] = 'recording'
    if change == 'session': collector.control_dir = Path('/test/another-session')
    if change == 'attempt': state['attempt'] = 2
    if change == 'inactive': state['active'] = False
    router.dispatch_pending()
    assert commands == []


def test_stale_gui_queue_and_press_during_return_are_not_replayed_later(routing, monkeypatch):
    router, collector, state, commands = routing
    clock = [1.]
    monkeypatch.setattr(pedals, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    router.receive('record')
    clock[0] = 3
    router.dispatch_pending()
    assert commands == []
    state['phase'] = 'returning'
    router.receive('record')
    state['phase'] = 'ready'
    router.dispatch_pending()
    assert commands == []


def test_pedal_calls_the_same_file_command_path_as_the_gui_buttons(tmp_path):
    import json
    from orca_teleop.collector_app import Collector
    collector = Collector(tmp_path)
    collector.process = SimpleNamespace(poll=lambda: None)
    collector.control_dir = tmp_path / 'control'
    collector.control_dir.mkdir()
    (collector.control_dir/'status.json').write_text(json.dumps({'phase':'ready','attempt':1}))
    router = pedals.PedalCommands(collector)
    router.receive('record')
    router.dispatch_pending()
    files = list((collector.control_dir/'commands').glob('*.json'))
    assert len(files) == 1
    assert json.loads(files[0].read_text())['command'] == 'record'


def test_command_rejection_is_logged_without_a_modal_dialog_or_gui_crash(routing):
    router, collector, state, commands = routing
    router.receive('record')
    def reject(command): raise ValueError('phase changed')
    collector.command = reject
    router.dispatch_pending()
    assert commands == []
    assert 'phase changed' in collector.logs[-1]
