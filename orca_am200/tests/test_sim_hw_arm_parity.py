"""Replay identical clutch/wrist inputs through the actual two CLI setups.

Neither runner may connect a sink or start a publisher in these tests.
"""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from orca_core import OrcaJointPositions

from orca_teleop.am200.sim_env import DEFAULT_MODEL_PATH
from orca_teleop.am200.tasks import make_task, task_scene_path
from orca_teleop.ingress.server import HandLandmarks

ROOT = Path(__file__).resolve().parents[2]


def configured_composer(monkeypatch, runner, scale):
    path = ROOT / "orca_teleop/scripts" / f"teleop_am200_{runner}.py"
    spec = importlib.util.spec_from_file_location(f"arm_parity_{runner}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class OfflineSink:
        def __init__(self, model_path=None, task=None, **kwargs):
            self.model_path = Path(model_path or (task_scene_path(make_task(task)) if task else DEFAULT_MODEL_PATH))

        def connect(self):
            raise AssertionError("No hardware, renderer, or publisher connection allowed")

    captured = {}
    monkeypatch.setattr(module, "Am200SimSink" if runner == "sim" else "Am200HardwareSink", OfflineSink)
    monkeypatch.setattr(module, "run_metaquest_local", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(module.OperatorCalibration, "load", lambda *_: None)
    argv = [str(path), "--clutch", "--hand-config", str(ROOT / "configs/orcahand-right-v1-feetech/config.yaml")]
    if runner == "sim":
        argv += ["--task", "hwatu_mat"]
    if scale is not None:
        argv += ["--arm-scale", str(scale)]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    assert captured["clutch_enabled"] is True
    assert captured["handedness"] == "right"
    c = captured["action_composer"]
    assert c._config.scale == pytest.approx(0.35 if scale is None else scale)
    assert c._config.max_velocity_rad_s is None
    assert c._config.max_acceleration_rad_s2 is None
    return c


@pytest.mark.parametrize("scale", [None, 0.2, 0.7])
def test_sim_card_scene_matches_hardware_arm_targets_and_clutch(monkeypatch, scale):
    sim = configured_composer(monkeypatch, "sim", scale)
    hw = configured_composer(monkeypatch, "hw", scale)
    action = OrcaJointPositions({j: 0.0 for j in hw._map.hand_joint_ids})
    states = [(False, [0, 0, 0]), (True, [0, 0, 0])]
    states += [(True, [0, 0, 0.004 * n]) for n in range(1, 26)]
    states += [(False, [0.005 * n, 0, 0.1]) for n in range(20)]
    states += [(True, [0.095, 0, 0.1]), (True, [0.095, 0, 0.12])]
    commands = []
    for i, (engaged, position) in enumerate(states):
        wrist = np.eye(4)
        wrist[:3, 3] = position
        frame = HandLandmarks(np.zeros((21, 3)), "right", 1_000_000_000 + i * 33_333_333,
                              wrist_transform=wrist, clutch_supported=True, clutch_engaged=engaged)
        sim_cmd = sim(frame, action).position
        hw_cmd = hw(frame, action).position
        np.testing.assert_allclose(sim_cmd, hw_cmd, atol=1e-12)
        commands.append(sim_cmd[:3].copy())
    commands = np.array(commands)
    assert np.linalg.norm(commands[26] - commands[1]) > 0.01
    # Released frames 27..46 and the first re-engaged frame 47 hold the target.
    np.testing.assert_allclose(commands[27:48], np.broadcast_to(commands[26], (21, 3)), atol=1e-10)
