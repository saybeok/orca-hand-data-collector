"""Audience-show choreography stays timed, bounded and ends safely."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml
from orca_core import OrcaJointPositions

from orca_teleop.am200.canonical import CanonicalJointMap
from orca_teleop.am200.show_demo import SAFE_THUMB_POSE, build_show_demo


MODEL = Path(__file__).resolve().parents[1] / "robot_model" / "am_arm200_orca_right.xml"


def _contract():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    mapping = CanonicalJointMap.load(MODEL.parent / "joint_map.json", model)
    config_path = MODEL.parents[2] / "configs" / "orcahand-right-v1-feetech" / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    positions = dict(config["neutral_position"])
    positions["wrist"] = 0.0
    start = np.zeros(mapping.dof)
    start[mapping.hand_slice] = mapping.hand_from_physical(OrcaJointPositions(positions))
    return mapping, start


def test_show_has_requested_order_and_returns_to_start():
    mapping, start = _contract()
    frames = build_show_demo(mapping.roles, start, mapping.lower, mapping.upper, cycles=1)
    phases = list(dict.fromkeys(frame.phase for frame in frames))
    assert phases == [
        "start",
        "raise_and_open",
        "greeting_wave",
        "main_open_hand",
        "main_power_grasp",
        "main_pinch",
        "abduction_fan_out",
        "abduction_fan_in",
        "abduction_spread_grasp",
        "ripple_ready",
        "finger_ripple",
        "pair_scissors",
        "fan_flutter",
        "final_open",
        "final_open_hold",
        "safe_return",
    ]
    np.testing.assert_allclose(frames[-1].position, np.clip(start, mapping.lower, mapping.upper), atol=1e-12)


def test_show_targets_stay_in_joint_limits_with_small_steps():
    mapping, start = _contract()
    frames = build_show_demo(mapping.roles, start, mapping.lower, mapping.upper)
    q = np.stack([frame.position for frame in frames])
    assert np.isfinite(q).all()
    assert np.all(q >= mapping.lower - 1e-12)
    assert np.all(q <= mapping.upper + 1e-12)
    delta = np.abs(np.diff(q, axis=0))
    # Keep the heavy arm conservative; fingers may move faster for the flourish.
    assert np.max(delta[:, mapping.arm_slice]) < 0.04
    assert np.max(delta[:, mapping.hand_slice]) < 0.12


def test_greeting_waves_both_directions_and_official_grasp_closes_fingers():
    mapping, start = _contract()
    frames = build_show_demo(mapping.roles, start, mapping.lower, mapping.upper)
    index = {name: i for i, name in enumerate(mapping.roles)}
    greeting = np.stack([f.position for f in frames if f.phase == "greeting_wave"])
    pan = greeting[:, index["arm.shoulder_pan"]] - start[index["arm.shoulder_pan"]]
    assert pan.min() < -0.1 and pan.max() > 0.1

    open_hand = [f.position for f in frames if f.phase == "main_open_hand"][-1]
    grasp = [f.position for f in frames if f.phase == "main_power_grasp"][-1]
    for finger in ("index", "middle", "ring", "pinky"):
        assert grasp[index[f"hand.{finger}_mcp"]] > open_hand[index[f"hand.{finger}_mcp"]] + 0.5
        assert grasp[index[f"hand.{finger}_pip"]] > open_hand[index[f"hand.{finger}_pip"]] + 0.5


def test_power_grasp_uses_official_orca_rom_fraction():
    mapping, start = _contract()
    frames = build_show_demo(mapping.roles, start, mapping.lower, mapping.upper)
    grasp = [f.position for f in frames if f.phase == "main_power_grasp"][-1]
    i = mapping.roles.index("hand.index_mcp")
    assert grasp[i] == pytest.approx(mapping.lower[i] + 0.85 * (mapping.upper[i] - mapping.lower[i]))


def test_raised_greeting_palm_faces_the_audience():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    mapping, start = _contract()
    frames = build_show_demo(mapping.roles, start, mapping.lower, mapping.upper)
    raised = [f for f in frames if f.phase == "raise_and_open"][-1]
    data.qpos[mapping.qpos_index] = raised.position
    mujoco.mj_kinematics(model, data)
    palm = model.body("right_palm").id
    palm_normal = data.xmat[palm].reshape(3, 3)[:, 1]
    assert palm_normal[1] < -0.9  # audience/camera is in world -y
    assert abs(palm_normal[2]) < 0.35


def test_finger_ripple_moves_fingers_out_of_phase():
    mapping, start = _contract()
    frames = build_show_demo(mapping.roles, start, mapping.lower, mapping.upper)
    ripple = np.stack([f.position for f in frames if f.phase == "finger_ripple"])
    peaks = []
    for finger in ("index", "middle", "ring", "pinky"):
        i = mapping.roles.index(f"hand.{finger}_mcp")
        peaks.append(int(np.argmax(ripple[:, i])))
        assert np.ptp(ripple[:, i]) > 0.4
    assert len(set(peaks)) == 4


def test_thumb_stays_in_clearance_pose_during_flourishes():
    mapping, start = _contract()
    frames = build_show_demo(mapping.roles, start, mapping.lower, mapping.upper)
    q = np.stack([
        f.position
        for f in frames
        if f.phase not in ("start", "raise_and_open", "safe_return")
    ])
    for joint in ("mcp", "abd", "pip", "dip"):
        i = mapping.roles.index(f"hand.thumb_{joint}")
        np.testing.assert_allclose(q[:, i], SAFE_THUMB_POSE[f"hand.thumb_{joint}"])


def test_speed_scales_the_whole_show_duration():
    mapping, start = _contract()
    normal = build_show_demo(mapping.roles, start, mapping.lower, mapping.upper, cycles=1, speed=1.0)
    fast = build_show_demo(mapping.roles, start, mapping.lower, mapping.upper, cycles=1, speed=1.5)
    assert fast[-1].time_s < normal[-1].time_s
    assert fast[-1].time_s > normal[-1].time_s / 1.5  # safety rate limit remains active


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cycles": 0}, "cycles"),
        ({"speed": 0.4}, "speed"),
        ({"hz": 5}, "hz"),
        ({"motion_scale": 0.8}, "motion_scale"),
    ],
)
def test_invalid_show_settings_are_rejected(kwargs, message):
    mapping, start = _contract()
    with pytest.raises(ValueError, match=message):
        build_show_demo(mapping.roles, start, mapping.lower, mapping.upper, **kwargs)
