"""Fixed finger mode must reach both the robot and the recorded action."""
import queue
import threading
from pathlib import Path

import numpy as np
import pytest
import yaml
from orca_core import OrcaJointPositions

from orca_teleop.am200 import Am200ActionComposer
from orca_teleop.hand_modes import FoldThreeFingersComposer
from orca_teleop.ingress.server import HandLandmarks
from orca_teleop.recording import TeleopActionMirror, action_to_feature, teleop_consumer_loop

ROOT = Path(__file__).resolve().parents[2]
HAND = ROOT / 'configs/orcahand-right-v1-feetech/config.yaml'
MODEL = ROOT / 'orca_am200/robot_model/am_arm200_orca_right.xml'


def hand_action():
    return OrcaJointPositions(dict(yaml.safe_load(HAND.read_text())['neutral_position']))


def test_folding_only_changes_the_three_fingers_without_mutating_the_input():
    mode = FoldThreeFingersComposer(HAND)
    original = hand_action()
    before = dict(original.data)
    result = mode(None, original)
    assert original.data == before
    assert len(mode.pose_deg) == 9
    for joint, value in result.data.items():
        assert value == pytest.approx(mode.pose_deg.get(joint, before[joint]))
    for finger in ('middle', 'ring', 'pinky'):
        assert result.data[f'{finger}_mcp'] == 75
        assert result.data[f'{finger}_pip'] == 90
        assert result.data[f'{finger}_abd'] == {'middle': 8, 'ring': 0, 'pinky': -15}[finger]
    mode.reset()
    assert mode(None, original).data == result.data


def test_folded_angles_respect_the_hand_rom(tmp_path):
    config = yaml.safe_load(HAND.read_text())
    config['joint_roms']['middle_mcp'] = [-20, 60]
    path = tmp_path / 'hand.yaml'
    path.write_text(yaml.safe_dump(config))
    assert FoldThreeFingersComposer(path).pose_deg['middle_mcp'] == 60


def test_folded_middle_leaves_clearance_for_index_flexion_in_the_robot_model():
    import mujoco
    from orca_teleop.am200.canonical import CanonicalJointMap

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    mapping = CanonicalJointMap.load(MODEL.parent / 'joint_map.json', model)
    mode = FoldThreeFingersComposer(HAND)
    collision = {}
    for finger in ('index', 'middle'):
        collision[finger] = [i for i in range(model.ngeom)
            if finger in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                          int(model.geom_bodyid[i])) or '')
            and (model.geom_contype[i] or model.geom_conaffinity[i])]
    gaps = []
    for mcp, pip in ((0, 0), (30, 30), (60, 60), (85, 90)):
        pose = hand_action()
        pose.data.update(index_mcp=mcp, index_pip=pip)
        folded = mode(None, pose)
        q = np.zeros(mapping.dof)
        q[mapping.hand_slice] = mapping.hand_from_physical(folded)
        data.qpos[mapping.qpos_index] = q
        mujoco.mj_forward(model, data)
        gaps.append(min(mujoco.mj_geomDistance(model, data, i, j, .2, None)
                        for i in collision['index'] for j in collision['middle']))
    # Sampled index flexion path with its normal ABD command; this is a model
    # clearance check, not a guarantee about the actual silicone skin shape.
    assert min(gaps) > .002


def test_full_command_keeps_arm_clutch_and_free_joints_and_is_recorded_after_folding():
    original = Am200ActionComposer.from_model_path(MODEL, hand_config_path=HAND)
    wrapped = Am200ActionComposer.from_model_path(MODEL, hand_config_path=HAND)
    mode = FoldThreeFingersComposer(HAND, wrapped)
    action = hand_action()
    action.data['thumb_dip'] = 40
    action.data['index_mcp'] = 30
    action.data['wrist'] = -60
    for i, (engaged, x) in enumerate(((False, 0), (True, 0), (True, .025), (False, .05))):
        wrist = np.eye(4)
        wrist[0, 3] = x
        frame = HandLandmarks(keypoints=np.zeros((21, 3)), handedness='right',
                              timestamp_ns=1_000_000_000 + i * 33_333_333,
                              wrist_transform=wrist, clutch_supported=True, clutch_engaged=engaged)
        expected = original(frame, action)
        command = mode(frame, action)
        np.testing.assert_array_equal(command.position[:3], expected.position[:3])
        actual = wrapped._map.hand_to_physical(command.position[wrapped._map.hand_slice])
        baseline = original._map.hand_to_physical(expected.position[original._map.hand_slice])
        for joint in actual:
            # MJCF range constants are rounded to five decimal radians.
            assert actual[joint] == pytest.approx(mode.pose_deg.get(joint, baseline[joint]), abs=1e-3)

    mirror = TeleopActionMirror()
    actions = queue.Queue()
    stop = threading.Event()
    sentinel = object()
    dispatched = []
    actions.put(command)
    actions.put(sentinel)
    teleop_consumer_loop(actions, mirror=mirror, stop_event=stop,
                        shutdown_sentinel=sentinel, dispatch_action=dispatched.append)
    assert len(dispatched) == 1
    recorded = mirror.snapshot()
    np.testing.assert_array_equal(action_to_feature(recorded, list(command.joint_ids)),
                                  action_to_feature(dispatched[0], list(command.joint_ids)))
    assert recorded is command
    mode.reset()
    assert wrapped._operator_origin is None
    assert not wrapped._engaged
    assert len(mode.pose_deg) == 9
