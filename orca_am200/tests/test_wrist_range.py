"""The live hardware wrist must reach both physical ROM endpoints."""
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import yaml

from orca_teleop.am200 import Am200ActionComposer, CanonicalJointMap
from orca_teleop.am200.hw_sink import Am200HardwareSink
from orca_teleop.ingress.server import HandLandmarks
from orca_teleop.retargeting import adaptive_analytical as module
from orca_teleop.retargeting.retargeter import TargetPose

ROOT = Path(__file__).resolve().parents[2]
HAND = ROOT / 'configs/orcahand-right-v1-feetech/config.yaml'
PROFILE = ROOT / 'configs/adaptive_analytical_orca_v1_pinch_calibrated.yaml'
MODEL = ROOT / 'orca_am200/robot_model/am_arm200_orca_right.xml'
URDF = ROOT / 'orcahand_description/v1/models/urdf/orcahand_right.urdf'


@pytest.fixture(scope='module')
def retargeter():
    return module.AdaptiveAnalyticalRetargeter(
        hand_config_path=str(HAND), urdf_path=str(URDF), config_path=str(PROFILE),
    )


@pytest.mark.parametrize('input_deg,expected_deg', [
    (0, 0), (10, -20), (-10, 20), (32.5, -65), (-17.5, 35), (90, -65), (-90, 35),
])
def test_full_wrist_range_survives_retarget_composer_and_hardware_sink(
        retargeter, monkeypatch, input_deg, expected_deg):
    retargeter._calibration_done = True
    monkeypatch.setattr(module.retargeter_utils, 'get_normalized_local_manohand_joint_pos',
                        lambda *args: np.zeros((21, 3)))
    monkeypatch.setattr(retargeter, '_target_keypoints_in_urdf_frame', lambda points: points)
    monkeypatch.setattr(retargeter, '_solve', lambda *args, **kwargs: retargeter._neutral_qpos_phys.copy())
    monkeypatch.setattr(retargeter, '_log_pinch_diagnostic', lambda *args: None)
    hand = retargeter.retarget(TargetPose(np.zeros((21, 3)), wrist_angle_degrees=input_deg))
    assert hand.data['wrist'] == pytest.approx(expected_deg)

    composer = Am200ActionComposer.from_model_path(MODEL, hand_config_path=HAND)
    frame = HandLandmarks(keypoints=np.zeros((21, 3)), handedness='right', timestamp_ns=1,
                          wrist_transform=np.eye(4), clutch_supported=True, clutch_engaged=False)
    command = composer(frame, hand)
    sink = Am200HardwareSink(hand_config_path=HAND)
    sink._map = CanonicalJointMap.load(MODEL.parent / 'joint_map.json',
                                      mujoco.MjModel.from_xml_path(str(MODEL)), hand_config_path=HAND)
    sink._target = np.zeros(sink._map.dof)
    writes = []
    sink._hand = SimpleNamespace(set_joint_positions=lambda value, **kwargs: writes.append(value))
    sink.dispatch_action(command)
    assert writes[0]['wrist'] == pytest.approx(expected_deg)
    np.testing.assert_allclose(command.position[:3], 0)


def test_hardware_wrist_override_leaves_fingers_arm_and_simulation_limits_unchanged():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    simulated = CanonicalJointMap.load(MODEL.parent / 'joint_map.json', model)
    hardware = CanonicalJointMap.load(MODEL.parent / 'joint_map.json', model, hand_config_path=HAND)
    index = hardware.roles.index('hand.wrist')
    np.testing.assert_allclose(np.rad2deg([hardware.lower[index], hardware.upper[index]]), [-65, 35])
    keep = np.arange(hardware.dof) != index
    np.testing.assert_array_equal(hardware.lower[keep], simulated.lower[keep])
    np.testing.assert_array_equal(hardware.upper[keep], simulated.upper[keep])
    np.testing.assert_array_equal(hardware.ref, simulated.ref)
    joint = model.joint('right_wrist').id
    np.testing.assert_allclose([simulated.lower[index], simulated.upper[index]], model.jnt_range[joint])


@pytest.mark.parametrize('gain', [0, -1, float('nan'), float('inf')])
def test_invalid_wrist_input_gain_is_rejected(tmp_path, gain):
    cfg = yaml.safe_load(PROFILE.read_text())
    cfg['retarget']['wrist_input_scale'] = gain
    cfg['retarget']['surface_contact_calibration'] = str(ROOT / 'data/contact_calibration/full_hand_surface.json')
    path = tmp_path / 'profile.yaml'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='wrist_input_scale'):
        module.AdaptiveAnalyticalRetargeter(hand_config_path=str(HAND), urdf_path=str(URDF), config_path=str(path))
