"""Hand size must not grow when the operator curls or pinches fingers."""
from pathlib import Path

import numpy as np
import pytest

from orca_teleop.retargeting.adaptive_analytical import AdaptiveAnalyticalRetargeter
from orca_teleop.retargeting.constants import CALIBRATION_FRAMES

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / 'configs/adaptive_analytical_orca_v1_pinch_calibrated.yaml'


@pytest.fixture
def retargeter():
    return AdaptiveAnalyticalRetargeter(
        hand_config_path=str(ROOT / 'configs/orcahand-right-v1-feetech/config.yaml'),
        urdf_path=str(ROOT / 'orcahand_description/v1/models/urdf/orcahand_right.urdf'),
        config_path=str(PROFILE),
    )


def operator_hand(folded=False, scale=1.0):
    kp = np.zeros((21, 3))
    for f in range(5):
        base = 1 + 4*f
        kp[base] = [f * .018, .04, .005]
        for bone, length in enumerate((.03, .025, .02)):
            # Flex all three joints; lengths are unchanged even for a fist.
            angle = (bone + 1) * 1.0 if folded else 0
            direction = np.array([0, np.cos(angle), np.sin(angle)])
            kp[base + bone + 1] = kp[base + bone] + length * direction
    return kp * scale


def calibrate(r, points):
    r.reset()
    for _ in range(CALIBRATION_FRAMES):
        r._ingest_calibration_frame(points, 'mediapipe')
    assert r._calibration_done
    return r._mano_scale


def test_recalibrating_curled_fingers_preserves_hand_scale(retargeter):
    r = retargeter
    assert r._scale_calibration == 'bone_lengths'
    opened = calibrate(r, operator_hand())
    folded = calibrate(r, operator_hand(folded=True))
    assert folded == pytest.approx(opened, rel=1e-12)
    # Still compensate for actual operator size rather than fixing a constant.
    enlarged = calibrate(r, operator_hand(folded=True, scale=1.5))
    assert enlarged == pytest.approx(opened / 1.5, rel=1e-12)


@pytest.mark.parametrize('value', [0.0, float('nan')])
def test_invalid_scale_sample_does_not_advance_calibration(retargeter, value):
    kp = operator_hand()
    kp[9:13] = value
    with pytest.raises(ValueError, match='Degenerate'):
        retargeter._ingest_calibration_frame(kp, 'mediapipe')
    assert len(retargeter._calibration_mags) == 0


def test_legacy_profile_retains_original_scale_rule(retargeter):
    r = retargeter
    r._retarget_cfg.pop('scale_calibration')
    r._build_retarget_params()
    kp = operator_hand()
    # Original palm-to-tip magnitude calculation (without a hardware connection).
    palm = (kp[1] + kp[17]) * .5
    mags = np.linalg.norm(kp[[4, 8, 12, 16, 20]] - palm, axis=1)
    expected = np.median(r._urdf_keyvector_mags / mags)
    assert calibrate(r, kp) == pytest.approx(expected, rel=1e-6)


def test_unknown_scale_method_is_rejected(retargeter):
    retargeter._retarget_cfg['scale_calibration'] = 'unknown'
    with pytest.raises(ValueError, match='scale_calibration'):
        retargeter._build_retarget_params()
