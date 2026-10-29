"""Check thumb diagnostics against the actual live WebXR point layout."""
import numpy as np
import pytest

from orca_teleop.ingress.metaquest.landmarks import retargeter_landmarks_from_webxr
from orca_teleop.ingress.visualizer import thumb_bend_degrees
from orca_teleop.retargeting.utils import get_normalized_local_manohand_joint_pos


@pytest.mark.parametrize("ip_degrees", [0, 45, 90])
def test_live_webxr_layout_keeps_thumb_ip_bend_through_normalization(ip_degrees):
    # Non-degenerate palm, straight proximal thumb, known distal bend.
    xr = np.zeros((25, 3))
    xr[1], xr[2], xr[3] = [.03, .02, 0], [.06, .04, 0], [.09, .06, 0]
    direction = np.array([3, 2, 0]) / np.sqrt(13)
    angle = np.radians(ip_degrees)
    xr[4] = xr[3] + .025 * (np.cos(angle)*direction + np.sin(angle)*np.array([0,0,1]))
    xr[6], xr[11], xr[16], xr[21] = [.03,.06,0], [0,.07,0], [-.02,.065,0], [-.04,.05,0]
    points = retargeter_landmarks_from_webxr(xr, 'right')
    mcp, ip = thumb_bend_degrees(points)
    assert mcp == pytest.approx(0, abs=1e-6)
    assert ip == pytest.approx(ip_degrees, abs=1e-6)
    normalized = get_normalized_local_manohand_joint_pos(points, 'mediapipe')
    np.testing.assert_allclose(thumb_bend_degrees(normalized), [mcp, ip], atol=2e-6)


def test_missing_thumb_segments_are_unavailable_not_zero_flexion():
    assert thumb_bend_degrees(np.zeros((21,3))) == (None, None)
