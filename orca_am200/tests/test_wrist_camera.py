"""The wrist camera rides on the flange mount and can be recorded next to the
task camera."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from orca_teleop.am200 import parse_sim_camera_spec
from orca_teleop.am200.sim_env import resolve_model_path


def _camera_in_flange_frame(model, data):
    mujoco.mj_forward(model, data)
    cam = model.camera("wrist_cam").id
    flange = model.body("orca_flange").id
    R = data.xmat[flange].reshape(3, 3)
    pos = R.T @ (data.cam_xpos[cam] - data.xpos[flange])
    rot = R.T @ data.cam_xmat[cam].reshape(3, 3)
    return pos, rot


def test_wrist_camera_moves_with_the_arm():
    model = mujoco.MjModel.from_xml_path(str(resolve_model_path(None)))
    data = mujoco.MjData(model)
    pos0, rot0 = _camera_in_flange_frame(model, data)
    world0 = data.cam_xpos[model.camera("wrist_cam").id].copy()
    for name, q in (("shoulder_pan", 0.4), ("shoulder_lift", -0.3), ("elbow_flex", 0.5)):
        data.qpos[model.joint(name).qposadr[0]] = q
    pos1, rot1 = _camera_in_flange_frame(model, data)
    assert np.linalg.norm(data.cam_xpos[model.camera("wrist_cam").id] - world0) > 0.05
    np.testing.assert_allclose(pos1, pos0, atol=1e-9)
    np.testing.assert_allclose(rot1, rot0, atol=1e-9)
    # palm side (flange -y), beside the tower base, looking along the hand (+z)
    assert pos0[1] < -0.06 and 0.04 < pos0[2] < 0.07
    view = -rot0[:, 2]
    assert view[2] > 0.95


def test_parse_sim_camera_spec():
    assert parse_sim_camera_spec("wrist:wrist_cam") == ("wrist", "wrist_cam")
    assert parse_sim_camera_spec("wrist_cam") == ("wrist_cam", "wrist_cam")
    with pytest.raises(ValueError):
        parse_sim_camera_spec("wrist:")


def test_sim_sink_records_the_wrist_camera():
    from orca_teleop.am200 import Am200SimSink, SimCameraConfig

    sink = Am200SimSink(
        render_mode="rgb_array",
        task="hwatu",
        randomize=False,
        camera_config=SimCameraConfig(name="front", width=160, height=120),
        extra_cameras={"wrist": "wrist_cam"},
    )
    sink.connect()
    try:
        assert sink.camera_shapes == {"front": (120, 160, 3), "wrist": (120, 160, 3)}
        images = sink.get_observation().images
        assert set(images) == {"front", "wrist"}
        assert images["wrist"].shape == (120, 160, 3)
        assert not np.array_equal(images["front"], images["wrist"])
        assert images["wrist"].std() > 5
    finally:
        sink.close()


def test_sim_sink_rejects_unknown_camera():
    from orca_teleop.am200 import Am200SimSink

    sink = Am200SimSink(render_mode="rgb_array", task="hwatu", extra_cameras={"x": "nope"})
    with pytest.raises(ValueError, match="nope"):
        sink.connect()
    sink.close()
