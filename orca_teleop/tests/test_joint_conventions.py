"""The physical <-> model angle convention, pinned against the shipped models.

Two conventions exist for the same joint. ``orca_core`` and the retargeter speak
*physical* degrees, measured from the hand's own zero. The URDF and the MJCF
speak *model* angles, offset per joint by the MJCF's ``ref``.

Getting the sign of that offset backwards is silent: forward kinematics still
evaluates, just on a pose the robot cannot reach. These tests fail loudly
instead, using only files that ship with the repositories.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import yaml

from orca_teleop.retargeting.urdf_offsets import (
    PASSTHROUGH_JOINTS,
    load_ref_offsets,
    model_to_physical,
    physical_to_model,
)

DESCRIPTION_DIR = Path(
    os.environ.get("ORCAHAND_DESCRIPTION_DIR", Path.home() / "orca_hand" / "orcahand_description")
)
URDF_PATH = DESCRIPTION_DIR / "v1" / "models" / "urdf" / "orcahand_right.urdf"
HAND_CONFIG_PATH = Path(
    os.environ.get(
        "ORCA_MODEL_PATH",
        Path.home() / "orca_hand" / "configs" / "orcahand-right-v1-feetech" / "config.yaml",
    )
)

# thumb_pip's upper stop and ring_abd's differ between the orca_core config and
# the model by 4.01 and 0.57 degrees. That predates this test; the tolerance is
# wide enough to admit it and far below the smallest ref offset (9.74 degrees).
LIMIT_TOLERANCE_DEG = 4.5


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"{path} is not available in this checkout")
    return path


@pytest.fixture(scope="module")
def hand_config() -> dict:
    return yaml.safe_load(_require(HAND_CONFIG_PATH).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ref_offsets_deg() -> dict[str, float]:
    offsets = load_ref_offsets(str(_require(URDF_PATH)), "right")
    if not offsets:
        pytest.skip("no companion MJCF found next to the v1 URDF")
    return {joint: np.rad2deg(value) for joint, value in offsets.items()}


@pytest.fixture(scope="module")
def urdf_limits_deg() -> dict[str, tuple[float, float]]:
    limits = {}
    for joint in ET.parse(_require(URDF_PATH)).getroot().iter("joint"):
        if joint.get("type") != "revolute":
            continue
        limit = joint.find("limit")
        limits[joint.get("name")] = (
            np.rad2deg(float(limit.get("lower"))),
            np.rad2deg(float(limit.get("upper"))),
        )
    return limits


def test_conversions_round_trip():
    physical = np.array([-20.0, 0.0, 42.0])
    ref = np.array([-42.0, 9.74, 29.98])
    np.testing.assert_allclose(
        model_to_physical(physical_to_model(physical, ref), ref), physical, atol=1e-12
    )


def test_offset_joints_exist_so_the_sign_is_actually_exercised(ref_offsets_deg):
    offset = [j for j, v in ref_offsets_deg.items() if abs(v) > 1e-9]
    assert len(offset) >= 6, (
        "the v1 hand should offset the thumb and abduction joints; "
        f"only {offset} carry a ref"
    )


def test_physical_rom_maps_onto_the_urdf_joint_limits(
    hand_config, ref_offsets_deg, urdf_limits_deg
):
    """The defining property: the physical ROM *is* the URDF limit, shifted by ref.

    This is what makes the sign checkable from files alone. With the sign
    reversed, thumb_abd maps to [22, 84] against a URDF limit of [-62, 0].
    """
    checked = 0
    for joint_id, (rom_low, rom_high) in hand_config["joint_roms"].items():
        name = f"right_{joint_id}"
        if name not in urdf_limits_deg or joint_id in PASSTHROUGH_JOINTS:
            continue
        ref = ref_offsets_deg.get(joint_id, 0.0)
        expected_low, expected_high = urdf_limits_deg[name]

        assert physical_to_model(rom_low, ref) == pytest.approx(
            expected_low, abs=LIMIT_TOLERANCE_DEG
        ), f"{joint_id} lower stop"
        assert physical_to_model(rom_high, ref) == pytest.approx(
            expected_high, abs=LIMIT_TOLERANCE_DEG
        ), f"{joint_id} upper stop"
        checked += 1

    assert checked >= 16, f"only {checked} joints were checked"


def test_reversing_the_sign_would_leave_the_thumb_unreachable(
    hand_config, ref_offsets_deg, urdf_limits_deg
):
    """Guards the specific regression: a reversed sign must not look plausible."""
    rom_low, rom_high = hand_config["joint_roms"]["thumb_abd"]
    ref = ref_offsets_deg["thumb_abd"]
    limit_low, limit_high = urdf_limits_deg["right_thumb_abd"]

    reversed_low, reversed_high = rom_low - ref, rom_high - ref
    assert reversed_low > limit_high or reversed_high < limit_low, (
        "thumb_abd no longer distinguishes the two signs, so this test has lost its teeth"
    )


def test_the_wrist_is_a_passthrough(ref_offsets_deg):
    """The wrist angle is fed through from the operator, not solved for."""
    for joint_id in PASSTHROUGH_JOINTS:
        assert ref_offsets_deg.get(joint_id) == pytest.approx(0.0)


def test_neutral_pose_is_reachable_in_model_space(
    hand_config, ref_offsets_deg, urdf_limits_deg
):
    """The configured rest pose must be inside the model's own joint limits."""
    for joint_id, physical in hand_config["neutral_position"].items():
        name = f"right_{joint_id}"
        if name not in urdf_limits_deg:
            continue
        model = physical_to_model(float(physical), ref_offsets_deg.get(joint_id, 0.0))
        low, high = urdf_limits_deg[name]
        assert low - LIMIT_TOLERANCE_DEG <= model <= high + LIMIT_TOLERANCE_DEG, (
            f"{joint_id} neutral maps to {model:.2f} deg, outside [{low:.2f}, {high:.2f}]"
        )
