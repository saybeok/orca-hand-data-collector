"""Deterministic audience-demo choreography for the AM-ARM200 + ORCA hand.

The choreography is generated in the canonical 20-DoF convention.  Arm values
are offsets from the connect-time pose, so the operator must place the arm in a
safe, repeatable neutral pose before connecting the hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


# From ORCA Core's GitHub examples/data demo_poses.yaml. Wrist is intentionally
# ignored below so these hand poses retain the audience-facing arm orientation.
OFFICIAL_ORCA_DEMOS = {
    "main": {
        "sequence": ("open_hand", "power_grasp", "pinch"),
        "poses": {
            "open_hand": {
                "thumb_cmc": 0.7, "thumb_abd": 0.8, "thumb_mcp": 0.85, "thumb_dip": 0.75,
                "index_abd": 0.1, "middle_abd": 0.5, "ring_abd": 0.7, "pinky_abd": 0.85,
                "index_mcp": 0.15, "middle_mcp": 0.15, "ring_mcp": 0.15, "pinky_mcp": 0.15,
                "index_pip": 0.1, "middle_pip": 0.1, "ring_pip": 0.1, "pinky_pip": 0.1,
            },
            "power_grasp": {
                "thumb_cmc": 0.35, "thumb_abd": 0.55, "thumb_mcp": 0.2, "thumb_dip": 0.85,
                "index_mcp": 0.85, "middle_mcp": 0.85, "ring_mcp": 0.85, "pinky_mcp": 0.85,
                "index_pip": 0.9, "middle_pip": 0.9, "ring_pip": 0.9, "pinky_pip": 0.9,
            },
            "pinch": {
                "thumb_cmc": 0.45, "thumb_abd": 0.65, "thumb_mcp": 0.4, "thumb_dip": 0.75,
                "index_abd": 0.3, "index_mcp": 0.7, "index_pip": 0.75,
                "middle_mcp": 0.3, "middle_pip": 0.2, "ring_mcp": 0.2, "ring_pip": 0.15,
                "pinky_mcp": 0.2, "pinky_pip": 0.15,
            },
        },
    },
    "abduction": {
        "sequence": ("fan_out", "fan_in", "spread_grasp"),
        "poses": {
            "fan_out": {
                "thumb_abd": 0.85, "index_abd": 0.1, "middle_abd": 0.5,
                "ring_abd": 0.8, "pinky_abd": 0.9,
            },
            "fan_in": {
                "thumb_abd": 0.25, "index_abd": 0.85, "middle_abd": 0.5,
                "ring_abd": 0.2, "pinky_abd": 0.15,
            },
            "spread_grasp": {
                "thumb_cmc": 0.55, "thumb_abd": 0.7, "thumb_mcp": 0.45, "thumb_dip": 0.65,
                "index_abd": 0.15, "middle_abd": 0.5, "ring_abd": 0.8, "pinky_abd": 0.9,
                "index_mcp": 0.65, "middle_mcp": 0.65, "ring_mcp": 0.65, "pinky_mcp": 0.65,
                "index_pip": 0.7, "middle_pip": 0.7, "ring_pip": 0.7, "pinky_pip": 0.7,
            },
        },
    },
}

# Model-radian pose that keeps the v1 thumb outside every flourish trajectory.
SAFE_THUMB_POSE = {
    "hand.thumb_mcp": -0.20,
    "hand.thumb_abd": -0.20,
    "hand.thumb_pip": 0.0,
    "hand.thumb_dip": 0.0,
}


@dataclass(frozen=True)
class DemoFrame:
    """One timestamped full-robot target."""

    time_s: float
    phase: str
    position: np.ndarray


def _minimum_jerk(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)


class _Timeline:
    def __init__(
        self,
        joint_ids: Sequence[str],
        start: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        hz: float,
        motion_scale: float,
    ) -> None:
        self.index = {name: i for i, name in enumerate(joint_ids)}
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        self.start = np.clip(np.asarray(start, dtype=np.float64), self.lower, self.upper)
        self.hz = float(hz)
        self.motion_scale = float(motion_scale)
        self.frames = [DemoFrame(0.0, "start", self.start.copy())]
        self.current = self.start.copy()

    def pose(self, offsets: dict[str, float]) -> np.ndarray:
        target = self.start.copy()
        for role, value in offsets.items():
            try:
                target[self.index[role]] += self.motion_scale * value
            except KeyError as exc:
                raise ValueError(f"show demo requires canonical joint {role!r}") from exc
        return np.clip(target, self.lower, self.upper)

    def fraction_pose(self, center: np.ndarray, fractions: dict[str, float]) -> np.ndarray:
        """Resolve an official ORCA ROM-fraction pose while preserving arm/wrist."""
        target = center.copy()
        intensity = self.motion_scale / 0.7
        for joint, fraction in fractions.items():
            if joint == "wrist" or joint.startswith("thumb_"):
                continue  # keep palm audience-front and thumb safely outside
            role = f"hand.{joint}"
            if role not in self.index:
                continue
            i = self.index[role]
            official = self.lower[i] + fraction * (self.upper[i] - self.lower[i])
            target[i] += intensity * (official - center[i])
        return np.clip(target, self.lower, self.upper)

    def _append(self, phase: str, duration_s: float, sample: Callable[[float], np.ndarray]) -> None:
        steps = max(1, round(duration_s * self.hz))
        t0 = self.frames[-1].time_s
        for step in range(1, steps + 1):
            u = step / steps
            self.frames.append(DemoFrame(t0 + u * duration_s, phase, sample(u).copy()))
        self.current = self.frames[-1].position.copy()

    def transition(
        self,
        phase: str,
        target: np.ndarray,
        duration_s: float,
        *,
        max_speed: float = 1.2,
    ) -> None:
        source = self.current.copy()
        delta = target - source
        # Minimum jerk peaks at 1.875 * distance / duration.
        duration_s = max(duration_s, 1.875 * float(np.max(np.abs(delta))) / max_speed)
        self._append(phase, duration_s, lambda u: source + _minimum_jerk(u) * delta)

    def motion(
        self,
        phase: str,
        center: np.ndarray,
        duration_s: float,
        offset_at: Callable[[float], dict[str, float]],
    ) -> None:
        def sample(u: float) -> np.ndarray:
            target = center.copy()
            for role, value in offset_at(u).items():
                target[self.index[role]] += self.motion_scale * value
            return np.clip(target, self.lower, self.upper)

        self._append(phase, duration_s, sample)

    def hold(self, phase: str, duration_s: float) -> None:
        pose = self.current.copy()
        self._append(phase, duration_s, lambda _u: pose)


def build_show_demo(
    joint_ids: Sequence[str],
    start: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    cycles: int = 1,
    speed: float = 1.0,
    hz: float = 30.0,
    motion_scale: float = 0.7,
    return_to_start: bool = True,
) -> list[DemoFrame]:
    """Build an audience-facing greeting followed by dexterous hand flourishes.

    ``start`` is both the arm's connect-time zero and the hand's initialized
    neutral pose.  At the default 0.7 scale the raised pose is approximately
    -0.25 rad shoulder, +1.0 rad elbow and +0.55 rad wrist: the palm faces the
    audience instead of the floor. The flourish poses come from ORCA Core's
    packaged ``main`` and ``abduction`` GitHub demos.
    """

    if cycles < 1:
        raise ValueError("cycles must be at least 1")
    if not 0.5 <= speed <= 1.5:
        raise ValueError("speed must be in [0.5, 1.5]")
    if hz < 10.0:
        raise ValueError("hz must be at least 10")
    if not 0.1 <= motion_scale <= 0.7:
        raise ValueError("motion_scale must be in [0.1, 0.7]")

    tl = _Timeline(joint_ids, start, lower, upper, hz, motion_scale)
    official = OFFICIAL_ORCA_DEMOS

    raised_offsets = {
        "arm.shoulder_lift": -0.357,
        "arm.elbow_flex": 1.429,
        "hand.wrist": 0.786,
    }
    raised = tl.pose(raised_offsets)
    for role, value in SAFE_THUMB_POSE.items():
        i = tl.index[role]
        raised[i] = np.clip(value, tl.lower[i], tl.upper[i])
    open_hand = tl.fraction_pose(raised, official["main"]["poses"]["open_hand"])
    tl.transition("raise_and_open", open_hand, 2.5 / speed)

    # A clearly visible whole-arm greeting, with a smaller wrist wave layered on.
    greeting_s = 3.5 / speed
    tl.motion(
        "greeting_wave",
        open_hand,
        greeting_s,
        lambda u: {
            "arm.shoulder_pan": 0.22 * np.sin(2.0 * np.pi * 3.0 * u),
            "hand.wrist": 0.04 * np.sin(2.0 * np.pi * 3.0 * u),
        },
    )

    # Reuse the official GitHub demo data, but keep the arm raised and palm-front.
    pose_duration = 0.35 / speed
    for _ in range(cycles):
        for demo_name in ("main", "abduction"):
            demo = official[demo_name]
            for pose_name in demo["sequence"]:
                target = tl.fraction_pose(open_hand, demo["poses"][pose_name])
                tl.transition(
                    f"{demo_name}_{pose_name}",
                    target,
                    pose_duration,
                    max_speed=3.0,
                )

    # A continuous travelling finger wave adds motion without target jumps.
    flourish_speed = min(speed, 1.2)
    ripple_s = 4.5 / flourish_speed

    def ripple(u: float) -> dict[str, float]:
        envelope = np.sin(np.pi * u) ** 2
        offsets = {
            "arm.shoulder_pan": 0.08 * envelope * np.sin(6.0 * np.pi * u),
        }
        fingers = ("index", "middle", "ring", "pinky")
        for k, finger in enumerate(fingers):
            curl = 0.58 * envelope * (1.0 + np.sin(6.0 * np.pi * u - k * np.pi / 2.0))
            offsets[f"hand.{finger}_mcp"] = curl
            offsets[f"hand.{finger}_pip"] = 0.85 * curl
        return offsets

    tl.transition("ripple_ready", open_hand, 0.3 / speed, max_speed=3.0)
    tl.motion("finger_ripple", open_hand, ripple_s, ripple)

    # Alternating pairs make the silhouette change much more than a single
    # sequential curl, while the smooth envelope keeps phase boundaries still.
    pair_s = 3.2 / flourish_speed

    def pair_scissors(u: float) -> dict[str, float]:
        envelope = np.sin(np.pi * u) ** 2
        wave = np.sin(5.0 * np.pi * u)
        offsets = {"arm.shoulder_pan": 0.06 * envelope * wave}
        for k, finger in enumerate(("index", "middle", "ring", "pinky")):
            pulse = 0.5 + 0.5 * wave * (1.0 if k % 2 == 0 else -1.0)
            offsets[f"hand.{finger}_mcp"] = 0.95 * envelope * pulse
            offsets[f"hand.{finger}_pip"] = 0.85 * envelope * pulse
        return offsets

    tl.motion("pair_scissors", open_hand, pair_s, pair_scissors)

    fan_s = 3.0 / flourish_speed

    def fan_flutter(u: float) -> dict[str, float]:
        envelope = np.sin(np.pi * u) ** 2
        wave = np.sin(6.0 * np.pi * u)
        return {
            "hand.index_abd": -0.32 * envelope * wave,
            "hand.middle_abd": 0.22 * envelope * wave,
            "hand.ring_abd": 0.28 * envelope * wave,
            "hand.pinky_abd": 0.32 * envelope * wave,
        }

    tl.motion("fan_flutter", open_hand, fan_s, fan_flutter)
    tl.transition("final_open", open_hand, 0.35 / speed, max_speed=3.0)
    tl.hold("final_open_hold", 0.5 / speed)

    # Hardware cleanup is deliberately part of the trajectory: disabling arm
    # torque while it is still raised would let it fall.
    if return_to_start:
        tl.transition("safe_return", tl.start, 3.0 / speed)

    return tl.frames
