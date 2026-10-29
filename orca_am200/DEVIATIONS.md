# Deviations from the handoff

`handoff/START_HERE.md` rule 5 says `scene.xml` is the canonical sim embodiment and
must not be regenerated. This stage does replace its hand subtree. The reasons are
below, each reproducible from the files in this directory.

The arm is **not** regenerated: link origins, joint axes, inertials, the J4 STL
placement and the `orca_flange` frame are copied verbatim from
`handoff/scene_v4_original.xml`, which itself matches
`handoff/am-arm200_original.urdf`.

## 1. The handoff hand is v2; the physical hand is v1

`handoff/scene_v4_original.xml` builds its hand from the `orcahand_description`
**v2** export (meshes `right_R-Carpals.stl`, `right_T-PP.stl`, ...). The hardware
here is a **v1 Feetech** right hand: 16x HLS3915 plus an HLS3930 wrist at ID 17,
configured in `configs/orcahand-right-v1-feetech/config.yaml`.

v1 and v2 differ in link lengths, joint ranges, abduction axis sign and rest
offsets (`ref`), so a v2 sim would not predict what the v1 hardware does. Running
`scripts/derive_mount.py` reports the residual between the two designs after a
best-fit alignment: 8.5 mm rms and 19.5 mm max at the joint anchors.

## 2. The handoff joint names cannot be mapped by upstream `orca_teleop`

`OrcaHandSimSink._map_actuators` resolves each actuator by joint name: it strips
the `right_` prefix and passes the remainder to `_sim_joint_to_config_id`, which
accepts either a bare orca_core id (`thumb_mcp`) or the v2 abbreviation
(`t-mcp`). The handoff joints are raw CAD export names such as

    orcahand_right_R-Carpals_8d1f1041_to_TopTower-Model_4a80d30e

None of the 17 resolve, so the sim sink raises
`ValueError: orca_sim actuators have no orca_core joint_id counterpart`.
`handoff/joint_map_v4_original.json` does not help: it labels them
`hand.orca_right_00` .. `hand.orca_right_16` with no joint semantics.

The semantics were recoverable — see `V1_TO_V2_JOINTS` in
`scripts/derive_mount.py`, verified by anchor and axis agreement across all 17
joints — but the v1 model gives the correct kinematics *and* names that upstream
already understands, so it is used directly.
`tests/test_model_contract.py::test_hand_joint_names_are_mappable_by_the_upstream_sim_sink`
calls the real upstream function to keep this true.

## 3. The handoff hand inertias are off by roughly 10^6

The v2 subtree carries un-converted CAD inertia (mm^2 instead of m^2):

| quantity | handoff `scene.xml` | this model | real |
| --- | --- | --- | --- |
| forearm shell mass / inertia | 7.42 kg / 133 kg m^2 | n/a (v1 has no such link) | ~0.1 kg |
| hand subtree mass | 8.01 kg | 1.117 kg | ~1.1 kg |
| largest link inertia | 133 kg m^2 | 1.5e-3 kg m^2 | — |
| total robot mass | 9.04 kg | 2.14 kg | — |
| arm droop, 2 s at rest pose, ctrl=0 | 7.3 deg / 4.9 deg | 1.28 deg / 2.86 deg | — |

An 8 kg hand on a 1 kg arm driven by 8 N m actuators makes the arm unusable for
validating reach, IK or rate limits.

## 4. The handoff hand self-collides and the arm collides with nothing

Each v2 hand link carried two mesh geoms: one non-colliding, and one with no
class that inherited the default `contype=1 conaffinity=1`. That left 19
full-mesh colliding hand geoms with **zero** `<contact><exclude>` pairs, so the
rest pose already had the carpals interpenetrating the thumb by 5.1 mm.

Meanwhile every arm geom had `contype=0`, including `col_j4_proxy`, which is
named a collision proxy but collided with nothing.

This model uses the v1 hand's separated visual/collision meshes and its 18
exclude pairs, gives `col_j4_proxy` real collision parameters, and adds the two
excludes that the proxy needs against the hand base. The rest pose has no
contacts.

## 5. The handoff hand actuators could command past their joint stops

Every v2 hand actuator inherited `ctrlrange="-2 2"` from the `hand_joint`
default, while the joints themselves range from +/-0.26 to 1.87 rad. The v1 model
declares a per-joint `ctrlrange` that matches the joint limit;
`test_actuators_cannot_command_past_a_joint_stop` asserts it for all 20.

## 6. Canonical hand ordering follows the orca_core config, not the model

`handoff/START_HERE.md` defines indices 3..19 as "the ORCA right-hand joints in
the exact order used by `scene.xml`". With the hand replaced, this model instead
orders them by the `joint_ids` list in the orca_core hand config, which puts the
wrist last rather than first.

The reason is that the physical robot is the primary dataset target: the hardware
backend can then concatenate its measured and commanded hand vector with no
reordering, which is where an ordering mistake would be most costly. The
simulation backend needs a permutation instead, emitted once as
`model_actuator_index` in `robot_model/joint_map.json` rather than recomputed at
runtime.

## 7. Upstream `orca_teleop` had to change for the arm to be drivable

The handoff assumes the arm can be commanded from data the existing pipeline
already carries. It cannot: `HandFrame` had no field for the wrist pose. The
Quest bridge computes a full 4x4 FLU wrist matrix, and `publisher.py` reduced it
to a single flexion angle before transmitting, discarding the translation that
arm IK needs.

Four additive changes were made rather than working around it:

- `HandFrame.wrist_transform`, 16 floats, empty when the publisher has no wrist
  pose. Consumers must read empty as absent; identity would be a real pose.
  Stubs regenerated with `grpcio-tools==1.62.3`, matching the protobuf 4.25
  runtime the venv already had.
- `HandLandmarks.wrist_transform`, decoded by the ingress server, `None` when
  absent or malformed.
- an optional `action_composer` hook on `retargeter_worker` and `run`, so an
  embodiment with more joints than the hand can build its own command from the
  same frame. Without it the hand-only path is unchanged.
- `HandFrame.clutch_supported` / `clutch_engaged` plus an opt-in
  `clutch_enabled` on the Quest publisher, which derives the clutch from a pinch
  on whichever hand is *not* being teleoperated. Two proto fields rather than
  one because "no clutch in this session" must not read the same as "clutch
  released": collapsing them would make an untracked off hand grant permission
  to move.

All the arm-specific logic lives in `orca_teleop.am200`, so nothing above knows
about the AM arm.

## 8. The arm mapping is calibrated to a file, not to the first frame

The handoff says nothing about how the operator's frame relates to the robot's,
and the obvious implementation — anchor on the first tracked frame — is not
usable for imitation learning. The mapping then depends on where the operator's
hand happened to be when tracking started, so the same posture commands a
different robot pose on every run. The recorded labels stay self-consistent,
because `observation.state` and `action` are robot-absolute, but the operator's
reachable region shifts between episodes, which saturates the 0.12–0.40 m
workspace clamp on some and not others and inflates the variance of the
demonstrations.

`orca_core` already models the fix. Its calibration is an explicit user-invoked
step that writes `{config_dir}/calibration.yaml`, is loaded at construction, and
warns rather than fails when absent. `orca_teleop.am200.operator_cal` copies that
shape for the human side, storing the operator's neutral wrist position in
`configs/operator_calibration.yaml`, captured by
`orca_am200/scripts/calibrate_operator.py`.

The clutch is kept separate from it on purpose. An absolute mapping and a clutch
cannot simply be combined: re-engaging after repositioning would teleport the arm
to whatever the absolute mapping demands. So episode starts use the calibration
and are identical every time, while re-anchoring is confined to within an
episode, where the drift is bounded by the operator and invisible to a
robot-absolute recording.

## Values still needed before physical AM motion

`handoff/START_HERE.md` rule 7 already notes there is no AM driver in the bundle.
In addition, the arm's kinematic limits are unknown, not merely undocumented:
`handoff/am-arm200_original.urdf` ships `lower=-3.14 upper=3.14 effort=0
velocity=0` for every joint, so the `+/-3.14` ranges in this model are
placeholders carried over from that source.

`am_arm_upstream/` is a shallow clone of
[liyiteng/AM-ARM](https://github.com/liyiteng/AM-ARM). Its
`am-arm200/urdf/urdf/am-arm200.urdf` is byte-identical to the handoff copy, so
the real limits are not there either. That repository is CAD, BOM and assembly
only; its README points all software at
[liyiteng/lerobot_alohamini](https://github.com/liyiteng/lerobot_alohamini),
whose `docs/alohamini/am-arm200.md` covers port configuration and calibration.

That means the AM arm is driven through LeRobot's Feetech servo bus. Useful,
because the ORCA v1 hand is also Feetech (HLS3915/HLS3930 on its own bus), so the
two adapters can share a servo family even though they stay on separate ports.

Still needed before enabling physical AM motion:

- real joint limits for `shoulder_pan`, `shoulder_lift`, `elbow_flex`
- serial port, servo IDs, baud rate for the three STS3095 motors
- home offsets and per-joint sign conventions
- torque/velocity limits and the emergency-stop path

The BOM confirms the follower uses 4x STS3215 plus 3x STS3095, and this model
keeps exactly the three STS3095 joints.
