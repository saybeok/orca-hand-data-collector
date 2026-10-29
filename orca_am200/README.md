# AM-ARM200 + ORCA right hand v1

Combined 20-DoF embodiment for the `orca_teleop` teleoperation and LeRobot
recording stack: a 3-DoF AM-ARM200 arm carrying an ORCA right hand v1.

This directory holds the robot model and its contract. Teleoperation of all 20
joints works in simulation; the hardware backend and dataset recording described
in `handoff/IMPLEMENTATION_SPEC.md` do not — see [Status](#status).

## Layout

```text
robot_model/
  am_arm200_orca_right.xml   canonical combined MJCF
  orcahand_right_v1.xml      verbatim copy of orca_sim models/v1/right.mjcf
  joint_map.json             generated canonical 20-DoF order
  assets/am/                 AM arm meshes (from the handoff bundle)
  assets/mjcf/right/         ORCA v1 hand meshes (from orca_sim)
scripts/
  derive_mount.py            re-derives the flange -> hand mount transform
  build_joint_map.py         regenerates robot_model/joint_map.json
  view_model.py              offscreen renders, or --interactive viewer
  smoke_arm_teleop.py        end-to-end: gRPC wrist pose -> arm motion, in cm
  calibrate_operator.py      captures the operator's neutral wrist, once
tests/
  test_model_contract.py     ordering, limit, mass, collision and stability gates
  test_sim_sink.py           the 20-DoF sink: ordering, conversion, arm hold
  test_arm_teleop.py         wrist pose on the wire, arm IK, composed command
  test_operator_cal.py       clutch, persisted calibration, reproducibility
handoff/                     original bundle, unchanged, for reference
DEVIATIONS.md                what differs from the handoff, and why
```

## Canonical 20-DoF order

`robot_model/joint_map.json` is the source of truth. Indices 0-2 are the arm;
indices 3-19 are the hand in the order the orca_core hand config declares, so the
hardware vector needs no reordering. The simulation permutation is stored per
entry as `model_actuator_index`.

| index | role | MuJoCo joint |
| --- | --- | --- |
| 0-2 | `arm.shoulder_pan`, `arm.shoulder_lift`, `arm.elbow_flex` | same names |
| 3-6 | `hand.thumb_mcp/abd/pip/dip` | `right_thumb_*` |
| 7-9 | `hand.index_abd/mcp/pip` | `right_index_*` |
| 10-12 | `hand.middle_abd/mcp/pip` | `right_middle_*` |
| 13-15 | `hand.ring_abd/mcp/pip` | `right_ring_*` |
| 16-18 | `hand.pinky_abd/mcp/pip` | `right_pinky_*` |
| 19 | `hand.wrist` | `right_wrist` |

## Running

All commands need the repository environment, which clears the ROS `PYTHONPATH`
and `LD_LIBRARY_PATH` that otherwise shadow the virtualenv:

```bash
source env.sh
```

```bash
# all gates: model contract, sim sink, arm teleop
python -m pytest orca_am200/tests -q

# renders four poses into orca_am200/renders/
python orca_am200/scripts/view_model.py

# interactive viewer
python orca_am200/scripts/view_model.py --interactive --pose arm_reach_and_pinch

# end-to-end, no headset: a synthetic operator raises their wrist 25 cm and the
# script reports how far the flange actually followed
python orca_am200/scripts/smoke_arm_teleop.py

# regenerate derived files after editing the model
python orca_am200/scripts/derive_mount.py
python orca_am200/scripts/build_joint_map.py
```

### Teleoperation

Default arm control is relative: the first tracked wrist pose anchors to the
arm's home, then the flange follows your displacement. No operator calibration
file is required.

```bash
# Quest 3S. Needs a secure context; over USB that is:
#   adb reverse tcp:8765 tcp:8765
# then open http://localhost:8765 in Quest Browser and start hand tracking.
./scripts/teleop_am200.sh

# pinch the OTHER hand to let the arm follow, release to park it
./scripts/teleop_am200.sh --clutch

# webcam instead: the hand follows, the arm holds still (no wrist pose)
./scripts/teleop_am200.sh --source webcam --show-video

# hand only, even on the Quest
./scripts/teleop_am200.sh --no-arm

# larger or smaller arm motion per unit of wrist motion
./scripts/teleop_am200.sh --arm-scale 0.35
```

Optional absolute mapping (same operator posture → same robot pose across
sessions) is `python orca_am200/scripts/calibrate_operator.py`. Skip it for
ordinary following.

## How the arm is driven

The retargeter solves the 17 hand joints as before. The arm's 3 come from
position IK on the operator's wrist, and the two halves are combined into one
canonical 20-DoF command by `orca_teleop.am200.Am200ActionComposer`.

**Position only.** Three joints can place the flange but cannot orient it, so
your wrist rotation is not reproduced by the arm. Wrist *flexion* still reaches
the hand's own wrist joint through the existing `wrist_angle_degrees` path.

**Clamped.** Targets are held inside a spherical shell 0.12–0.40 m from the base,
away from the fully-extended and the near-base singularities, and each solve
moves any joint at most 0.06 rad. An out-of-reach request is reported once in
the log rather than silently saturating.

Measured end to end by `scripts/smoke_arm_teleop.py`: a 25 cm wrist lift at
scale 0.5 commands a 12.5 cm flange rise, of which 11.5 cm is realised — the
1 cm shortfall is the position servo sagging under the hand's weight, not the
mapping.

### Relating the operator to the robot

The Quest's origin is wherever the headset established its reference space, and
your shoulder is not the robot's base, so something has to relate the two. What
does the relating decides whether recorded demonstrations are reproducible, so
it is split deliberately in two.

**Episode starts are absolute**, from
`configs/operator_calibration.yaml`: your calibrated neutral posture maps to the
arm's home pose. Being on disk it is identical across episodes and across
sessions, so the same posture commands the same robot pose every time. This
follows `orca_core`'s own `calibration.yaml` pattern — an explicit user-invoked
capture, loaded thereafter, a missing file being a warning rather than an error.
Starting far from your calibrated neutral is legal and logs a warning; the arm
walks to the matching offset rather than jumping.

**Within an episode the clutch may re-anchor**, if `--clutch` is on. Pinching
your off hand lets the arm follow; releasing parks it while the *hand* keeps
tracking, so you can reposition and carry on the way lifting a mouse works. The
clutch is on the hand you are not teleoperating, so no controller is needed and
your working hand stays free. An untracked off hand reads as released, which is
the safe direction to fail in.

That split is the point. Reproducibility is a property of episode starts, where
it is enforced from a file. Comfort is a property of what happens inside an
episode, where re-anchoring drift costs nothing a robot-absolute recording would
notice, since the recorded `observation.state` and `action` are the robot's own.

Without a calibration file the mapping falls back to anchoring on the first
frame it sees. That still teleoperates, but the mapping differs run to run, so
`Am200ActionComposer` warns and
`tests/test_operator_cal.py::test_without_calibration_two_runs_disagree` pins
the difference — if that test ever passes, the file has stopped earning its
complexity.

For context on why this shape and not another: leader-follower rigs (ALOHA and
kin) need none of it, because a physical replica arm maps joint-to-joint and the
correspondence is absolute by construction. Headset-driven rigs use incremental
mapping like this one, and the ingredient that makes their data usable is
precisely a deliberate clutch plus a mapping that does not depend on when the
session happened to start.

## Model summary

| property | value |
| --- | --- |
| joints / actuators | 20 / 20 |
| total mass | 2.14 kg (arm 1.02, hand 1.12) |
| contact exclude pairs | 20 |
| contacts at rest pose | 0 |
| arm droop, 2 s holding rest pose | 1.28 deg, 2.86 deg |
| hand mount | tower face flush on the flange, palm down within 4 deg |

The hand is attached with MuJoCo's `<attach>`, so `orcahand_right_v1.xml` stays a
verbatim copy of the `orca_sim` v1 model. Joint names therefore keep the
`right_<orca_core id>` form that `orca_teleop`'s simulation sink already resolves;
`test_hand_joint_names_are_mappable_by_the_upstream_sim_sink` calls the real
upstream function to prove it.

## Status

Done:

- combined model with the AM arm copied verbatim and the correct v1 hand
- derived and regression-tested flange mount
- canonical 20-DoF ordering, generated rather than hand-written
- contract tests covering ordering, actuator limits, masses, collisions and
  gravity stability
- 20-DoF simulation backend: `orca_teleop.am200.Am200SimSink` plus
  `scripts/teleop_am200_sim.py`. It accepts either a full-robot
  `RobotJointCommand` or a hand-only `OrcaJointPositions`, so a publisher
  without a wrist pose still drives the hand.
- Quest wrist pose plumbing. `HandFrame.wrist_transform` now carries the full
  4x4 FLU matrix that `bridge.py` already computed and `publisher.py` used to
  discard, and the ingress decodes it onto `HandLandmarks`.
- 3-DoF arm IK with workspace, joint and rate limits
  (`orca_teleop.am200.ArmIk`), and the incremental wrist mapping
  (`Am200ActionComposer`). See [How the arm is driven](#how-the-arm-is-driven).
- `retargeter_worker`/`run` gained an optional `action_composer` hook, which is
  the only change the arm needed in the shared pipeline; the hand-only path
  behaves exactly as before when it is absent.
- persisted operator calibration and an off-hand pinch clutch, so the arm
  mapping is reproducible across sessions. See
  [Relating the operator to the robot](#relating-the-operator-to-the-robot).

While building the sink, the physical-to-model angle conversion turned out to
be inverted throughout upstream `orca_teleop`, which is fixed in this tree; see
[Angle conventions](#angle-conventions).

Not started, from `handoff/IMPLEMENTATION_SPEC.md`:

- the hardware backend over the canonical 20-DoF vector (the sim one is done).
- extending `scripts/record_dataset.py` to 20-D `observation.state` / `action`.
  That path also owns the per-episode `reset_event`, which the Quest publisher
  already consumes to re-zero the wrist estimator and the pinch latch;
  `Am200ActionComposer.reset()` needs wiring to it there so each episode starts
  from the calibrated mapping rather than inheriting the previous episode's
  clutch drift. In the plain teleop path `run_metaquest_local` passes
  `reset_event=None`, so nothing resets and the whole session is one episode.
- the AM hardware adapter, blocked on the values listed in `DEVIATIONS.md`.
  The arm's joint limits are still the URDF's `±3.14` placeholders, which is
  harmless in simulation but must be resolved before driving the real arm.

## Angle conventions

Two conventions describe the same joint and must not be mixed.

| | unit | zero |
| --- | --- | --- |
| physical | degrees | the hand's own zero; what `orca_core` and the retargeter use |
| model | radians | offset per joint by the MJCF `ref`; what MuJoCo and the URDF use |

MuJoCo applies `qpos - ref` as a hinge's rotation, so `qpos == ref` is the pose
the file describes, which is the physical zero. Hence `model = physical + ref`.

Upstream had this inverted in four places, and omitted it entirely in the sim
sink. It was invisible on the v2 hand, whose finger joints all carry `ref = 0`;
the v1 hand offsets seven of them by up to 42 degrees. The check that settles it
needs no hardware: the config's `joint_roms` shifted by `ref` reproduce the
URDF's own joint limits to within 0.01 degrees on 14 of 17 joints, whereas the
reversed sign maps `thumb_abd` to [22, 84] degrees against a URDF limit of
[-62, 0] — no overlap, so forward kinematics ran on an unreachable thumb.

The conversion now lives only in
`orca_teleop.retargeting.urdf_offsets.physical_to_model` / `model_to_physical`,
with `orca_teleop/tests/test_joint_conventions.py` pinning it against the shipped
models. The wrist is exempt: it is a passthrough whose zeros already coincide,
and unlike the fingers its ROM does not reconcile with the model range under
either sign.

## Provenance

- Arm geometry: `handoff/scene_v4_original.xml`, matching
  `handoff/am-arm200_original.urdf` from
  [liyiteng/AM-ARM](https://github.com/liyiteng/AM-ARM).
- Hand: `orca_sim` `models/v1/right.mjcf` and its meshes.
- Read `DEVIATIONS.md` before changing the model; it records why the handoff's
  hand subtree is not used.
