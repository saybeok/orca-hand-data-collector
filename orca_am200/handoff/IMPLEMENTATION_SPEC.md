# IMPLEMENTATION SPEC — `orca_teleop` + AM-ARM200 + ORCA right hand

## 1. Architectural target

```text
Meta Quest 3S / WebXR
        |
        v
existing orca_teleop Meta Quest ingress
        |
        +-----------------------------+
        |                             |
        v                             v
existing ORCA hand retargeting    AM arm retargeting
(right hand, preserve)            Quest wrist pose -> calibrated Cartesian target -> 3DoF IK
        |                             |
        +-------------+---------------+
                      v
             canonical RobotAction
                 20 joint targets
                      |
             +--------+--------+
             |                 |
             v                 v
       composite sim      composite hardware
       MuJoCo scene.xml   AM arm + official ORCA API
             |                 |
             +--------+--------+
                      v
            existing LeRobot recorder
```

## 2. Reuse from upstream

Keep and extend the official `orcahand/orca_teleop` pipeline rather than replacing it. The upstream README documents:

- Quest Browser/WebXR ingress hosted by the recorder.
- Meta Quest source selection with `--source metaquest`.
- right-hand selection with `--hand right`.
- simulation validation using `--backend sim`.
- physical hand operation using `--backend hardware`.
- LeRobot episodes where measured hardware joints are `observation.state`, commanded joints are `action`, and cameras are `observation.images.<name>`.

The implementation should preserve these user-facing concepts and extend them to the combined 20-DoF robot.

## 3. Robot representation

Canonical state/action size: **20**.

- indices 0..2: AM-ARM200 arm
- indices 3..19: ORCA right hand

Never silently reorder joints. Use `joint_map.json` as the source of truth and add an assertion at startup that backend joint order equals the canonical order.

Recommended internal type (exact class name is flexible):

```python
@dataclass(frozen=True)
class RobotJointCommand:
    joint_ids: tuple[str, ...]
    position: np.ndarray  # shape (20,)
    timestamp_ns: int
```

Likewise robot state should carry the same `joint_ids` and measured position vector. Avoid an interface that assumes every robot action is only `OrcaJointPositions`.

## 4. Quest -> ORCA path

Do not rewrite the existing right-hand retargeter. Preserve upstream Quest hand landmarks/convention handling and the existing ORCA retargeting implementation.

The upstream README says the Quest path receives 25 WebXR hand joints and maps them to the hand layout used by the ORCA demonstrations. Keep this convention inside the ORCA path.

## 5. Quest wrist -> AM arm path

Use the full Quest wrist pose/transform available in the Meta Quest ingress path if exposed by the current upstream code. The arm controller should use a calibration transform rather than raw world coordinates.

At calibration capture:

```text
Quest wrist reference pose
AM arm reference q
AM end-effector reference position
```

At runtime:

```text
relative Quest wrist translation
 -> axis/frame transform
 -> workspace scale/gain
 -> desired AM end-effector XYZ
 -> workspace clamp
 -> 3DoF IK
 -> shoulder_pan / shoulder_lift / elbow_flex target
 -> joint-limit clamp
 -> rate/velocity limiting
```

Do not assign one Quest Cartesian axis directly to one AM joint.

For the first production-quality implementation, prefer a deterministic analytical/numerical 3-DoF IK built from the exact transforms in `robot_model/scene.xml`. If using MuJoCo/Pinocchio for IK, keep the arm IK independent of the viewer and simulator stepping loop so the same retargeter can be used for real hardware.

## 6. Simulation backend

Use `robot_model/scene.xml` exactly as the combined embodiment. Copy/symlink it into the appropriate `orca_teleop` model location instead of reconstructing it.

Simulation responsibilities:

- validate Quest coordinate mapping;
- validate right-hand retargeting;
- validate AM 3DoF IK and limits;
- validate the exact 20-element action order;
- validate safety/rate limits before enabling physical actuation;
- optionally record test episodes to verify dataset schema.

Do not spend time on high-fidelity sim-to-real dynamics unless later required for a separate learning objective.

## 7. Hardware backend

Composite hardware backend has two children:

```text
AMArmHardwareBackend      OrcaRightHandHardwareBackend
3 targets                 17 targets
        \                 /
         CompositeHardwareBackend
```

The ORCA half should use the existing official hardware integration in upstream `orca_teleop`/ORCA packages.

The AM half must be an adapter around the user's actual STS3095 control implementation. This ZIP does **not** contain verified AM serial port, servo IDs, baud rate, homing offsets, sign conventions, torque limits, or SDK API. Make those explicit configuration inputs. Do not guess them.

Before physical enable, require:

- successful calibration/home;
- joint-limit check;
- command rate limiter;
- stale Quest-frame timeout that stops/holds commands;
- explicit enable/deadman behavior;
- emergency-stop path supplied by the hardware stack.

## 8. Dataset contract

Extend the existing LeRobot recorder, do not create a second dataset format.

Required learning fields:

```text
observation.state        float[20] measured robot positions
                         same order as joint_map.json

action                   float[20] commanded target positions
                         same order as joint_map.json

observation.images.*     synchronized RGB cameras
```

Recommended optional debug metadata (only if the existing LeRobot schema supports it cleanly):

- raw/reduced Quest hand landmarks;
- Quest wrist pose;
- retargeting status/error;
- arm IK residual;
- calibration ID/version;
- dropped/stale-frame counters.

Quest skeleton data is **not** the imitation-learning action. The action is the robot's canonical 20-DoF command.

## 9. Deliverables Cursor should produce

1. Combined Meta Quest teleop command that drives both the AM arm and ORCA right hand.
2. Combined `sim` backend using this ZIP's `scene.xml`.
3. Combined `hardware` backend with a clean AM driver adapter plus existing ORCA hardware path.
4. A deterministic canonical 20-DoF joint order assertion.
5. Calibration routine for Quest wrist -> AM workspace.
6. AM 3DoF IK + workspace/joint/rate safety limits.
7. Existing LeRobot `record_dataset.py` flow extended to record 20-DoF state/action and cameras.
8. Smoke tests for shape/order/safety timeout and a one-episode sim recording test.
9. No patch-at-runtime workflow for model assets; all model files should be referenced directly.

## 10. Acceptance test

The integration is ready for hardware trials when one command can:

1. launch Meta Quest WebXR input;
2. show/control the combined right-hand MuJoCo model;
3. move the three AM joints from Quest wrist translation via IK;
4. move the ORCA right-hand joints via the existing retargeter;
5. produce one LeRobot episode with 20-D `observation.state`, 20-D `action`, and at least one camera stream;
6. switch from sim to hardware without changing the retargeting/data contract.
