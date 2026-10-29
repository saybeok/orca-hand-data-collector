# CURSOR PROMPT — paste this into Cursor

You are extending the official `orcahand/orca_teleop` repository for a custom robot embodiment. Read `START_HERE.md`, `IMPLEMENTATION_SPEC.md`, and `joint_map.json` first, then inspect `robot_model/scene.xml`.

Use https://github.com/orcahand/orca_teleop as the main codebase. Do not build a separate teleoperation framework and do not reimplement components already provided by upstream.

Target robot:
- AM-ARM200: 3 DoF arm (`shoulder_pan`, `shoulder_lift`, `elbow_flex`), STS3095 x3.
- ORCA: right hand, 17 DoF.
- Canonical action/state: exactly 20 DoF in the order in `joint_map.json`.
- Meta Quest 3S is the teleoperation input.
- The existing upstream ORCA right-hand retargeting must be preserved unless a verified bug requires a change.
- Extend the Quest wrist path to control the AM arm: calibrated relative wrist translation -> Cartesian end-effector target -> 3DoF IK -> joint targets. Never map Quest XYZ directly to three joint angles.
- Generalize/composite the action/state/backend path so sim, hardware, and LeRobot recording all use the identical 20-DoF ordering.
- Use `robot_model/scene.xml` as the canonical MuJoCo combined robot. Do not regenerate or runtime-patch the model.
- Extend the existing `scripts/record_dataset.py`/LeRobot path instead of creating another dataset logger.
- Imitation-learning action is the commanded robot 20-DoF vector. Quest data is optional debug/retargeting input metadata, not policy action.
- Simulation is primarily a teleoperation/safety/data-contract validation backend; physical AM-ARM200 + ORCA demonstrations are the primary dataset target.

Important missing information: this handoff does not contain verified AM-ARM200 serial/servo IDs/baud/home offsets/sign conventions/hardware SDK wiring. Create the AM hardware adapter boundary and configuration, but do not invent those values. Flag the exact values needed from the user before enabling physical AM motion.

Implement this as production-quality code: typed interfaces, deterministic joint ordering, calibration persistence, workspace/joint/rate limits, stale-input timeout, clear sim/hardware separation, tests, and minimal changes to upstream ORCA code.

Start by mapping the current upstream modules and propose the smallest set of source-file changes. Then implement them. Do not modify the ORCA hand's MJCF internal joint transforms or mesh geometry.
