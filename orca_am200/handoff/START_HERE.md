# START HERE — Cursor handoff

## Goal

Use the official `orcahand/orca_teleop` repository as the **main codebase** and extend it for this embodiment:

- AM-ARM200 arm: 3 DoF (`shoulder_pan`, `shoulder_lift`, `elbow_flex`)
- ORCA Hand: **right hand**, 17 DoF
- Canonical robot action/state: **20 DoF total**
- Input device: Meta Quest 3S via the repository's existing Meta Quest WebXR path
- Dataset: extend the repository's existing LeRobot recording path

Upstream: https://github.com/orcahand/orca_teleop

The upstream README currently documents Meta Quest WebXR recording with `scripts/record_dataset.py`, `--source metaquest`, `--hand right`, both `--backend sim` and `--backend hardware`, and LeRobot recording. **Do not reimplement those components from scratch.**

## Files in this ZIP

- `robot_model/scene.xml` — canonical MuJoCo model for AM-ARM200 3DoF + ORCA right hand.
- `robot_model/assets/` — all local meshes required by `scene.xml`; no asset download/patch step should be required.
- `robot_model/run.py`, `test_load.py`, `validate.py` — model smoke tests / viewer helpers.
- `joint_map.json` — exact 20-DoF joint and actuator order extracted from `scene.xml`.
- `IMPLEMENTATION_SPEC.md` — what to implement in `orca_teleop`.
- `CURSOR_PROMPT.md` — ready-to-paste task prompt.

## Non-negotiable rules

1. Treat `orca_teleop` as upstream/base. Do not build a separate teleop framework beside it.
2. Preserve the existing Quest WebXR ingress and ORCA retargeting unless a verified bug requires a change.
3. Add AM arm retargeting from Quest wrist motion to a Cartesian target and solve 3-DoF IK. Do **not** map Quest XYZ directly to three motor angles.
4. Generalize/composite the robot action/state path so one ordered 20-DoF vector reaches sim, hardware, and LeRobot recording consistently.
5. `scene.xml` is the canonical sim embodiment. Do not regenerate it or patch its geometry/joint transforms casually.
6. Simulation is primarily for teleop/safety/integration validation. The target imitation-learning demonstrations are physical-robot episodes.
7. There is no AM-ARM200 real-hardware transport/serial driver included in this handoff. Do not invent ports, IDs, baud rates, or SDK calls. Create a clean adapter boundary and wire it to the user's real AM driver/config when provided.

## Canonical 20-DoF ordering

Always use `joint_map.json` as source of truth. Indices 0..2 are AM arm. Indices 3..19 are the ORCA right-hand joints in the exact order used by `scene.xml`.
