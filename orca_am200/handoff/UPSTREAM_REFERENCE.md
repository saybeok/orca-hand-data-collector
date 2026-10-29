# Upstream reference

Primary codebase:
https://github.com/orcahand/orca_teleop

Checked against the upstream `main` README on 2026-08-23 (Asia/Seoul). The README documents:

- default adaptive analytical ORCA retargeting;
- Meta Quest WebXR demonstration recording;
- `scripts/record_dataset.py` with `--source metaquest` and `--hand right`;
- both `--backend sim` and `--backend hardware` paths;
- LeRobot recording of measured hardware positions as `observation.state`, commanded positions as `action`, and cameras as `observation.images.<name>`.

Cursor should inspect the current checkout rather than assuming file internals from this handoff, because upstream may change.
