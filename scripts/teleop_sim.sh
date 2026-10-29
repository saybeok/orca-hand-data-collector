#!/usr/bin/env bash
# Webcam -> MediaPipe -> retargeter -> MuJoCo physics sim (orca_sim).
# Heavier than teleop_viz.sh but simulates contact, so you can grasp objects.
set -euo pipefail

source "$(dirname "$(readlink -f "$0")")/../env.sh"
cd "$ORCA_HAND_ROOT/orca_teleop"

exec .venv/bin/python scripts/teleop_sim.py \
  --env right \
  --version v1 \
  --local \
  --show-video \
  --model_path "$ORCA_MODEL_PATH" \
  --urdf_path "$ORCA_URDF_PATH" \
  --retarget-config "$ORCA_RETARGET_CONFIG" \
  "$@"
