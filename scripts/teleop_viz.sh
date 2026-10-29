#!/usr/bin/env bash
# Webcam -> MediaPipe -> retargeter -> Meshcat URDF viewer. No hardware touched.
# Open http://127.0.0.1:7000/static/ and hold your right hand still in frame
# until the retargeter finishes auto-scaling.
set -euo pipefail

source "$(dirname "$(readlink -f "$0")")/../env.sh"
cd "$ORCA_HAND_ROOT/orca_teleop"

exec .venv/bin/python scripts/teleop_urdf.py \
  --hand right \
  --version v1 \
  --local \
  --show-video \
  --model_path "$ORCA_MODEL_PATH" \
  --urdf_path "$ORCA_URDF_PATH" \
  --retarget-config "$ORCA_RETARGET_CONFIG" \
  "$@"
