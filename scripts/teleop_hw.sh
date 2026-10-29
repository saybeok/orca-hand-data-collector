#!/usr/bin/env bash
# Webcam -> MediaPipe -> retargeter -> physical ORCA hand.
# Same flags as teleop_sim.sh (default MediaPipe mirror). No joint remapping hacks.
set -euo pipefail

source "$(dirname "$(readlink -f "$0")")/../env.sh"
cd "$ORCA_HAND_ROOT/orca_teleop"

args=(
  --hand right
  --show-video
  --model_path "$ORCA_MODEL_PATH"
  --urdf_path "$ORCA_URDF_PATH"
  --retarget-config "$ORCA_RETARGET_CONFIG"
)
if [ -n "${ORCA_CAMERA_ID:-}" ]; then
  args+=(--camera-id "$ORCA_CAMERA_ID")
fi

exec .venv/bin/python "$ORCA_HAND_ROOT/scripts/teleop_hw.py" "${args[@]}" "$@"
