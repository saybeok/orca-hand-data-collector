#!/usr/bin/env bash
# Quest WebXR -> retargeter -> physical ORCA hand.
#
# USB (recommended):
#   adb devices
#   adb reverse tcp:8765 tcp:8765
#   ./scripts/teleop_quest_hw.sh
#   Quest Browser → http://localhost:8765 → Start hand tracking
#
# Wi‑Fi: open http://<PC_IP>:8765 (may need HTTPS / ngrok for WebXR).
set -euo pipefail

source "$(dirname "$(readlink -f "$0")")/../env.sh"
cd "$ORCA_HAND_ROOT/orca_teleop"

exec .venv/bin/python "$ORCA_HAND_ROOT/scripts/teleop_quest_hw.py" \
  --hand right \
  --model_path "$ORCA_MODEL_PATH" \
  --urdf_path "$ORCA_URDF_PATH" \
  --retarget-config "$ORCA_RETARGET_CONFIG" \
  "$@"
