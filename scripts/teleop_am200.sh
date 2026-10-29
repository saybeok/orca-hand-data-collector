#!/usr/bin/env bash
# Quest 3S (or webcam) -> retargeter -> AM-ARM200 + ORCA right hand v1 in MuJoCo.
#
# All 20 joints follow the operator: the hand from the retargeter, the arm from
# position IK on the wrist pose. Three arm joints can track wrist *position*
# only, not its rotation. A webcam sends no wrist pose, so the arm holds still.
#
# Default: first-frame relative follow. Optional absolute mapping:
#   python orca_am200/scripts/calibrate_operator.py
#
# With --clutch, pinch your OTHER hand to let the arm follow and release to park
# it, so you can reposition without dragging the robot.
#
# For --source quest the headset needs a secure context. Over USB:
#   adb devices                       # the Quest must appear as 'device'
#   adb reverse tcp:8765 tcp:8765
# then open http://localhost:8765 in Quest Browser and tap Start hand tracking.
#
# Usage:
#   ./scripts/teleop_am200.sh                            # Quest, arm + hand
#   ./scripts/teleop_am200.sh --clutch                   # off-hand pinch clutch
#   ./scripts/teleop_am200.sh --no-arm                   # hand only
#   ./scripts/teleop_am200.sh --no-arm --pinch-legacy    # earlier abrupt pinch comparison
#   ./scripts/teleop_am200.sh --arm-scale 0.35           # gentler arm motion
#   ./scripts/teleop_am200.sh --source webcam --show-video
set -euo pipefail

source "$(dirname "$(readlink -f "$0")")/../env.sh"
cd "$ORCA_HAND_ROOT/orca_teleop"

# Defaults the caller can override, since a repeated flag would otherwise be
# passed twice and only silently work because argparse keeps the last one.
source_arg="quest"
hand_arg="right"
retarget_arg="$ORCA_RETARGET_CONFIG"
backend_arg="adaptive_analytical"
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --source) source_arg="$2"; shift 2 ;;
    --hand)   hand_arg="$2";   shift 2 ;;
    --retargeter) backend_arg="$2"; shift 2 ;;
    --dexpilot) backend_arg="dexpilot"; retarget_arg="$ORCA_HAND_ROOT/configs/dexpilot_orca_v1.yaml"; shift ;;
    --key-objectives) backend_arg="key_objectives"; retarget_arg="$ORCA_HAND_ROOT/configs/key_objectives_orca_v1.yaml"; shift ;;
    --pinch) retarget_arg="$ORCA_HAND_ROOT/configs/adaptive_analytical_orca_v1_pinch.yaml"; shift ;;
    --pinch-refined) retarget_arg="$ORCA_HAND_ROOT/configs/adaptive_analytical_orca_v1_pinch_refined.yaml"; shift ;;
    --pinch-calibrated) retarget_arg="$ORCA_HAND_ROOT/configs/adaptive_analytical_orca_v1_pinch_calibrated.yaml"; shift ;;
    --pinch-taught) retarget_arg="$ORCA_HAND_ROOT/configs/adaptive_analytical_orca_v1_pinch_taught.yaml"; shift ;;
    --pinch-legacy) retarget_arg="$ORCA_HAND_ROOT/configs/adaptive_analytical_orca_v1_pinch_legacy.yaml"; shift ;;
    --retarget-config) retarget_arg="$2"; shift 2 ;;
    *)        args+=("$1");    shift   ;;
  esac
done

if [ "$source_arg" = "quest" ]; then
  adb reverse tcp:8765 tcp:8765 >/dev/null 2>&1 \
    && echo "Quest: open http://localhost:8765 in Quest Browser" \
    || echo "WARNING: adb reverse failed — connect the Quest by USB and allow USB debugging" >&2
fi

exec .venv/bin/python scripts/teleop_am200_sim.py \
  --source "$source_arg" \
  --hand "$hand_arg" \
  --hand-config "$ORCA_MODEL_PATH" \
  --urdf_path "$ORCA_URDF_PATH" \
  --retarget-config "$retarget_arg" \
  --retargeter "$backend_arg" \
  ${args[@]+"${args[@]}"}
