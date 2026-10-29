#!/usr/bin/env bash
# Quest / webcam -> IK + retarget -> physical AM-ARM200 + ORCA hand.
#
# Ports (this unit) — each config names its own USB adapter; the sink refuses
# to start if one is missing or both resolve to the same device:
#   hand  configs/orcahand-right-v1-feetech/config.yaml
#   arm   configs/am-arm200-3dof/config.yaml
#
# Quest (arm moves; adb reverse is set up automatically):
#   ./scripts/teleop_am200_hw.sh --clutch --arm-scale 0.35 --arm-max-step 0.3
# Off-hand (left when teleoping right):
#   pinch  = arm clutch (follow while held)
#   fist   = re-run hand retarget auto-scale (~1 s neutral pose)
#
# Webcam (hand only; arm holds connect pose — no wrist pose on MediaPipe):
#   ./scripts/teleop_am200_hw.sh --source webcam --show-video --no-mirror --camera-id 0
#
# Hand-only on hardware (arm bus closed):
#   ./scripts/teleop_am200_hw.sh --no-arm --source webcam --show-video --no-mirror --camera-id 0
# Quest pinch retargeting trial, hand only:
#   ./scripts/teleop_am200_hw.sh --source quest --no-arm --pinch
# Continuous empirical skin-clearance calibration (hand.sh contacts fit-surface):
#   ./scripts/teleop_am200_hw.sh --no-arm --pinch-calibrated --hand-feedback-debug
# Recorded physical contact configurations (run hand.sh contacts teach first):
#   ./scripts/teleop_am200_hw.sh --no-arm --pinch-taught --hand-feedback-debug
# Reconstructed earlier, abrupt pinch transition (explicit comparison only):
#   ./scripts/teleop_am200_hw.sh --source quest --no-arm --pinch-legacy
set -euo pipefail

source "$(dirname "$(readlink -f "$0")")/../env.sh"
cd "$ORCA_HAND_ROOT/orca_teleop"

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

exec .venv/bin/python scripts/teleop_am200_hw.py \
  --source "$source_arg" \
  --hand "$hand_arg" \
  --hand-config "$ORCA_MODEL_PATH" \
  --urdf_path "$ORCA_URDF_PATH" \
  --retarget-config "$retarget_arg" \
  --retargeter "$backend_arg" \
  --arm-max-velocity-deg-s 15 --arm-max-acceleration-deg-s2 45 \
  ${args[@]+"${args[@]}"}
