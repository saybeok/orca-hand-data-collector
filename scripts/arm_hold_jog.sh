#!/usr/bin/env bash
# Hold + tiny jog the physical AM-ARM200 3-DoF (no hand motion).
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/../env.sh"
cd "$ORCA_HAND_ROOT"
exec "$ORCA_HAND_ROOT/orca_teleop/.venv/bin/python" scripts/arm_hold_jog.py "$@"
