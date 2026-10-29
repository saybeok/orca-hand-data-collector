#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/../env.sh"
cd "$ORCA_HAND_ROOT"
exec "$ORCA_HAND_ROOT/orca_teleop/.venv/bin/python" scripts/arm_sysid_fit.py "$@"
