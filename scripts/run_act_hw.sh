#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/../env.sh"
cd "$ORCA_HAND_ROOT"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec "$ORCA_HAND_ROOT/orca_teleop/.venv/bin/python" \
  "$ORCA_HAND_ROOT/orca_teleop/scripts/run_act_hw.py" "$@"
