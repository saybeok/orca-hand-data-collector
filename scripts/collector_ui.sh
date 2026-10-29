#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/../env.sh"
cd "$ORCA_HAND_ROOT"
# Ubuntu Tk renders Korean through native fontconfig/Xft. The recorder still
# runs in the project venv, independently of the desktop Python interpreter.
export PYTHONPATH="$ORCA_HAND_ROOT/orca_teleop/src"
exec /usr/bin/python3 -m orca_teleop.collector_app "$@"
