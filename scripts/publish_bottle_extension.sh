#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/orca_teleop/.venv/bin/python"
OUTPUT='/media/lee/My Book/orcahand_lerobot/orca-hand-bottle-v2-165'
cd "$ROOT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_DATASETS_CACHE="$ROOT/data/act/.hf_cache"
export HF_XET_CACHE="$ROOT/data/act/.xet_upload_cache"
export HF_XET_FIXED_UPLOAD_CONCURRENCY=4
export HF_XET_CLIENT_READ_TIMEOUT=900s
export HF_HUB_DISABLE_TELEMETRY=1
export TMPDIR='/media/lee/My Book/orcahand_lerobot/.conversion_tmp'
mkdir -p "$TMPDIR"
"$PY" -u "$ROOT/scripts/extend_public_bottle_dataset.py" prepare --output "$OUTPUT"
"$PY" -u "$ROOT/scripts/extend_public_bottle_dataset.py" publish --output "$OUTPUT"
