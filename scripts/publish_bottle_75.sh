#!/usr/bin/env bash
# Explicit public-publication job for the frozen, curated 75-episode collection.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/orca_teleop/.venv/bin/python"
OUTPUT='/media/lee/My Book/orcahand_lerobot/orca-hand-bottle-v1-75'
SOURCE='/media/lee/My Book/orcahand_data/hwatu_20261108_224923_2a8b15'
cd "$ROOT"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HUB_DISABLE_TELEMETRY=1
export HF_DATASETS_CACHE="$ROOT/data/act/.hf_cache"
export HF_XET_CACHE='/media/lee/My Book/orcahand_lerobot/.xet_cache'
if [[ -f "$SOURCE/.conversion.lock" ]]; then
  printf 'Waiting for current source conversion to finish...\n'
  flock "$SOURCE/.conversion.lock" true
fi
"$PY" -u "$ROOT/scripts/prepare_public_bottle_dataset.py" --output "$OUTPUT" \
  --inventory "${ORCA_PUBLIC_INVENTORY:-$ROOT/data/act/external_dataset_inventory.json}"
"$PY" -u "$ROOT/scripts/upload_public_bottle_dataset.py" --dataset-root "$OUTPUT" \
  --repo-id "${1:-saybeok03/orca-hand-bottle}"
