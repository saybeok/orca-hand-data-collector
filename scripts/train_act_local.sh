#!/usr/bin/env bash
# Local-only ACT baseline for the selected bottle demonstrations.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/orca_teleop/.venv/bin/python"
CAPTURE="${ORCA_ACT_CAPTURE:-$ROOT/data/act/bottle_20261108/capture}"
if [[ -z "${ORCA_ACT_CAPTURE:-}" && -f "$ROOT/data/act/active_dataset.json" ]]; then
  CAPTURE="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["active_capture"])' "$ROOT/data/act/active_dataset.json")"
fi
DATASET="$CAPTURE/lerobot"
OUTPUT="${ORCA_ACT_OUTPUT:-$ROOT/outputs/train/act_bottle_$(date +%Y%m%d_%H%M%S)}"
cd "$ROOT"

export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled
export ACCELERATE_MIXED_PRECISION=fp16
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU is unavailable; run from the PC terminal with GPU access."; print("ACT GPU:", torch.cuda.get_device_name(0), flush=True)'
if [[ -f "$CAPTURE/.conversion.lock" ]]; then
  printf 'Waiting for any current dataset conversion to finish...\n'
  flock "$CAPTURE/.conversion.lock" true
fi
if [[ -f "$DATASET/conversion.json" && -f "$DATASET/collection_act_split.json" ]]; then
  "$PY" - "$CAPTURE" <<'PY'
import json,sys
from pathlib import Path
from orca_teleop.deferred_recording import capture_fingerprint
root=Path(sys.argv[1])
assert json.loads((root/'lerobot/conversion.json').read_text())['source_sha256']==capture_fingerprint(root), 'Capture changed after conversion; refusing stale dataset.'
PY
else
  "$PY" "$ROOT/orca_teleop/scripts/prepare_act_dataset.py" \
    --sources "$ROOT/configs/act_bottle_20261108_sources.json" --destination "$CAPTURE"
fi
REPO_ID="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["repo_id"])' "$CAPTURE/capture.json")"
EPISODES="$("$PY" -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["train_episodes"]))' "$CAPTURE/collection_act_preparation.json")"
printf 'Dataset: %s\nOutput: %s\nTraining episodes: %s\n' "$DATASET" "$OUTPUT" "$EPISODES"

exec "$PY" -u -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$DATASET" \
  --dataset.episodes="$EPISODES" \
  --dataset.video_backend=pyav \
  --dataset.use_imagenet_stats=true \
  --policy.type=act \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.push_to_hub=false \
  --policy.chunk_size=100 \
  --policy.n_action_steps=10 \
  --batch_size=4 \
  --num_workers=2 \
  --steps=20000 \
  --save_freq=5000 \
  --log_freq=50 \
  --eval_freq=0 \
  --seed=42 \
  --wandb.enable=false \
  --output_dir="$OUTPUT" \
  "$@"
