#!/usr/bin/env bash
set -u

# One serial, resumable master per RTX 5090 host.
# Usage:
#   bash run_medmnist_matrix.sh pathmnist /path/to/pathmnist.npz /output/root /python/path

DATASET=${1:?dataset required}
DATA_FILE=${2:?data .npz required}
OUT_ROOT=${3:?output root required}
PY=${4:-python}
EPOCHS=${EPOCHS:-100}
GPU=${GPU:-0}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/results" "$OUT_ROOT/checkpoints"

CELLS=(
  "0.05 0.0 42"
  "0.05 0.0 0"
  "0.05 0.0 123"
  "0.05 0.2 42"
  "0.05 0.2 0"
  "0.05 0.2 123"
  "0.05 0.4 42"
  "0.05 0.4 0"
  "0.05 0.4 123"
  "0.3 0.0 42"
  "0.3 0.0 0"
  "0.3 0.0 123"
)

if [[ "${PILOT_ONLY:-0}" == "1" ]]; then
  CELLS=("0.05 0.0 42")
  echo "[$(date '+%F %T')] PILOT_ONLY=1: running one clean seed-42 cell"
fi

for cell in "${CELLS[@]}"; do
  read -r ALPHA NOISE SEED <<< "$cell"
  A_TAG=${ALPHA/./p}
  N_TAG=${NOISE/./p}
  TAG="${DATASET}_a${A_TAG}_k5_noise${N_TAG}_s${SEED}"
  RESULT="$OUT_ROOT/results/${TAG}.json"
  LOG="$OUT_ROOT/logs/${TAG}.log"
  if [[ -f "$RESULT" ]]; then
    echo "[$(date '+%F %T')] skip complete $TAG"
    continue
  fi
  CORR_ARGS=()
  if [[ "$NOISE" == "0.0" && "$ALPHA" == "0.05" ]]; then
    CORR_ARGS=(--corruptions pixelate,jpeg_compression,brightness_down,contrast_down --severities 1,3,5)
  fi
  echo "[$(date '+%F %T')] start $TAG"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" -u "$SCRIPT_DIR/medmnist_fedsra.py" \
    --dataset "$DATASET" --data "$DATA_FILE" --output "$OUT_ROOT" \
    --alpha "$ALPHA" --n_clients 5 --noise_rate "$NOISE" --seed "$SEED" \
    --epochs "$EPOCHS" --batch_size 256 --workers 4 --save_every 10 \
    "${CORR_ARGS[@]}" > "$LOG" 2>&1
  RC=$?
  echo "[$(date '+%F %T')] finish $TAG rc=$RC"
  if [[ $RC -ne 0 ]]; then
    echo "FAILED $TAG; see $LOG" >&2
    exit $RC
  fi
done

echo "[$(date '+%F %T')] MATRIX COMPLETE dataset=$DATASET"
