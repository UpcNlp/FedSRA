#!/usr/bin/env bash
set -euo pipefail

ROOT=/public/home/dongshou/fedETF
PY=/public/home/dongshou/anaconda/envs/ct/bin/python
OUT="$ROOT/realfed_out"

export LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/opencl/lib:${LD_LIBRARY_PATH:-}
export HIP_PATH=/opt/dtk ROCM_PATH=/opt/dtk DTKROOT=/opt/dtk

mkdir -p "$OUT/logs" "$OUT/results" "$OUT/checkpoints"
cd "$ROOT"

if [[ $# -lt 2 ]]; then
  echo "usage: $0 METHOD GPU:SEED [GPU:SEED ...]" >&2
  exit 2
fi
method=$1
shift
if [[ "$method" != fedsra && "$method" != ce && "$method" != fafi ]]; then
  echo "METHOD must be fedsra, ce, or fafi" >&2
  exit 2
fi

wait_for_main() {
  local seed=$1
  local main_tag="realfed_binary_${method}_heldout-none_s${seed}"
  while [[ ! -f "$OUT/results/${main_tag}.json" ]]; do
    echo "[$(date '+%F %T')] waiting for main ${main_tag}"
    sleep 60
  done
}

run_one() {
  local gpu=$1 seed=$2
  local tag="realfed_binary_${method}_heldout-mbrset_s${seed}"
  if [[ -f "$OUT/results/${tag}.json" ]]; then
    echo "[$(date '+%F %T')] [g${gpu}] skip complete ${tag}"
    return
  fi
  wait_for_main "$seed"
  echo "[$(date '+%F %T')] [g${gpu}] start ${tag}"
  if [[ "$method" == fafi ]]; then
    CUDA_VISIBLE_DEVICES=$gpu HIP_VISIBLE_DEVICES=$gpu \
      "$PY" -u "$ROOT/review_response/experiments/realfed_fafi.py" \
        --data_root "$ROOT/realfed_data" --output "$OUT" \
        --seed "$seed" --epochs 30 --image_size 224 --batch_size 32 \
        --workers 2 --lr 1e-4 --save_every 5 --heldout mbrset \
        > "$OUT/logs/${tag}.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES=$gpu HIP_VISIBLE_DEVICES=$gpu \
      "$PY" -u "$ROOT/review_response/experiments/realfed_fundus.py" \
        --method "$method" --data_root "$ROOT/realfed_data" --output "$OUT" \
        --seed "$seed" --epochs 30 --image_size 224 --batch_size 64 \
        --workers 2 --lr 1e-4 --save_every 5 --heldout mbrset \
        > "$OUT/logs/${tag}.log" 2>&1
  fi
  echo "[$(date '+%F %T')] [g${gpu}] done ${tag}"
}

for assignment in "$@"; do
  gpu=${assignment%%:*}
  seed=${assignment##*:}
  run_one "$gpu" "$seed" &
done
wait
echo "[$(date '+%F %T')] held-out ${method} complete: $*"
