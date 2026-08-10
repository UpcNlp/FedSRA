#!/usr/bin/env bash
set -euo pipefail

ROOT=/public/home/dongshou/fedETF
PY=/public/home/dongshou/anaconda/envs/ct/bin/python
OUT="$ROOT/realfed_out"
SCRIPT="$ROOT/review_response/experiments/realfed_fundus.py"

export LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/opencl/lib:${LD_LIBRARY_PATH:-}
export HIP_PATH=/opt/dtk ROCM_PATH=/opt/dtk DTKROOT=/opt/dtk

mkdir -p "$OUT/logs" "$OUT/results" "$OUT/checkpoints"
cd "$ROOT"

run_one() {
  local gpu=$1 seed=$2
  local tag="realfed_binary_ce_heldout-none_s${seed}"
  if [[ -f "$OUT/results/${tag}.json" ]]; then
    echo "[$(date '+%F %T')] [g${gpu}] skip complete ${tag}"
    return
  fi
  echo "[$(date '+%F %T')] [g${gpu}] start ${tag}"
  CUDA_VISIBLE_DEVICES=$gpu HIP_VISIBLE_DEVICES=$gpu \
    "$PY" -u "$SCRIPT" \
      --method ce --data_root "$ROOT/realfed_data" --output "$OUT" \
      --seed "$seed" --epochs 30 --image_size 224 --batch_size 64 \
      --workers 2 --lr 1e-4 --save_every 5 \
      > "$OUT/logs/${tag}.log" 2>&1
  echo "[$(date '+%F %T')] [g${gpu}] done ${tag}"
}

run_one 3 0 &
run_one 4 42 &
run_one 5 123 &
wait
echo "[$(date '+%F %T')] CE core complete"
