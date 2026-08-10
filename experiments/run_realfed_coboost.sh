#!/usr/bin/env bash
set -euo pipefail

ROOT=/public/home/dongshou/fedETF
PY=/public/home/dongshou/anaconda/envs/ct/bin/python
OUT="$ROOT/realfed_out"
SCRIPT="$ROOT/review_response/experiments/realfed_coboost.py"

export LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/opencl/lib:${LD_LIBRARY_PATH:-}
export HIP_PATH=/opt/dtk ROCM_PATH=/opt/dtk DTKROOT=/opt/dtk

mkdir -p "$OUT/logs" "$OUT/results" "$OUT/checkpoints"
cd "$ROOT"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 GPU:SEED [GPU:SEED ...]" >&2
  exit 2
fi

wait_for_teachers() {
  local seed=$1
  local teacher_dir="$OUT/checkpoints/realfed_binary_ce_heldout-none_s${seed}"
  while true; do
    local ready=1
    for source in brset mbrset odir; do
      [[ -f "$teacher_dir/${source}.pt" ]] || ready=0
    done
    if [[ $ready -eq 1 ]]; then
      return
    fi
    echo "[$(date '+%F %T')] waiting for CE teachers seed=${seed}"
    sleep 60
  done
}

run_one() {
  local gpu=$1 seed=$2
  local tag="realfed_binary_coboost_heldout-none_s${seed}"
  if [[ -f "$OUT/results/${tag}.json" ]]; then
    echo "[$(date '+%F %T')] [g${gpu}] skip complete ${tag}"
    return
  fi
  wait_for_teachers "$seed"
  echo "[$(date '+%F %T')] [g${gpu}] start ${tag}"
  CUDA_VISIBLE_DEVICES=$gpu HIP_VISIBLE_DEVICES=$gpu \
    "$PY" -u "$SCRIPT" \
      --data_root "$ROOT/realfed_data" --output "$OUT" \
      --teacher_output "$OUT" --seed "$seed" --epochs 200 \
      --image_size 224 --synth_size 128 --batch_size 64 --g_steps 30 \
      --kd_lr 0.01 --lr_g 1e-3 --kd_temperature 4 \
      --save_every 5 --workers 2 \
      > "$OUT/logs/${tag}.log" 2>&1
  echo "[$(date '+%F %T')] [g${gpu}] done ${tag}"
}

for assignment in "$@"; do
  gpu=${assignment%%:*}
  seed=${assignment##*:}
  run_one "$gpu" "$seed" &
done
wait
echo "[$(date '+%F %T')] Co-Boosting assignments complete: $*"
