#!/usr/bin/env bash
set -u

ROOT=/public/home/dongshou/fedETF
SCRIPT=$ROOT/review_response/experiments/realfed_fundus.py
DATA=$ROOT/realfed_data
OUT=$ROOT/realfed_out
PY=/public/home/dongshou/anaconda/envs/ct/bin/python

export LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/opencl/lib:$LD_LIBRARY_PATH
export HIP_PATH=/opt/dtk ROCM_PATH=/opt/dtk DTKROOT=/opt/dtk

mkdir -p "$OUT/logs" "$OUT/results" "$OUT/checkpoints"

# Populate this container's torchvision cache before concurrent workers start.
$PY -c 'from torchvision import models; models.resnet18(weights=models.ResNet18_Weights.DEFAULT)' \
  > "$OUT/logs/imagenet_weight_cache.log" 2>&1

CELLS=(
  "0 fedsra 0"
  "1 fedsra 42"
  "2 fedsra 123"
  "3 ce 0"
  "4 ce 42"
  "5 ce 123"
)

run_cell () {
  local gpu=$1 method=$2 seed=$3
  local tag="realfed_binary_${method}_heldout-none_s${seed}"
  local result="$OUT/results/${tag}.json"
  local log="$OUT/logs/${tag}.log"
  if [ -f "$result" ]; then
    echo "[$(date '+%F %T')] skip complete $tag"
    return
  fi
  echo "[$(date '+%F %T')] [g$gpu] start $tag"
  HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
    $PY -u "$SCRIPT" \
      --method "$method" --data_root "$DATA" --output "$OUT" \
      --seed "$seed" --epochs 30 --image_size 224 --batch_size 64 \
      --workers 2 --lr 1e-4 --save_every 5 \
      > "$log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] [g$gpu] finish $tag rc=$rc"
  return $rc
}

pids=()
for cell in "${CELLS[@]}"; do
  run_cell $cell &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done

if [ "$failed" -ne 0 ]; then
  echo "REALFED CORE FAILED; inspect $OUT/logs"
  exit 1
fi
echo "REALFED CORE COMPLETE $(date)"
