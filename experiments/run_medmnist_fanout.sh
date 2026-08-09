#!/bin/bash
# Group C: MedMNIST FedSRA matrix fanned across 8 GPUs (PathMNIST + OrganAMNIST).
# 2 datasets x 12 cells = 24 cells. Resume-safe (skips existing result JSON).
# Usage: bash run_medmnist_fanout.sh
ROOT=/public/home/dongshou/fedETF
DATA=$ROOT/medmnist_data
OUT=$ROOT/medmnist_out
PY=/public/home/dongshou/anaconda/envs/ct/bin/python
SDIR=$ROOT/review_response/experiments
export LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/opencl/lib:$LD_LIBRARY_PATH
export HIP_PATH=/opt/dtk ROCM_PATH=/opt/dtk DTKROOT=/opt/dtk
EPOCHS=${EPOCHS:-100}

# cells: "dataset alpha noise seed"
CELLS=()
for ds in pathmnist organamnist; do
  for s in 42 0 123; do
    CELLS+=("$ds 0.05 0.0 $s" "$ds 0.05 0.2 $s" "$ds 0.05 0.4 $s" "$ds 0.3 0.0 $s")
  done
done
N=${#CELLS[@]}

run_cell () {
  local gpu=$1 ds=$2 a=$3 noise=$4 seed=$5
  local A_TAG=${a/./p} N_TAG=${noise/./p}
  local oroot=$OUT/$ds
  local tag="${ds}_a${A_TAG}_k5_noise${N_TAG}_s${seed}"
  local result="$oroot/results/${tag}.json"
  mkdir -p "$oroot/logs" "$oroot/results"
  [ -f "$result" ] && { echo "[g$gpu] skip $tag (resume)"; return; }
  local CORR=()
  [ "$noise" == "0.0" ] && [ "$a" == "0.05" ] && \
    CORR=(--corruptions pixelate,jpeg_compression,brightness_down,contrast_down --severities 1,3,5)
  HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu "$PY" -u "$SDIR/medmnist_fedsra.py" \
    --dataset "$ds" --data "$DATA/${ds}.npz" --output "$oroot" \
    --alpha "$a" --n_clients 5 --noise_rate "$noise" --seed "$seed" \
    --epochs "$EPOCHS" --batch_size 256 --workers 4 --save_every 10 \
    "${CORR[@]}" > "$oroot/logs/${tag}.log" 2>&1
  echo "[$(date +%H:%M:%S)][g$gpu] done $tag rc=$?"
}

for gpu in 0 1 2 3 4 5 6 7; do
  ( i=$gpu; while [ $i -lt $N ]; do run_cell $gpu ${CELLS[$i]}; i=$((i+8)); done ) &
done
wait
echo "MEDMNIST MATRIX COMPLETE $(date)"
