#!/bin/bash
set +u; source /opt/dtk/env.sh 2>/dev/null || true; set -u
cd /public/home/dongshou/fedETF/ETF-pesuade

# Back up existing 4-norm results
mkdir -p results/_backup_pre_v2
for f in results/ablation_agg_cifar100_a*_k*_s42.json; do
  [ -f "$f" ] && cp "$f" results/_backup_pre_v2/ 2>/dev/null
done
echo "Backed up existing files to results/_backup_pre_v2/"

PYTHON=/public/home/dongshou/anaconda/envs/ct/bin/python
LOGDIR=logs/gpa_variants_v2
mkdir -p $LOGDIR

# Job list: 16 cells (CIFAR-100 × 4 K × 4 α)
JOBS=()
for K in 5 10 20 50; do
  for a in 0.05 0.1 0.3 0.5; do
    JOBS+=("$K $a")
  done
done

NUM_GPU=4
i=0
for job in "${JOBS[@]}"; do
  read K a <<< "$job"
  gpu=$((i % NUM_GPU))
  LOG=$LOGDIR/c100_K${K}_a${a}.log
  echo "[GPU $gpu] CIFAR-100 K=$K α=$a -> $LOG"
  HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
    $PYTHON ablation_aggregation_v2.py \
      --dataset cifar100 \
      --alpha $a \
      --n_clients $K \
      --seed 42 \
      > $LOG 2>&1 &
  i=$((i+1))
  if (( i % NUM_GPU == 0 )); then wait; fi
done
wait
echo "===== ALL DONE ====="
