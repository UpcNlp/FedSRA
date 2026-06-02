#!/bin/bash
set +u; source /opt/dtk/env.sh 2>/dev/null || true; set -u
set -e

cd /public/home/dongshou/fedETF/ETF-pesuade
PYTHON=/public/home/dongshou/anaconda/envs/ct/bin/python
LOGDIR=./logs/r_train_seeds
mkdir -p $LOGDIR

# 12 cells: 8 cifar10 (all α × {0,123}) + 4 cifar100 (α=.3,.5 × {0,123})
CELLS=(
  "cifar10  0.05 0"   "cifar10  0.1 0"   "cifar10  0.3 0"   "cifar10  0.5 0"
  "cifar10  0.05 123" "cifar10  0.1 123" "cifar10  0.3 123" "cifar10  0.5 123"
  "cifar100 0.3 0"    "cifar100 0.5 0"   "cifar100 0.3 123" "cifar100 0.5 123"
)
NUM_GPU=7  # GPUs 0-6; skip 7 (busy)
i=0
for cell in "${CELLS[@]}"; do
  read ds a s <<< "$cell"
  gpu=$((i % NUM_GPU))
  LOG=$LOGDIR/${ds}_K5_a${a}_s${s}.log
  echo "[GPU $gpu] $ds K=5 α=$a seed=$s -> $LOG"
  HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
    $PYTHON run_ablation_RIJ.py \
      --dataset $ds --loss_type R --alpha $a --n_clients 5 --seed $s --resume \
      > $LOG 2>&1 &
  i=$((i+1))
  if (( i % NUM_GPU == 0 )); then wait; fi
done
wait
echo "14233 ALL DONE"
