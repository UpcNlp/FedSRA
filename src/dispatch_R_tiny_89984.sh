#!/bin/bash
set +u; source /opt/dtk/env.sh 2>/dev/null || true; set -u
set -e

cd /public/home/dongshou/fedETF/ETF-pesuade
PYTHON=/public/home/dongshou/anaconda/envs/ct/bin/python
LOGDIR=./logs/r_train_seeds
mkdir -p $LOGDIR

# 12 Tiny K=5 R cells: seeds {0, 42, 123} × α {0.05, 0.1, 0.3, 0.5}
CELLS=(
  "tinyimagenet 0.05 0"   "tinyimagenet 0.1 0"   "tinyimagenet 0.3 0"   "tinyimagenet 0.5 0"
  "tinyimagenet 0.05 42"  "tinyimagenet 0.1 42"  "tinyimagenet 0.3 42"  "tinyimagenet 0.5 42"
  "tinyimagenet 0.05 123" "tinyimagenet 0.1 123" "tinyimagenet 0.3 123" "tinyimagenet 0.5 123"
)
NUM_GPU=4
i=0
for cell in "${CELLS[@]}"; do
  read ds a s <<< "$cell"
  gpu=$((i % NUM_GPU))
  LOG=$LOGDIR/${ds}_K5_a${a}_s${s}.log
  echo "[GPU $gpu] $ds K=5 α=$a seed=$s -> $LOG"
  HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
    $PYTHON run_ablation_RIJ_tiny.py \
      --dataset $ds --loss_type R --alpha $a --n_clients 5 --seed $s --resume \
      > $LOG 2>&1 &
  i=$((i+1))
  if (( i % NUM_GPU == 0 )); then wait; fi
done
wait
echo "89984 Tiny ALL DONE"
