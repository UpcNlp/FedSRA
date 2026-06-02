#!/bin/bash
set +u; source /opt/dtk/env.sh 2>/dev/null || true; set -u
PYTHON=/public/home/dongshou/anaconda/envs/ct/bin/python
SCRIPT=/public/home/dongshou/fedETF/Co-Boosting-PP-master/eval_noetf_geom.py
LOGDIR=/public/home/dongshou/fedETF/ETF-pesuade/logs/ce_gpa_c100k5
mkdir -p $LOGDIR

# CIFAR-100 K=5 × 4 α, Le=300 (matching CE training)
ALPHAS=(0.05 0.1 0.3 0.5)
for i in 0 1 2 3; do
  a=${ALPHAS[$i]}
  gpu=$((i + 4))  # use GPUs 4-7 since 0-3 might be busy
  LOG=$LOGDIR/c100_K5_a${a}.log
  echo "[GPU $gpu] cifar100 K=5 α=$a -> $LOG"
  HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
    $PYTHON $SCRIPT --dataset cifar100 --alpha $a --le 300 --n_clients 5 --seed 42 \
    > $LOG 2>&1 &
done
wait
echo "DONE"
