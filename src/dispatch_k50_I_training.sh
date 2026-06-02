#!/bin/bash
set +u; source /opt/dtk/env.sh 2>/dev/null || true; set -u
cd /public/home/dongshou/fedETF/ETF-pesuade

PYTHON=/public/home/dongshou/anaconda/envs/ct/bin/python
LOGDIR=logs/k50_I_training
mkdir -p $LOGDIR

# 3 cells in parallel,one per GPU (use idle GPUs 1, 2, 3 on 14233)
# GPU 0 is busy (80%+ HCU)
ALPHAS=(0.1 0.3 0.5)
GPUS=(1 2 3)

for i in 0 1 2; do
  a=${ALPHAS[$i]}
  gpu=${GPUS[$i]}
  LOG=$LOGDIR/c100_K50_I_a${a}.log
  echo "[GPU $gpu] cifar100 K=50 loss=I α=$a -> $LOG"
  HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
    $PYTHON run_ablation_RIJ.py \
      --dataset cifar100 \
      --loss_type I \
      --alpha $a \
      --n_clients 50 \
      --seed 42 \
      --skip_experts \
      > $LOG 2>&1 &
done
wait
echo "===== K=50 I TRAINING DONE ====="

# Auto-trigger inference for these 3 cells after training
echo "Auto-dispatching RIJ eval for the 3 newly-trained cells..."
for a in ${ALPHAS[@]}; do
  HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 \
    $PYTHON eval_ablation_RIJ.py \
      --dataset cifar100 --alpha $a --n_clients 50 --seed 42 \
      > $LOGDIR/eval_K50_I_a${a}.log 2>&1 &
done
wait
echo "===== K=50 I EVAL DONE ====="
