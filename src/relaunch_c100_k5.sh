#!/bin/bash
cd /public/home/dongshou/fedETF/ETF-pesuade
PY=/public/home/dongshou/anaconda/envs/ct/bin/python
SEED=42; GPU=1
for a in 0.05 0.1 0.3 0.5; do
  for loss in R I; do
    tag="cifar100_K5_a${a}_${loss}"
    echo "[$(date +%H:%M:%S)] start $tag"
    HIP_VISIBLE_DEVICES=$GPU CUDA_VISIBLE_DEVICES=$GPU \
      $PY -u run_ablation_RIJ.py --dataset cifar100 --loss_type $loss \
          --alpha $a --n_clients 5 --seed $SEED --resume \
      > logs/ablation_RIJ_${tag}.log 2>&1
    echo "[$(date +%H:%M:%S)] done $tag"
  done
done
echo "[$(date +%H:%M:%S)] C100 K5 R+I DONE"
