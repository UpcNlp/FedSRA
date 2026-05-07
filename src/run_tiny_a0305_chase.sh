#!/bin/bash
# OURS Tiny-ImageNet on chase GPU 1
# alpha = {0.3, 0.5}, seed = {0, 42, 123}, K=5

export CUDA_VISIBLE_DEVICES=1
mkdir -p logs

for a in 0.3 0.5; do
  for s in 0 42 123; do
    if [ -f "results/znorm_tiny_a${a}_k5_s${s}.json" ]; then
      echo "=== $(date) SKIP a=$a s=$s (exists) ==="
      continue
    fi
    echo "=== $(date) tiny a=$a s=$s ==="
    python -u run_znorm_tinyimagenet.py \
      --alpha $a --n_clients 5 --seed $s --no_save 2>&1 | tee -a logs/tiny_a${a}_s${s}.log
  done
done

echo "=== ALL DONE $(date) ==="
