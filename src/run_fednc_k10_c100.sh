#!/bin/bash
# Cluster launcher for fednc_ablation measurement.
# Loops 3 variants × 4 α on CIFAR-100, K=10, seed=42.
# Single GPU is enough (only inference); ~1-2 min per cell, ~25 min total.
#
# Usage: bash run_fednc_k10_c100.sh <gpu_id>

set -e
GPU=${1:-0}

cd /public/home/dongshou/fedETF/ETF-pesuade

# ROCm env (cluster ssh non-interactive shells lack it — see reference_cluster.md)
set +u; source /opt/dtk/env.sh 2>/dev/null || true; set -u

# Sanity probe — fail loudly if torch can't import
PY=/public/home/dongshou/anaconda/envs/ct/bin/python
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
    || { echo "[FATAL] torch import failed"; exit 1; }

mkdir -p logs results

DS=cifar100; K=10; SEED=42
for variant in J I R; do
  for a in 0.05 0.1 0.3 0.5; do
    out="results/fednc_meta_${DS}_${variant}_a${a}_k${K}_s${SEED}.json"
    if [ -f "$out" ]; then
      echo "[skip] $out exists"
      continue
    fi
    save_dir="saved_models/ablation_${DS}/${variant}_a${a}_k${K}_s${SEED}"
    log="logs/fednc_${variant}_a${a}.log"
    echo "[$(date +%H:%M:%S)] start variant=$variant a=$a -> $out"
    HIP_VISIBLE_DEVICES=$GPU CUDA_VISIBLE_DEVICES=$GPU \
      $PY -u measure_fednc.py \
          --dataset $DS --alpha $a --K $K --seed $SEED \
          --variant $variant --save_dir $save_dir --out $out \
          > $log 2>&1
    rc=$?
    echo "[$(date +%H:%M:%S)] done  variant=$variant a=$a rc=$rc"
    if [ $rc -ne 0 ]; then
      echo "[FATAL] cell failed; see $log"
      tail -20 $log
      exit $rc
    fi
  done
done

echo "[$(date +%H:%M:%S)] ALL CELLS DONE"
ls -la results/fednc_meta_*.json | tail -12
