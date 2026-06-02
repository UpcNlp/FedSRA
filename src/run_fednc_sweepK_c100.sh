#!/bin/bash
# Sweep NC measurement across K∈{5,20,50} × variants {J,I,R} × 4 α.
# 36 cells total (one missing on cluster: R K=50 a=0.1 — gracefully skipped).
# ~30 s/cell → ~18 min wall-clock on single GPU.
#
# Usage: bash run_fednc_sweepK_c100.sh <gpu_id>

set -e
GPU=${1:-0}

cd /public/home/dongshou/fedETF/ETF-pesuade

set +u; source /opt/dtk/env.sh 2>/dev/null || true; set -u

PY=/public/home/dongshou/anaconda/envs/ct/bin/python
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
    || { echo "[FATAL] torch import failed"; exit 1; }

mkdir -p logs results

DS=cifar100; SEED=42
for K in 5 20 50; do
  for variant in J I R; do
    for a in 0.05 0.1 0.3 0.5; do
      out="results/fednc_meta_${DS}_${variant}_a${a}_k${K}_s${SEED}.json"
      if [ -f "$out" ]; then
        echo "[skip] $out exists"; continue
      fi
      save_dir="saved_models/ablation_${DS}/${variant}_a${a}_k${K}_s${SEED}"
      if [ ! -d "$save_dir" ]; then
        echo "[skip-missing] $save_dir does not exist"; continue
      fi
      log="logs/fednc_${variant}_a${a}_k${K}.log"
      echo "[$(date +%H:%M:%S)] start K=$K variant=$variant a=$a -> $out"
      HIP_VISIBLE_DEVICES=$GPU CUDA_VISIBLE_DEVICES=$GPU \
        $PY -u measure_fednc.py \
            --dataset $DS --alpha $a --K $K --seed $SEED \
            --variant $variant --save_dir $save_dir --out $out \
            > $log 2>&1
      rc=$?
      echo "[$(date +%H:%M:%S)] done  K=$K variant=$variant a=$a rc=$rc"
      if [ $rc -ne 0 ]; then
        echo "[FATAL] cell failed; see $log"; tail -20 $log; exit $rc
      fi
    done
  done
done

echo "[$(date +%H:%M:%S)] ALL CELLS DONE"
ls results/fednc_meta_cifar100_*_k*_s42.json | wc -l
