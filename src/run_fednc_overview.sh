#!/bin/bash
# Cluster pipeline for fednc_overview figure:
#   1. train centralized ETF-anchored ResNet18 on full CIFAR-10 (~1-2 h)
#   2. dump features for centralized + federated preL2 + federated GPA
#
# Usage: bash run_fednc_overview.sh <gpu_id>

set -e
GPU=${1:-0}

cd /public/home/dongshou/fedETF/ETF-pesuade

set +u; source /opt/dtk/env.sh 2>/dev/null || true; set -u

PY=/public/home/dongshou/anaconda/envs/ct/bin/python
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
    || { echo "[FATAL] torch import failed"; exit 1; }

mkdir -p logs results saved_models/centralized_cifar10_s42

# --- 1. centralized training (skip if backbone already saved) -----------
CENT_BB=saved_models/centralized_cifar10_s42/client_0/backbone.pt
if [ -f "$CENT_BB" ]; then
    echo "[skip] centralized backbone exists: $CENT_BB"
else
    echo "[$(date +%H:%M:%S)] start centralized training"
    HIP_VISIBLE_DEVICES=$GPU CUDA_VISIBLE_DEVICES=$GPU \
        $PY -u train_centralized_cifar10.py --seed 42 \
        > logs/centralized_cifar10_train.log 2>&1
    rc=$?
    echo "[$(date +%H:%M:%S)] centralized training done rc=$rc"
    if [ $rc -ne 0 ]; then
        echo "[FATAL] training failed; see logs/centralized_cifar10_train.log"
        tail -20 logs/centralized_cifar10_train.log
        exit $rc
    fi
fi

# --- 2. dump features for the figure ------------------------------------
# We only need ONE α (the most heterogeneous, where the federated-naive
# vs FedDSI contrast is sharpest). α=0.05 default.
ALPHA=${ALPHA:-0.05}
echo "[$(date +%H:%M:%S)] start dump (α=$ALPHA, K=10, seed=42)"
HIP_VISIBLE_DEVICES=$GPU CUDA_VISIBLE_DEVICES=$GPU \
    $PY -u dump_fednc_overview.py --alpha $ALPHA --K 10 --seed 42 \
    > logs/fednc_overview_dump_a${ALPHA}.log 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] dump done rc=$rc"
if [ $rc -ne 0 ]; then
    echo "[FATAL] dump failed"
    tail -20 logs/fednc_overview_dump_a${ALPHA}.log
    exit $rc
fi

echo "[$(date +%H:%M:%S)] ALL DONE"
ls -la results/fednc_overview_*_a${ALPHA}_*.npz
