#!/bin/bash
# Launch 4 α in parallel on GPU 0/2/4/5 (free GPUs from rocm-smi).
# Each runs joint single-stage training to 600 epochs with shared expert.
#
# Usage:  bash run_epoch_joint_parallel.sh
# Logs:   logs/joint_a*.log

set -u
cd "$(dirname "$0")"
mkdir -p logs

SEED=42
NC=10
MAX_EP=600
SNAPS="10,20,30,50,75,100,150,200,250,300,400,500,600"

run() {
    local gpu=$1; local alpha=$2
    echo "[GPU$gpu] α=$alpha → logs/joint_a${alpha}_k${NC}.log"
    HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
        python -u run_epoch_analysis_joint.py \
            --alpha $alpha --seed $SEED --n_clients $NC \
            --max_ep $MAX_EP --snapshots "$SNAPS" \
            > logs/joint_a${alpha}_k${NC}.log 2>&1 &
}

run 0 0.05
run 2 0.1
run 4 0.3
run 5 0.5

echo "Launched 4 jobs, waiting..."
wait
echo "All done. Results in results/epoch_analysis_joint_a*_k${NC}_s${SEED}.json"
