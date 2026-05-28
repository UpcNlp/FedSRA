#!/bin/bash
# Launch 4 α in parallel on GPU 0/2/4/5 (free GPUs from rocm-smi).
# Each runs joint single-stage training to 600 epochs with shared expert.
#
# Usage:  bash run_epoch_joint_parallel.sh
# Logs:   logs/joint_a*.log

set -u
cd "$(dirname "$0")"
mkdir -p logs

PY=${PY:-/public/home/dongshou/anaconda/envs/ct/bin/python}
SEED=42
NC=10
MAX_EP=700
COSINE_END=700                                    # cosine 一路衰减到 700 (clean schedule)
SNAPS="1,2,3,5,7,10,15,20,30,50,75,100,150,200,300,400,500,600,650,700"

run() {
    local gpu=$1; local alpha=$2
    echo "[GPU$gpu] α=$alpha → logs/joint_a${alpha}_k${NC}.log"
    HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
        $PY -u run_epoch_analysis_joint.py \
            --alpha $alpha --seed $SEED --n_clients $NC \
            --max_ep $MAX_EP --cosine_end_ep $COSINE_END --snapshots "$SNAPS" \
            > logs/joint_a${alpha}_k${NC}.log 2>&1 &
}

run 0 0.05
run 2 0.1
run 4 0.3
run 5 0.5

echo "Launched 4 jobs, waiting..."
wait
echo "All done. Results in results/epoch_analysis_joint_a*_k${NC}_s${SEED}.json"
