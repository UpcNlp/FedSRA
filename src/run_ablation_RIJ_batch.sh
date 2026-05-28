#!/bin/bash
# Cluster batch launcher for R/I/J ablation training.
#
# Layout: 4 GPUs parallel. For each (dataset, K) cell, we run 4 α × 3 loss_type
#   = 12 jobs in 3 rounds of 4 parallel each.
#
# Cell priority order (fastest first, slowest last):
#   K=5  → K=10 → K=20 → K=50
# Datasets interleaved so partial completion still gives 2-dataset coverage.
#
# Usage:
#   bash run_ablation_RIJ_batch.sh             # run all 8 cells (full ablation)
#   CELLS="cifar10_5 cifar100_5" bash ...      # run only specified cells
#   ALPHAS="0.05 0.3" bash ...                 # subset of alphas
#
# All training is --resume aware: re-running skips completed backbones/experts.
# Total expected wallclock on 4-GPU: ~10-13 days for full ablation (K=50 dominant).

set -u
cd "$(dirname "$0")"

PY=${PY:-/public/home/dongshou/anaconda/envs/ct/bin/python}
SEED=${SEED:-42}
ALPHAS=${ALPHAS:-"0.05 0.1 0.3 0.5"}
LOSSES=${LOSSES:-"R I J"}
CELLS=${CELLS:-"cifar10_5 cifar100_5 cifar10_10 cifar100_10 cifar10_20 cifar100_20 cifar10_50 cifar100_50"}

mkdir -p logs

run_one() {
    # $1=gpu  $2=dataset  $3=K  $4=alpha  $5=loss
    local gpu=$1 ds=$2 K=$3 a=$4 loss=$5
    local tag="${ds}_K${K}_a${a}_${loss}"
    local logf="logs/ablation_RIJ_${tag}.log"
    HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
        $PY -u run_ablation_RIJ.py \
            --dataset $ds --loss_type $loss \
            --alpha $a --n_clients $K --seed $SEED \
            --resume \
            > "$logf" 2>&1 &
}

for CELL in $CELLS; do
    DS=${CELL%_*}
    K=${CELL##*_}
    echo "═══════ CELL: dataset=$DS  K=$K ═══════"
    date +"  start: %Y-%m-%d %H:%M:%S"

    # Build job list: 4 alphas × 3 losses = 12 jobs
    declare -a JOBS=()
    for a in $ALPHAS; do
        for loss in $LOSSES; do
            JOBS+=("$a:$loss")
        done
    done

    # Dispatch in rounds of 4 parallel
    i=0
    while [ $i -lt ${#JOBS[@]} ]; do
        echo "  Round at job $i: launching up to 4 in parallel"
        for gpu in 0 1 2 3; do
            if [ $i -ge ${#JOBS[@]} ]; then break; fi
            spec=${JOBS[$i]}
            a=${spec%:*}
            loss=${spec##*:}
            echo "    GPU$gpu: $DS K=$K α=$a loss=$loss"
            run_one $gpu $DS $K $a $loss
            i=$((i+1))
        done
        wait
        echo "  Round done at $(date +%H:%M:%S)"
    done

    echo "  CELL $DS K=$K finished at $(date +%H:%M:%S)"
    # Optional: commit + push intermediate results for each completed cell
    # cd ..
    # git add -A && git commit -m "ablation_RIJ: cell $DS K=$K done" && git push
    # cd ETF-pesuade
done

echo ""
echo "═══════ All cells finished ═══════"
date +"%Y-%m-%d %H:%M:%S"
