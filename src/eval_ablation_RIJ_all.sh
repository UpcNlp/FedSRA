#!/bin/bash
# Evaluate all available R/I/J ablation cells.
# Each cell takes a few minutes (forward + matrix ops); 32 cells ≈ 1-2 hours serial.
#
# Usage:
#   bash eval_ablation_RIJ_all.sh
#   DATASETS="cifar100" bash eval_ablation_RIJ_all.sh    # subset

set -u
cd "$(dirname "$0")"

PY=${PY:-/public/home/dongshou/anaconda/envs/ct/bin/python}
SEED=${SEED:-42}
DATASETS=${DATASETS:-"cifar10 cifar100"}
ALPHAS=${ALPHAS:-"0.05 0.1 0.3 0.5"}
KS=${KS:-"5 10 20 50"}
ALPHA_F=${ALPHA_F:-0.3}
MIN_N=${MIN_N:-10}

mkdir -p logs

for ds in $DATASETS; do
    for K in $KS; do
        for a in $ALPHAS; do
            echo "─── eval $ds α=$a K=$K ───"
            HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
                $PY -u eval_ablation_RIJ.py \
                    --dataset $ds --alpha $a --n_clients $K --seed $SEED \
                    --alpha_fusion $ALPHA_F --min_n $MIN_N \
                    2>&1 | tee logs/eval_RIJ_${ds}_K${K}_a${a}.log
        done
    done
done

echo ""
echo "═══ All eval done ═══"
date +"%Y-%m-%d %H:%M:%S"
