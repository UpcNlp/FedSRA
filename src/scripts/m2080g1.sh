#!/bin/bash
# machine4.sh  →  K=5 + K=10, all alphas（共12个点）

ALPHAS=(0.05 0.1 0.2 0.3 0.5 1.0)
KS=(5 10)
SEED=42
GPU=1

mkdir -p results

for K in "${KS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
        out="results/grid_a${alpha}_k${K}_s${SEED}.json"
        if [ -f "$out" ]; then
            echo "SKIP: alpha=${alpha} K=${K}"
            continue
        fi
        echo "=============================="
        echo "RUN: alpha=${alpha}  K=${K}  GPU=${GPU}"
        echo "=============================="
        python run_grid.py \
            --alpha ${alpha} \
            --n_clients ${K} \
            --seed ${SEED} \
            --gpu ${GPU}
    done
done

echo "Machine 4 done."