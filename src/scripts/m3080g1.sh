#!/bin/bash
# machine2.sh  →  K=50, all alphas

ALPHAS=(0.05 0.1 0.2 0.3 0.5 1.0)
K=50
SEED=42
GPU=1

mkdir -p results

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

echo "Machine 2 done."