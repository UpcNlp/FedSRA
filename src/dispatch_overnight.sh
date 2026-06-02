#!/bin/bash
# Overnight bin-packed I+Expert dispatch.
# All 16 CIFAR-100 cells across 5 GPUs (89984×4 + 50527×1).
# Critical path: ~7h. Each GPU runs serial worker on its assigned cell list.
#
# Usage on cluster:
#   bash dispatch_overnight.sh 89984    # launches 4 GPU workers (0-3)
#   bash dispatch_overnight.sh 50527    # launches 1 GPU worker (0)
#
set +u; source /opt/dtk/env.sh 2>/dev/null || true; set -u
cd /public/home/dongshou/fedETF/ETF-pesuade

PYTHON=/public/home/dongshou/anaconda/envs/ct/bin/python
NODE=${1:-89984}
SEED=42
LOGDIR=logs/overnight_${NODE}
mkdir -p $LOGDIR

ALPHA_F_LIST="0.5 1.0 1.5 2.0 3.0"

# ---- Per-GPU cell assignment (bin-packed greedy) -----------------------
case "$NODE" in
  89984)
    declare -A ASSIGN
    ASSIGN[0]="0.05:50 0.5:10 0.5:5"
    ASSIGN[1]="0.1:50  0.3:10 0.3:5"
    ASSIGN[2]="0.3:50  0.5:20"
    ASSIGN[3]="0.5:50  0.1:20 0.05:10 0.05:5"  # last 3 are skip-via-resume
    GPUS=(0 1 2 3)
    ;;
  50527)
    declare -A ASSIGN
    ASSIGN[0]="0.05:20 0.3:20 0.1:10 0.1:5"
    GPUS=(0)
    ;;
  *) echo "Unknown node: $NODE"; exit 1 ;;
esac

# ---- Worker function: serially train + α_f sweep through assigned cells
worker() {
  local gpu=$1
  local cells="${ASSIGN[$gpu]}"
  local mylog=$LOGDIR/worker_gpu${gpu}.log
  echo "[$(date +%F_%T)] [GPU $gpu @ $NODE] cells: $cells" > $mylog
  for pair in $cells; do
    a=${pair%%:*}
    K=${pair##*:}
    tag="c100_a${a}_k${K}"
    echo "[$(date +%F_%T)] === START $tag on GPU $gpu ===" >> $mylog
    # Train (or resume) I+Expert
    HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
      $PYTHON -u run_ablation_RIJ.py \
          --dataset cifar100 --loss_type I \
          --alpha $a --n_clients $K --seed $SEED \
          --resume --train_experts \
          >> $LOGDIR/train_${tag}.log 2>&1
    rc=$?
    echo "[$(date +%F_%T)]   train done rc=$rc" >> $mylog
    # α_f sweep
    swplog=$LOGDIR/sweep_${tag}.log
    : > $swplog
    for af in $ALPHA_F_LIST; do
      echo "--- α_f=$af ---" >> $swplog
      HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
        $PYTHON -u eval_ablation_RIJ.py \
            --dataset cifar100 --alpha $a --n_clients $K --seed $SEED \
            --alpha_fusion $af 2>&1 \
          | grep -E "row [RIJ] |row I\+|row J\+" >> $swplog
    done
    echo "[$(date +%F_%T)]   sweep done -> $swplog" >> $mylog
  done
  echo "[$(date +%F_%T)] [GPU $gpu] ALL DONE" >> $mylog
}

# Launch workers in parallel, one per GPU
PIDS=()
for gpu in "${GPUS[@]}"; do
  worker $gpu &
  PIDS+=($!)
done
echo "Launched ${#PIDS[@]} workers on $NODE: ${PIDS[*]}"
wait "${PIDS[@]}" 2>/dev/null
echo "===== $NODE ALL WORKERS DONE ====="
