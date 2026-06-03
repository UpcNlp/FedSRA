#!/bin/bash
# Train + eval the "w/o ETF" ablation (learnable W_k) across all CIFAR-100 cells.
# Bin-packed across GPUs of one node.
#
# Usage on cluster:
#   bash dispatch_woetf.sh 89984    # 4 GPU workers on 89984 (cells 0-11)
#   bash dispatch_woetf.sh 50527    # 1 GPU worker  on 50527 (cells 12-15)
#
set +u; source /opt/dtk/env.sh 2>/dev/null || true; set -u
cd /public/home/dongshou/fedETF/ETF-pesuade

PYTHON=/public/home/dongshou/anaconda/envs/ct/bin/python
NODE=${1:-89984}
SEED=42
LOGDIR=logs/woetf_${NODE}
mkdir -p $LOGDIR

# Rebalanced: each 89984 GPU does K=50+K=20 of same α (~7h); 50527 does all K=10+K=5 (~6h)
# Critical path 7h (was 8.5h with previous assignment)
case "$NODE" in
  89984)
    declare -A ASSIGN
    ASSIGN[0]="0.05:50 0.05:20"
    ASSIGN[1]="0.1:50  0.1:20"
    ASSIGN[2]="0.3:50  0.3:20"
    ASSIGN[3]="0.5:50  0.5:20"
    GPUS=(0 1 2 3)
    ;;
  50527)
    declare -A ASSIGN
    ASSIGN[0]="0.05:10 0.1:10 0.3:10 0.5:10 0.05:5 0.1:5 0.3:5 0.5:5"
    GPUS=(0)
    ;;
  *) echo "Unknown node: $NODE"; exit 1 ;;
esac

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

    # Train (resume-safe)
    HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
      $PYTHON -u train_bb_woetf.py \
          --dataset cifar100 --alpha $a --n_clients $K --seed $SEED --resume \
          >> $LOGDIR/train_${tag}.log 2>&1
    echo "[$(date +%F_%T)]   train done rc=$?" >> $mylog

    # Eval
    HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
      $PYTHON -u eval_bb_woetf.py \
          --dataset cifar100 --alpha $a --n_clients $K --seed $SEED \
          >> $LOGDIR/eval_${tag}.log 2>&1
    echo "[$(date +%F_%T)]   eval done rc=$?" >> $mylog
  done
  echo "[$(date +%F_%T)] [GPU $gpu] ALL DONE" >> $mylog
}

PIDS=()
for gpu in "${GPUS[@]}"; do
  worker $gpu &
  PIDS+=($!)
done
echo "Launched ${#PIDS[@]} workers on $NODE: ${PIDS[*]}"
wait "${PIDS[@]}" 2>/dev/null
echo "===== $NODE ALL WORKERS DONE ====="
