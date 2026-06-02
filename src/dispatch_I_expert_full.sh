#!/bin/bash
# Full I+Expert ablation: train experts on existing I-backbones for all
# K∈{5,10,20} cells, then per-cell α_f sweep to find I+Expert optimum.
#
# Usage on cluster:
#   bash dispatch_I_expert_full.sh phase1   # train+sweep K∈{5,10,20}
#   bash dispatch_I_expert_full.sh phase2   # train+sweep K=50 (slower)
#
set +u; source /opt/dtk/env.sh 2>/dev/null || true; set -u
cd /public/home/dongshou/fedETF/ETF-pesuade

PYTHON=/public/home/dongshou/anaconda/envs/ct/bin/python
PHASE=${1:-phase1}
SEED=42
LOGDIR=logs/I_expert_${PHASE}
mkdir -p $LOGDIR

# Cell selection
case "$PHASE" in
  phase1)  # K∈{5,10,20}; already-trained cells (probe) will skip via --resume
    CELLS=(
      "0.05 5" "0.1 5"  "0.3 5"  "0.5 5"
      "0.05 10" "0.1 10" "0.3 10" "0.5 10"
      "0.05 20" "0.1 20" "0.3 20" "0.5 20"
    )
    GPUS=(0 1 2 3)
    ALPHA_F_LIST="0.5 1.0 1.5 2.0 3.0"
    ;;
  phase2)
    CELLS=("0.05 50" "0.1 50" "0.3 50" "0.5 50")
    GPUS=(0 1 2 3)
    ALPHA_F_LIST="0.5 1.0 1.5 2.0 3.0"
    ;;
  *) echo "Unknown phase: $PHASE  (use phase1|phase2)"; exit 1 ;;
esac

NGPU=${#GPUS[@]}
echo "===== PHASE=$PHASE: ${#CELLS[@]} cells on ${NGPU} GPUs ====="

# Phase A: train I-experts in batches (resume-safe)
PIDS=()
for i in "${!CELLS[@]}"; do
  read a K <<< "${CELLS[$i]}"
  gpu=${GPUS[$((i % NGPU))]}
  tag="c100_a${a}_k${K}_I_exp"
  LOG=$LOGDIR/train_${tag}.log
  echo "[GPU $gpu] train I+Expert α=$a K=$K -> $LOG"
  HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
    $PYTHON -u run_ablation_RIJ.py \
        --dataset cifar100 --loss_type I \
        --alpha $a --n_clients $K --seed $SEED \
        --resume --train_experts \
        > $LOG 2>&1 &
  PIDS+=($!)
  # Wait between batches when we fill the GPUs
  if (( (i + 1) % NGPU == 0 )); then
    wait "${PIDS[@]}"; PIDS=()
    echo "  batch done at i=$((i+1))/${#CELLS[@]}"
  fi
done
wait "${PIDS[@]}" 2>/dev/null || true
echo "===== TRAINING DONE ====="

# Phase B: α_f sweep per cell (serial, ~3 min each)
SUMMARY=$LOGDIR/summary_alpha_f.txt
> $SUMMARY
for i in "${!CELLS[@]}"; do
  read a K <<< "${CELLS[$i]}"
  gpu=${GPUS[$((i % NGPU))]}
  tag="c100_a${a}_k${K}"
  LOG=$LOGDIR/sweep_${tag}.log
  echo "[GPU $gpu] sweep α_f for $tag -> $LOG"
  : > $LOG
  for af in $ALPHA_F_LIST; do
    echo "--- α_f=$af ---" >> $LOG
    HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
      $PYTHON -u eval_ablation_RIJ.py \
          --dataset cifar100 --alpha $a --n_clients $K --seed $SEED \
          --alpha_fusion $af 2>&1 \
        | grep -E "row [RIJ] |row I\+|row J\+" >> $LOG
  done

  # Extract best I+Expert across α_f
  best=$($PYTHON -c "
import re
log = open('$LOG').read()
rows = re.findall(r'--- α_f=([\d.]+) ---\s*\n(.*?)(?=--- α_f|\Z)', log, re.DOTALL)
best = (None, -1.0)
i_acc = None; r_acc = None; j_acc = None
for af, blk in rows:
    m = re.search(r'row I\+Expert:\s*([\d.]+)%', blk); ie = float(m.group(1)) if m else None
    if i_acc is None:
        m = re.search(r'row I\s+:\s*([\d.]+)%', blk); i_acc = float(m.group(1)) if m else None
    if r_acc is None:
        m = re.search(r'row R\s+:\s*([\d.]+)%', blk); r_acc = float(m.group(1)) if m else None
    if j_acc is None:
        m = re.search(r'row J\s+:\s*([\d.]+)%', blk); j_acc = float(m.group(1)) if m else None
    if ie is not None and ie > best[1]:
        best = (af, ie)
print(f'α=$a K=$K | I={i_acc:.2f} R={r_acc:.2f} J={j_acc:.2f} I+E*={best[1]:.2f}@α_f={best[0]} Δ={(best[1]-i_acc):+.2f}')
")
  echo "$best" | tee -a $SUMMARY
done
echo "===== SWEEP DONE ====="
echo; echo "===== SUMMARY ($SUMMARY) ====="
cat $SUMMARY
