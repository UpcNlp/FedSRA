#!/bin/bash
# Train experts on top of EXISTING I-backbones, then eval I+Expert row.
# Used for the loss-ablation table: AL -> AL+Expert -> AL+Expert+CL=FULL.
#
# Usage:
#   bash dispatch_I_expert.sh probe   # 3 cells (α=0.05,K=5/10 and α=0.1,K=20)
#   bash dispatch_I_expert.sh full    # all 16 CIFAR-100 (α,K) cells
#   bash dispatch_I_expert.sh k50     # K=50 cells only (slow, run on big GPU)
#
# Prereq: I-backbones at saved_models/ablation_cifar100/I_a{α}_k{K}_s42/client_{k}/backbone.pt
set +u; source /opt/dtk/env.sh 2>/dev/null || true; set -u
cd /public/home/dongshou/fedETF/ETF-pesuade

PYTHON=/public/home/dongshou/anaconda/envs/ct/bin/python
MODE=${1:-probe}
SEED=42
LOGDIR=logs/I_expert_${MODE}
mkdir -p $LOGDIR

# Cell selection
case "$MODE" in
  probe)
    # 3 cells with largest I-J gap (best signal for whether Expert can compensate)
    CELLS=("0.05 5" "0.05 10" "0.1 20")
    GPUS=(0 1 2)
    ;;
  full)
    # All 16 CIFAR-100 cells; user must adjust GPUS to match host
    CELLS=(
      "0.05 5"  "0.05 10" "0.05 20"
      "0.1 5"   "0.1 10"  "0.1 20"
      "0.3 5"   "0.3 10"  "0.3 20"
      "0.5 5"   "0.5 10"  "0.5 20"
    )
    GPUS=(0 1 2 3 4 5 6 7)
    ;;
  k50)
    CELLS=("0.05 50" "0.1 50" "0.3 50" "0.5 50")
    GPUS=(0 1 2 3)
    ;;
  *) echo "Unknown mode: $MODE  (use probe|full|k50)"; exit 1 ;;
esac

NGPU=${#GPUS[@]}
echo "===== MODE=$MODE: ${#CELLS[@]} cells on ${NGPU} GPUs ====="

# Phase 1: train experts on I-backbones (resume-safe; skips if expert_*.pt exist)
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
  # Stagger so we don't slam the same GPU when cell count > NGPU
  if (( (i + 1) % NGPU == 0 )); then
    wait "${PIDS[@]}"; PIDS=()
  fi
done
wait "${PIDS[@]}" 2>/dev/null || true
echo "===== TRAINING DONE ====="

# Phase 2: eval (overwrites results/ablation_RIJ_eval_*.json with I+Expert row added)
PIDS=()
for i in "${!CELLS[@]}"; do
  read a K <<< "${CELLS[$i]}"
  gpu=${GPUS[$((i % NGPU))]}
  tag="c100_a${a}_k${K}"
  LOG=$LOGDIR/eval_${tag}.log
  echo "[GPU $gpu] eval $tag -> $LOG"
  HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
    $PYTHON -u eval_ablation_RIJ.py \
        --dataset cifar100 --alpha $a --n_clients $K --seed $SEED \
        > $LOG 2>&1 &
  PIDS+=($!)
  if (( (i + 1) % NGPU == 0 )); then
    wait "${PIDS[@]}"; PIDS=()
  fi
done
wait "${PIDS[@]}" 2>/dev/null || true
echo "===== EVAL DONE ====="

# Summary
echo; echo "===== I+Expert summary ====="
for cell in "${CELLS[@]}"; do
  read a K <<< "$cell"
  f="results/ablation_RIJ_eval_cifar100_a${a}_k${K}_s${SEED}.json"
  if [ -f "$f" ]; then
    $PYTHON -c "
import json
d = json.load(open('$f'))
r = d['rows']
def g(k): v = r.get(k,{}).get('acc'); return f'{v*100:5.2f}' if v is not None else ' ─── '
print(f'α=$a K=$K | I={g(\"I\")}  I+E={g(\"I+Expert\")}  J={g(\"J\")}  J+E={g(\"J+Expert\")}  Δ_I→I+E={(r[\"I+Expert\"][\"acc\"]-r[\"I\"][\"acc\"])*100 if r.get(\"I+Expert\",{}).get(\"acc\") else float(\"nan\"):+5.2f}')
"
  fi
done
