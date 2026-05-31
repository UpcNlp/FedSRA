#!/bin/bash
# Single-cell launcher: OURS (znorm_sqrt) Tiny-ImageNet, K=5, one (alpha, seed) on one GPU.
# Saving is ENABLED (no --no_save) so weights persist for future re-eval/visualization.
# Usage: bash run_tiny_moreseeds_cell.sh <GPU> <ALPHA> <SEED>
set -u
GPU=$1; ALPHA=$2; SEED=$3
cd "$(dirname "$0")"

# ROCm/DTK env for SSH-launched job (set +u wrap: env.sh trips set -u; see reference_cluster memory)
set +u; source /opt/dtk/env.sh 2>/dev/null || true; set -u

PY=/public/home/dongshou/anaconda/envs/ct/bin/python
mkdir -p logs
LOG=logs/tiny_moreseed_a${ALPHA}_k5_s${SEED}.log

# sanity: torch+GPU must work, else the run would silently produce nothing
$PY -c "import torch; assert torch.cuda.is_available()" || { echo "FATAL torch/GPU unavailable" | tee -a "$LOG"; exit 1; }

echo "=== $(date) START a=$ALPHA s=$SEED on GPU $GPU ($(hostname)) ===" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=$GPU HIP_VISIBLE_DEVICES=$GPU \
  $PY -u run_znorm_tinyimagenet.py --alpha $ALPHA --n_clients 5 --seed $SEED \
  >> "$LOG" 2>&1
RC=$?  # capture before $(date) clobbers $?
echo "=== $(date) DONE a=$ALPHA s=$SEED rc=$RC ===" | tee -a "$LOG"
