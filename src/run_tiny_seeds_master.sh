#!/bin/bash
# Master loop for ONE GPU: run a list of (alpha:seed) OURS tiny K=5 cells serially.
# Skip-if-result-exists makes it resume-safe. Reuses run_tiny_moreseeds_cell.sh
# (ROCm env + torch sanity probe + saving ENABLED).
# Usage: bash run_tiny_seeds_master.sh <GPU> <alpha:seed> [<alpha:seed> ...]
set -u
GPU=$1; shift
cd "$(dirname "$0")"
mkdir -p logs
MLOG=logs/master_tiny_gpu${GPU}.log

echo "=== $(date) master start gpu$GPU cells: $* ===" | tee -a "$MLOG"
for spec in "$@"; do
  ALPHA=${spec%%:*}; SEED=${spec##*:}
  RJSON="results/znorm_tiny_a${ALPHA}_k5_s${SEED}.json"
  if [ -f "$RJSON" ]; then
    echo "[skip] a=$ALPHA s=$SEED (result exists)" | tee -a "$MLOG"
    continue
  fi
  echo "[$(date +%H:%M:%S)] gpu$GPU -> a=$ALPHA s=$SEED" | tee -a "$MLOG"
  bash run_tiny_moreseeds_cell.sh "$GPU" "$ALPHA" "$SEED"
  echo "[$(date +%H:%M:%S)] gpu$GPU done a=$ALPHA s=$SEED (result: $([ -f "$RJSON" ] && echo OK || echo MISSING))" | tee -a "$MLOG"
done
echo "=== $(date) master gpu$GPU ALL DONE ===" | tee -a "$MLOG"
