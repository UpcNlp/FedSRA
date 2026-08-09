#!/usr/bin/env bash
set -u

# Waits without disturbing an existing job. Two consecutive low-utilization/low-memory
# snapshots are required before the experiment master starts.
# Usage: bash wait_for_idle_and_run.sh <master command...>

POLL_SECONDS=${POLL_SECONDS:-60}
MAX_UTIL=${MAX_UTIL:-5}
MAX_MEM_MB=${MAX_MEM_MB:-2048}
LOG_FILE=${IDLE_LOG:-idle_guard.log}
CONSECUTIVE=0

while true; do
  LINE=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | head -1)
  UTIL=$(echo "$LINE" | cut -d, -f1 | tr -d ' ')
  MEM=$(echo "$LINE" | cut -d, -f2 | tr -d ' ')
  NOW=$(date '+%F %T')
  if [[ "$UTIL" =~ ^[0-9]+$ && "$MEM" =~ ^[0-9]+$ && $UTIL -le $MAX_UTIL && $MEM -le $MAX_MEM_MB ]]; then
    CONSECUTIVE=$((CONSECUTIVE + 1))
  else
    CONSECUTIVE=0
  fi
  echo "[$NOW] util=${UTIL:-NA}% mem=${MEM:-NA}MB idle_streak=$CONSECUTIVE" | tee -a "$LOG_FILE"
  if [[ $CONSECUTIVE -ge 2 ]]; then
    break
  fi
  sleep "$POLL_SECONDS"
done

echo "[$(date '+%F %T')] GPU confirmed idle; launching: $*" | tee -a "$LOG_FILE"
exec "$@"
