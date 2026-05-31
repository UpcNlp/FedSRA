#!/usr/bin/env bash
# Loss-weight sensitivity sweep (lambda_al, tau) — AMD ROCm / NVIDIA, multi-GPU.
#
# Priority: all lambda_al cells first, then tau cells. The main-table config
# (lambda_al=0.5, tau=0.1) is skipped (reuse main result).
#
# A named-pipe semaphore caps concurrency at the number of GPUs given in $GPUS
# and maps each job to a free GPU. cell-level resume: a cell whose result json
# already exists is skipped, so re-launching after an interruption is safe.
#
# Single node with 16 GPUs:
#     GPUS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15" bash run_loss_sweep_16gpu.sh
# Split across nodes (run one per node; jobs are sharded by NSHARD/SHARD):
#     node A:  GPUS="0 1 2 3 4 5 6 7" NSHARD=2 SHARD=0 bash run_loss_sweep_16gpu.sh
#     node B:  GPUS="0 1 2 3 4 5 6 7" NSHARD=2 SHARD=1 bash run_loss_sweep_16gpu.sh
set -u
cd "$(dirname "$0")"
mkdir -p results logs/loss_sweep

SEED="${SEED:-42}"
K="${K:-5}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15}"
NSHARD="${NSHARD:-1}"      # number of nodes sharing the job list
SHARD="${SHARD:-0}"        # this node's shard index (0-based)
read -r -a GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}

# --- build job list:  "lambda_al tau alpha" ---
ALL_JOBS=()
# (1) lambda_al sweep  (tau fixed at default 0.1; skip lal=0.5)
for a in 0.05 0.1 0.3 0.5; do
  for lal in 0 0.1 0.25 1.0 2.0; do
    ALL_JOBS+=("$lal 0.1 $a")
  done
done
# (2) tau sweep  (lambda_al fixed at default 0.5; skip tau=0.1)
for a in 0.05 0.1 0.3 0.5; do
  for tau in 0.05 0.2 0.5; do
    ALL_JOBS+=("0.5 $tau $a")
  done
done

# --- keep only this node's shard ---
JOBS=()
for i in "${!ALL_JOBS[@]}"; do
  if [ $(( i % NSHARD )) -eq "$SHARD" ]; then JOBS+=("${ALL_JOBS[$i]}"); fi
done

echo "Node shard $SHARD/$NSHARD | GPUs: $GPUS | cells this node: ${#JOBS[@]} | K=$K seed=$SEED"

# --- GPU semaphore via fd 9 (preloaded with this node's GPU ids) ---
fifo=$(mktemp -u); mkfifo "$fifo"; exec 9<>"$fifo"; rm "$fifo"
for g in "${GPU_ARR[@]}"; do echo "$g" >&9; done

for job in "${JOBS[@]}"; do
  read -r lal tau a <<< "$job"
  read -r -u 9 g                       # blocks until a GPU frees up
  tag="lal${lal}_tau${tau}_a${a}_k${K}_s${SEED}"
  out="results/loss_sweep_lal${lal}_tau${tau}_a${a}_k${K}_s${SEED}.json"
  if [ -f "$out" ]; then
    echo "[skip] $tag (exists)"; echo "$g" >&9; continue
  fi
  echo "[$(date +%H:%M:%S)] [launch GPU$g] $tag"
  (
    CUDA_VISIBLE_DEVICES=$g HIP_VISIBLE_DEVICES=$g \
      python -u run_loss_sweep.py \
        --alpha "$a" --lambda_al "$lal" --tau "$tau" \
        --n_clients "$K" --seed "$SEED" \
        > "logs/loss_sweep/${tag}.log" 2>&1
    echo "[$(date +%H:%M:%S)] [done GPU$g rc=$?] $tag"
    echo "$g" >&9                       # return GPU to the pool
  ) &
done
wait
echo "==== shard $SHARD DONE ===="
