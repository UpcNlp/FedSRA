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
# Split across nodes by contiguous slice (run one per node; disjoint -> no race):
#     node 14233: GPUS="0 1 2 3 4 5 6 7" JSTART=0  JCOUNT=8 bash run_loss_sweep_16gpu.sh
#     node 89984: GPUS="0 1 2 3"         JSTART=8  JCOUNT=4 bash run_loss_sweep_16gpu.sh
#     node 37703: GPUS="0 1 2 3"         JSTART=12 JCOUNT=4 bash run_loss_sweep_16gpu.sh
set -u
cd "$(dirname "$0")"
mkdir -p results logs/loss_sweep

# --- ROCm/DTK env (SSH non-interactive shells lack it -> torch ImportError).
#     /opt/dtk/env.sh references an unbound var, so wrap against set -u. ---
set +u; source /opt/dtk/env.sh 2>/dev/null || true; set -u

PYTHON="${PYTHON:-/public/home/dongshou/anaconda/envs/ct/bin/python}"

# --- sanity probe: fail loudly instead of silently writing 0 results ---
"$PYTHON" -c "import torch; assert torch.cuda.is_available(); print('torch OK, GPUs:', torch.cuda.device_count())" \
  || { echo "FATAL: torch/ROCm not usable in this shell — aborting."; exit 1; }

SEED="${SEED:-42}"
K="${K:-5}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15}"
read -r -a GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}

# --- build job list:  "lambda_al tau alpha" ---
# Only the lambda_al sweep (the alignment-vs-discriminative weight, the single
# explicit balancing coefficient in Eq. bb). tau is a temperature, kept fixed
# at its default 0.1 and reported as a constant in the paper. lambda_al=0.5 is
# the main-table config -> skipped (reuse main result). lambda_al=0 (alignment
# term removed) is a component ablation already covered by the ablation study,
# so it is excluded here; every swept point keeps both loss terms active.
ALL_JOBS=()
for a in 0.05 0.1 0.3 0.5; do
  for lal in 0.1 0.25 1.0 2.0; do
    ALL_JOBS+=("$lal 0.1 $a")
  done
done

# --- keep only this node's contiguous slice (disjoint across nodes -> no race) ---
JSTART="${JSTART:-0}"
JCOUNT="${JCOUNT:-${#ALL_JOBS[@]}}"
JOBS=("${ALL_JOBS[@]:$JSTART:$JCOUNT}")

echo "Slice [$JSTART,+$JCOUNT) | GPUs: $GPUS | cells this node: ${#JOBS[@]} | K=$K seed=$SEED"

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
      "$PYTHON" -u run_loss_sweep.py \
        --alpha "$a" --lambda_al "$lal" --tau "$tau" \
        --n_clients "$K" --seed "$SEED" \
        > "logs/loss_sweep/${tag}.log" 2>&1
    echo "[$(date +%H:%M:%S)] [done GPU$g rc=$?] $tag"
    echo "$g" >&9                       # return GPU to the pool
  ) &
done
wait
echo "==== slice [$JSTART,+$JCOUNT) DONE ===="
