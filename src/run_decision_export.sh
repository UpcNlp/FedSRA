#!/usr/bin/env bash
# Export dual-signal decision features across alpha (fixed K) for the crossover
# t-SNE grid. Inference-only; one GPU; cell-resume (skip if out exists).
set -u
cd "$(dirname "$0")"
mkdir -p tmp logs
set +u; source /opt/dtk/env.sh 2>/dev/null || true; set -u
PYTHON="${PYTHON:-/public/home/dongshou/anaconda/envs/ct/bin/python}"
GPU="${GPU:-0}"
K="${K:-10}"
SEED="${SEED:-42}"

"$PYTHON" -c "import torch; assert torch.cuda.is_available(); print('torch OK', torch.cuda.device_count())" \
  || { echo "FATAL: torch/ROCm unusable"; exit 1; }

for a in 0.05 0.1 0.3 0.5; do
  out="tmp/decision_a${a}_k${K}.npz"
  if [ -f "$out" ]; then echo "[skip] $out"; continue; fi
  echo "[$(date +%H:%M:%S)] export a=$a K=$K"
  CUDA_VISIBLE_DEVICES=$GPU HIP_VISIBLE_DEVICES=$GPU \
    "$PYTHON" -u export_decision_feats.py \
      --alpha $a --K $K --seed $SEED \
      --ckpt_dir saved_models/a${a}_k${K}_s${SEED} \
      --out "$out" > logs/decision_a${a}_k${K}.log 2>&1
  echo "[$(date +%H:%M:%S)] done a=$a -> $out"
done
echo "ALL DONE"
