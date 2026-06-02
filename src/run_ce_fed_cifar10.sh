#!/bin/bash
# Launch 10 CE-federated CIFAR-10 clients in parallel (one per GPU).
# K=10 α=0.05 seed=42. Each client ResNet18 + Linear(256,10) head, CE loss,
# 600 epochs. ~1-2 h per client on K100_AI; runs concurrently on 10 GPUs.
#
# Usage:
#   bash run_ce_fed_cifar10.sh           # all 10 clients, GPUs 0..7+0..1 (or first 8 + duplicate)
#
# Adapt to whatever GPUs are free.

set -e
cd /public/home/dongshou/fedETF/ETF-pesuade

set +u; source /opt/dtk/env.sh 2>/dev/null || true; set -u
PY=/public/home/dongshou/anaconda/envs/ct/bin/python
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 1

mkdir -p logs saved_models

ALPHA=0.05; K=10; SEED=42

# 10 clients, mapped to 8 GPUs (clients 8 and 9 share GPUs 0 and 1)
declare -a GPU_MAP=(0 1 2 3 4 5 6 7 0 1)

for i in 0 1 2 3 4 5 6 7 8 9; do
  gpu=${GPU_MAP[$i]}
  out=saved_models/ce_cifar10_a${ALPHA}_k${K}_s${SEED}/client_${i}/backbone.pt
  if [ -f "$out" ]; then
    echo "[skip] client $i exists"
    continue
  fi
  log=logs/ce_fed_client_${i}.log
  echo "[$(date +%H:%M:%S)] launch client=$i on GPU $gpu -> $out"
  HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
    nohup $PY -u train_federated_ce_cifar10.py \
        --client_id $i --alpha $ALPHA --K $K --seed $SEED \
        > $log 2>&1 &
  # tiny stagger so each client logs the data summary cleanly
  sleep 2
done

wait
echo "[$(date +%H:%M:%S)] ALL CLIENTS DONE"
ls saved_models/ce_cifar10_a${ALPHA}_k${K}_s${SEED}/*/backbone.pt | wc -l
