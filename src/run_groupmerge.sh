#!/bin/bash
# Focused grouped-merge G-sweep at thr=0.95 (minimal merge -> max fidelity).
# Knob is G = number of merged models (= forward passes / per-model stats).
# Usage: bash run_groupmerge.sh <GPU> <K> "<G_list>"
set -e
cd /public/home/dongshou/fedETF/ETF-pesuade
export LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/opencl/lib:$LD_LIBRARY_PATH
export HIP_PATH=/opt/dtk/hip ROCM_PATH=/opt/dtk DTKROOT=/opt/dtk
GPU=$1; K=$2; GLIST=$3
export HIP_VISIBLE_DEVICES=$GPU CUDA_VISIBLE_DEVICES=$GPU
PY=/public/home/dongshou/anaconda/envs/ct/bin/python
mkdir -p logs results
for a in 0.05 0.5; do
  $PY -u eval_grouped_merge.py --dataset cifar100 --NL 100 --alpha "$a" --K "$K" \
      --save_dir "saved_models/cifar100_a${a}_k${K}_s42" --groups "$GLIST" --thrs 0.95 \
      --out "results/groupmerge_cifar100_a${a}_k${K}_s42.json" \
      > "logs/gm_k${K}_a${a}.log" 2>&1
  echo "[$(date +%H:%M:%S)] done K=$K a=$a"
done
echo "GM_DONE K=$K"
