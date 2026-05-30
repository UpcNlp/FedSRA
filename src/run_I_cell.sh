#!/bin/bash
set -u
cd /public/home/dongshou/fedETF/ETF-pesuade
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip DTKROOT=/opt/dtk
export PATH=/opt/dtk/bin:/opt/dtk/llvm/bin:/opt/dtk/hip/bin:/opt/conda/bin:$PATH
export LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/opencl/lib:${LD_LIBRARY_PATH:-}
PY=/public/home/dongshou/anaconda/envs/ct/bin/python
DS=$1; K=$2; GPU=$3; ALPHA=$4; SEED=42
mkdir -p logs
tag="${DS}_K${K}_a${ALPHA}_I"
echo "[$(date +%F_%T)] start $tag GPU=$GPU"
HIP_VISIBLE_DEVICES=$GPU CUDA_VISIBLE_DEVICES=$GPU \
  $PY -u run_ablation_RIJ.py --dataset $DS --loss_type I \
      --alpha $ALPHA --n_clients $K --seed $SEED --resume --skip_experts \
  > logs/ablation_RIJ_${tag}.log 2>&1
echo "[$(date +%F_%T)] done $tag (rc=$?)"
