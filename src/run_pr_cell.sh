#!/bin/bash
cd /public/home/dongshou/fedETF/ETF-pesuade
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip
export PATH=/opt/dtk/bin:/opt/dtk/hip/bin:/opt/conda/bin:$PATH
export LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/opencl/lib:${LD_LIBRARY_PATH:-}
PY=/public/home/dongshou/anaconda/envs/ct/bin/python
DS=$1; K=$2; GPU=$3; a=$4
mkdir -p logs
tag="${DS}_K${K}_a${a}_PR"
echo "[$(date +%T)] start $tag"
HIP_VISIBLE_DEVICES=$GPU CUDA_VISIBLE_DEVICES=$GPU $PY -u run_ablation_RIJ.py \
    --dataset $DS --loss_type PR --alpha $a --n_clients $K --seed 42 --resume \
    > logs/ablation_RIJ_${tag}.log 2>&1
echo "[$(date +%T)] done $tag"
