#!/bin/bash
# Confidence-ordered early-exit cascade (inference-only) for the serving-cost study.
# Sweeps CIFAR-100 over K in {10,20,50} x alpha in {0.05,0.1,0.3,0.5} = 12 cells,
# weight-ordered cascade. Each cell extracts features once then re-aggregates.
# Resume-aware via the per-cell JSON. Usage: bash run_early_exit.sh [GPU_ID]
set -e
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/opencl/lib:$LD_LIBRARY_PATH
export HIP_PATH=/opt/dtk/hip ROCM_PATH=/opt/dtk DTKROOT=/opt/dtk
PY=/public/home/dongshou/anaconda/envs/ct/bin/python
GPU=${1:-0}
export HIP_VISIBLE_DEVICES=$GPU CUDA_VISIBLE_DEVICES=$GPU
mkdir -p logs results

$PY -c "import torch; assert torch.cuda.is_available(); print('torch OK:', torch.cuda.get_device_name(0))" || exit 1

for K in 10 20 50; do
  for a in 0.05 0.1 0.3 0.5; do
    out="results/earlyexit_cifar100_a${a}_k${K}_weight_s42.json"
    if [ -f "$out" ]; then
      echo "skip c100 a=$a K=$K (resume)"
    else
      $PY -u eval_early_exit.py --dataset cifar100 --NL 100 --alpha "$a" --K "$K" \
          --save_dir "saved_models/cifar100_a${a}_k${K}_s42" --order weight \
          > "logs/earlyexit_c100_a${a}_k${K}.log" 2>&1
      echo "[$(date +%H:%M:%S)] done c100 a=$a K=$K"
    fi
  done
done

echo "ALL DONE: $(ls results/earlyexit_cifar100_*_weight_s42.json 2>/dev/null | wc -l)/12 jsons"
