#!/bin/bash
# Exp2b: class-sorted serving stream for the batch-zscore robustness study
# (inference-only). Re-runs eval_batch_zscore.py with --stream sorted on the
# same 12 cells as the random-stream batchz_* results:
#   cifar100 K=5  x 4 alphas  (saved_models/ablation_cifar100_a{a}_k5_s42)
#   cifar10  K=10/20 x 4 alphas (saved_models/a{a}_k{K}_s42)
# Usage: bash run_batchz_sorted.sh [GPU_ID]   (default GPU 0)
set -e
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/opencl/lib:$LD_LIBRARY_PATH
export HIP_PATH=/opt/dtk/hip ROCM_PATH=/opt/dtk DTKROOT=/opt/dtk
PY=/public/home/dongshou/anaconda/envs/ct/bin/python
GPU=${1:-0}
export HIP_VISIBLE_DEVICES=$GPU CUDA_VISIBLE_DEVICES=$GPU
mkdir -p logs results

$PY -c "import torch; assert torch.cuda.is_available(); print('torch OK:', torch.cuda.get_device_name(0))" || exit 1

for a in 0.05 0.1 0.3 0.5; do
  if [ -f "results/batchz_sorted_cifar100_a${a}_k5_s42.json" ]; then
    echo "skip c100 a=$a K=5 (resume)"
  else
    $PY -u eval_batch_zscore.py --dataset cifar100 --NL 100 --alpha "$a" --K 5 \
        --save_dir "saved_models/ablation_cifar100_a${a}_k5_s42" --stream sorted \
        > "logs/batchz_sorted_c100_a${a}_k5.log" 2>&1
    echo "[$(date +%H:%M:%S)] done c100 a=$a K=5"
  fi
done

for K in 10 20; do
  for a in 0.05 0.1 0.3 0.5; do
    if [ -f "results/batchz_sorted_cifar10_a${a}_k${K}_s42.json" ]; then
      echo "skip c10 a=$a K=$K (resume)"
    else
      $PY -u eval_batch_zscore.py --dataset cifar10 --NL 10 --alpha "$a" --K "$K" \
          --save_dir "saved_models/a${a}_k${K}_s42" --stream sorted \
          > "logs/batchz_sorted_c10_a${a}_k${K}.log" 2>&1
      echo "[$(date +%H:%M:%S)] done c10 a=$a K=$K"
    fi
  done
done

echo "ALL DONE: $(ls results/batchz_sorted_*.json 2>/dev/null | wc -l)/12 jsons"
