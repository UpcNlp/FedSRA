#!/bin/bash
cd /public/home/dongshou/fedETF/ETF-pesuade
export LD_LIBRARY_PATH="/opt/dtk/cuda/cuda-12/lib64:/usr/local/lib/python3.10/dist-packages/torch/lib:/opt/ucx/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/mpi/lib:/opt/hwloc/lib:"
export HIP_PATH=/opt/dtk ROCM_PATH=/opt/dtk
PY=/public/home/dongshou/anaconda/envs/ct/bin/python
mkdir -p tmp logs/misalign
run_col () {  # $1=gpu  $2=alpha
  local g=$1 a=$2
  for K in 5 10 20 50; do
    out=tmp/misalign_woetf_cifar100_a${a}_k${K}.npz
    [ -f "$out" ] && { echo "[gpu$g] skip a${a}_k${K} (exists)"; continue; }
    log=logs/misalign/a${a}_k${K}.log
    HIP_VISIBLE_DEVICES=$g CUDA_VISIBLE_DEVICES=$g $PY -u misalign_dump.py \
      --alpha $a --K $K --dataset cifar100 --NL 100 --max_imgs 4000 \
      --save_dir saved_models/ablation_woetf_cifar100/a${a}_k${K}_s42 --out $out > $log 2>&1
    echo "[gpu$g] a${a}_k${K} -> $(tail -1 $log | grep -o 'misalign%=.*')"
  done
}
run_col 2 0.1 & run_col 3 0.3 & run_col 4 0.5 &
wait
echo "REMAINING CELLS DONE"
