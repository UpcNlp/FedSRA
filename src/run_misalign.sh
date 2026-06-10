#!/bin/bash
cd /public/home/dongshou/fedETF/ETF-pesuade
export LD_LIBRARY_PATH="/opt/dtk/cuda/cuda-12/lib64:/usr/local/lib/python3.10/dist-packages/torch/lib:/opt/ucx/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/mpi/lib:/opt/hwloc/lib:"
export HIP_PATH=/opt/dtk ROCM_PATH=/opt/dtk
PY=/public/home/dongshou/anaconda/envs/ct/bin/python
mkdir -p tmp logs/misalign
ALPHAS=(0.05 0.1 0.3 0.5); KS=(5 10 20 50)
# K-major order so round-robin spreads the heavy K=50 cells 1-per-GPU
cells=(); for K in "${KS[@]}"; do for a in "${ALPHAS[@]}"; do cells+=("$a:$K"); done; done
run_gpu () { local g=$1; shift
  for cell in "$@"; do
    a=${cell%:*}; K=${cell#*:}
    sd=saved_models/ablation_woetf_cifar100/a${a}_k${K}_s42
    out=tmp/misalign_woetf_cifar100_a${a}_k${K}.npz
    log=logs/misalign/a${a}_k${K}.log
    HIP_VISIBLE_DEVICES=$g CUDA_VISIBLE_DEVICES=$g $PY -u misalign_dump.py \
      --alpha $a --K $K --dataset cifar100 --NL 100 --max_imgs 4000 \
      --save_dir $sd --out $out > $log 2>&1
    echo "[gpu$g] a${a}_k${K} -> $(tail -1 $log | grep -o 'misalign%=.*')"
  done
}
q0=(); q1=(); q2=(); q3=(); i=0
for cell in "${cells[@]}"; do case $((i%4)) in 0)q0+=($cell);;1)q1+=($cell);;2)q2+=($cell);;3)q3+=($cell);; esac; i=$((i+1)); done
run_gpu 0 "${q0[@]}" & run_gpu 1 "${q1[@]}" & run_gpu 2 "${q2[@]}" & run_gpu 3 "${q3[@]}" &
wait
echo "ALL 16 CELLS DONE"
