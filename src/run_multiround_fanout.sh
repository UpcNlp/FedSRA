#!/bin/bash
# R1-D4/R3-W7: multi-round fixed-ETF (FedETF-style) curve across 8 GPUs.
# CIFAR-10 & CIFAR-100, K=5, 10 rounds x 5 local epochs, eval at 1/3/5/10.
# Cheap (~50 local epochs total per cell). Resume-safe.
cd /public/home/dongshou/fedETF/ETF-pesuade
export LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/opencl/lib:$LD_LIBRARY_PATH
export HIP_PATH=/opt/dtk ROCM_PATH=/opt/dtk DTKROOT=/opt/dtk
PY=/public/home/dongshou/anaconda/envs/ct/bin/python
mkdir -p logs results

CELLS=(
  "cifar100 0.05 5" "cifar100 0.1 5" "cifar100 0.3 5" "cifar100 0.5 5"
  "cifar10 0.05 5"  "cifar10 0.1 5"  "cifar10 0.3 5"  "cifar10 0.5 5"
)
N=${#CELLS[@]}

run_cell () {
  local gpu=$1 ds=$2 a=$3 K=$4
  local out="results/multiround_${ds}_a${a}_k${K}_s42.json"
  [ -f "$out" ] && { echo "[g$gpu] skip $ds a$a K$K (resume)"; return; }
  HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
    $PY -u train_multiround_etf.py --dataset "$ds" --alpha "$a" --K "$K" \
        --rounds 10 --local_epochs 5 --eval_at 1,3,5,10 --out "$out" \
        > "logs/multiround_${ds}_a${a}_k${K}.log" 2>&1
  echo "[$(date +%H:%M:%S)][g$gpu] done $ds a$a K$K"
}

for gpu in 0 1 2 3 4 5 6 7; do
  ( i=$gpu; while [ $i -lt $N ]; do run_cell $gpu ${CELLS[$i]}; i=$((i+8)); done ) &
done
wait
echo "MULTIROUND COMPLETE $(date)"
