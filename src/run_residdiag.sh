#!/bin/bash
# A1 residual diagnostics fan-out across 8 GPUs. One (dataset,alpha,K) cell per task.
# Resume-safe: skips cells whose result JSON already exists. Usage: bash run_residdiag.sh
cd /public/home/dongshou/fedETF/ETF-pesuade
export LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/opencl/lib:$LD_LIBRARY_PATH
export HIP_PATH=/opt/dtk ROCM_PATH=/opt/dtk DTKROOT=/opt/dtk
PY=/public/home/dongshou/anaconda/envs/ct/bin/python
mkdir -p logs results

# cells: "dataset alpha K save_dir"
CELLS=(
  "cifar100 0.05 50 saved_models/cifar100_a0.05_k50_s42"
  "cifar100 0.1 50 saved_models/cifar100_a0.1_k50_s42"
  "cifar100 0.3 50 saved_models/cifar100_a0.3_k50_s42"
  "cifar100 0.5 50 saved_models/cifar100_a0.5_k50_s42"
  "cifar100 0.05 20 saved_models/cifar100_a0.05_k20_s42"
  "cifar100 0.1 20 saved_models/cifar100_a0.1_k20_s42"
  "cifar100 0.3 20 saved_models/cifar100_a0.3_k20_s42"
  "cifar100 0.5 20 saved_models/cifar100_a0.5_k20_s42"
  "cifar100 0.05 10 saved_models/cifar100_a0.05_k10_s42"
  "cifar100 0.1 10 saved_models/cifar100_a0.1_k10_s42"
  "cifar100 0.3 10 saved_models/cifar100_a0.3_k10_s42"
  "cifar100 0.5 10 saved_models/cifar100_a0.5_k10_s42"
  "cifar10 0.05 20 saved_models/a0.05_k20_s42"
  "cifar10 0.1 20 saved_models/a0.1_k20_s42"
  "cifar10 0.3 20 saved_models/a0.3_k20_s42"
  "cifar10 0.5 20 saved_models/a0.5_k20_s42"
  "cifar10 0.05 10 saved_models/a0.05_k10_s42"
  "cifar10 0.1 10 saved_models/a0.1_k10_s42"
  "cifar10 0.3 10 saved_models/a0.3_k10_s42"
  "cifar10 0.5 10 saved_models/a0.5_k10_s42"
)
N=${#CELLS[@]}

run_cell () {
  local gpu=$1 ds=$2 a=$3 K=$4 sd=$5
  local out="results/residdiag_${ds}_a${a}_k${K}_s42.json"
  [ -f "$out" ] && { echo "[g$gpu] skip $ds a$a K$K (resume)"; return; }
  [ -d "$sd" ] || { echo "[g$gpu] MISSING $sd -> skip"; return; }
  HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
    $PY -u eval_residual_diag.py --dataset "$ds" --alpha "$a" --K "$K" \
        --save_dir "$sd" --out "$out" > "logs/residdiag_${ds}_a${a}_k${K}.log" 2>&1
  echo "[$(date +%H:%M:%S)][g$gpu] done $ds a$a K$K"
}

for gpu in 0 1 2 3 4 5 6 7; do
  (
    i=$gpu
    while [ $i -lt $N ]; do
      run_cell $gpu ${CELLS[$i]}
      i=$((i + 8))
    done
  ) &
done
wait
echo "ALL DONE $(date)"
