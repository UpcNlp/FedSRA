#!/bin/bash
# B1: extra-seed FedSRA (J) training for statistical reporting (R2-W4/D3).
# CIFAR-100, J variant, alpha in {0.05,0.1,0.3,0.5}, K in {5,10,20}, seeds {0,123}.
# Backbone-only (--skip_experts) -- the main "Ours" row is J backbone + RGA.
# Resume-safe. Existing s42 already covers the third seed.
cd /public/home/dongshou/fedETF/ETF-pesuade
export LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/opencl/lib:$LD_LIBRARY_PATH
export HIP_PATH=/opt/dtk ROCM_PATH=/opt/dtk DTKROOT=/opt/dtk
PY=/public/home/dongshou/anaconda/envs/ct/bin/python
mkdir -p logs

# cells: "alpha K seed"  (K ascending = fast first; seed 0 before 123)
CELLS=()
for seed in 0 123; do
  for K in 5 10 20; do
    for a in 0.05 0.1 0.3 0.5; do
      CELLS+=("$a $K $seed")
    done
  done
done
N=${#CELLS[@]}

run_cell () {
  local gpu=$1 a=$2 K=$3 seed=$4
  local done_marker="saved_models/ablation_cifar100/J_a${a}_k${K}_s${seed}/client_0/backbone.pt"
  [ -f "$done_marker" ] && { echo "[g$gpu] skip a$a K$K s$seed (resume)"; return; }
  HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
    $PY -u run_ablation_RIJ.py --dataset cifar100 --loss_type J --alpha "$a" \
        --n_clients "$K" --seed "$seed" --skip_experts --resume \
        > "logs/bseed_a${a}_k${K}_s${seed}.log" 2>&1
  echo "[$(date +%H:%M:%S)][g$gpu] done a$a K$K s$seed"
}

for gpu in 0 1 2 3 4 5 6 7; do
  ( i=$gpu; while [ $i -lt $N ]; do run_cell $gpu ${CELLS[$i]}; i=$((i+8)); done ) &
done
wait
echo "BSEEDS COMPLETE $(date)"
