#!/bin/bash
# A2: grouped-merge FULL cost sweep across 8 GPUs. CIFAR-100, K in {10,20,50} x
# alpha in {0.05,0.1,0.3,0.5}, thr in {0.5,0.7,0.85,0.95}. Resume-safe.
cd /public/home/dongshou/fedETF/ETF-pesuade
export LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/opencl/lib:$LD_LIBRARY_PATH
export HIP_PATH=/opt/dtk ROCM_PATH=/opt/dtk DTKROOT=/opt/dtk
PY=/public/home/dongshou/anaconda/envs/ct/bin/python
mkdir -p logs results
THRS=0.5,0.7,0.85,0.95

# cells: "alpha K groups"
CELLS=(
  "0.05 50 5,10,25,50" "0.1 50 5,10,25,50" "0.3 50 5,10,25,50" "0.5 50 5,10,25,50"
  "0.05 20 2,4,5,10,20" "0.1 20 2,4,5,10,20" "0.3 20 2,4,5,10,20" "0.5 20 2,4,5,10,20"
  "0.05 10 1,2,5,10" "0.1 10 1,2,5,10" "0.3 10 1,2,5,10" "0.5 10 1,2,5,10"
)
N=${#CELLS[@]}

run_cell () {
  local gpu=$1 a=$2 K=$3 G=$4
  local out="results/groupmergecost_cifar100_a${a}_k${K}_s42.json"
  [ -f "$out" ] && { echo "[g$gpu] skip a$a K$K (resume)"; return; }
  local sd="saved_models/cifar100_a${a}_k${K}_s42"
  [ -d "$sd" ] || { echo "[g$gpu] MISSING $sd"; return; }
  HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
    $PY -u eval_grouped_merge_cost.py --dataset cifar100 --NL 100 --alpha "$a" --K "$K" \
        --save_dir "$sd" --groups "$G" --thrs "$THRS" --out "$out" \
        > "logs/gmcost_a${a}_k${K}.log" 2>&1
  echo "[$(date +%H:%M:%S)][g$gpu] done a$a K$K"
}

for gpu in 0 1 2 3 4 5 6 7; do
  ( i=$gpu; while [ $i -lt $N ]; do run_cell $gpu ${CELLS[$i]}; i=$((i+8)); done ) &
done
wait
echo "GMCOST SWEEP COMPLETE $(date)"
