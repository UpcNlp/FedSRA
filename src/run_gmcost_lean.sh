#!/bin/bash
# A2 (lean): grouped-merge cost sweep for the TRACTABLE case K=10 (small merged
# models -> fast, clean latency timing). Full thr x G. CIFAR-100, 4 alpha.
# Large-K (20/50) merges are impractically slow (union builds ~5000-channel models,
# ~1000s per combo) and are reported from params+acc only via the existing
# groupmerge_*.json; that impracticality is itself a finding for R2-D2.
cd /public/home/dongshou/fedETF/ETF-pesuade
export LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/opencl/lib:$LD_LIBRARY_PATH
export HIP_PATH=/opt/dtk ROCM_PATH=/opt/dtk DTKROOT=/opt/dtk
PY=/public/home/dongshou/anaconda/envs/ct/bin/python
mkdir -p logs results
THRS=0.5,0.7,0.85,0.95

CELLS=(
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
echo "GMCOST LEAN COMPLETE $(date)"
