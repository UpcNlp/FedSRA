#!/bin/bash
# Fan out the grouped-merge G-sweep across 8 GPUs: one (K, alpha, G) cell per task,
# one serial worker per GPU consuming every-8th cell (modulo spreads heavy cells).
# Each cell writes its own per-G result file. thr fixed at 0.95 (minimal merge).
# Usage: bash run_gm_fanout.sh
cd /public/home/dongshou/fedETF/ETF-pesuade
export LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/opencl/lib:$LD_LIBRARY_PATH
export HIP_PATH=/opt/dtk/hip ROCM_PATH=/opt/dtk DTKROOT=/opt/dtk
PY=/public/home/dongshou/anaconda/envs/ct/bin/python
mkdir -p logs results

# cells: "K ALPHA G"  (alpha=0.05 first = priority; heavy m=K/G first to spread)
CELLS=(
  "50 0.05 5"  "20 0.05 2"  "50 0.05 10" "10 0.05 2"  "20 0.05 5"  "50 0.05 25" "20 0.05 10" "10 0.05 5"
  "50 0.05 50" "20 0.05 20" "10 0.05 10"
  "50 0.5 5"   "20 0.5 2"   "50 0.5 10"  "10 0.5 2"   "20 0.5 5"   "50 0.5 25"  "20 0.5 10"  "10 0.5 5"
  "50 0.5 50"  "20 0.5 20"  "10 0.5 10"
)
N=${#CELLS[@]}

run_cell () {
  local gpu=$1 K=$2 a=$3 G=$4
  local out="results/groupmerge_cifar100_a${a}_k${K}_G${G}_s42.json"
  [ -f "$out" ] && { echo "[g$gpu] skip K=$K a=$a G=$G (resume)"; return; }
  HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
    $PY -u eval_grouped_merge.py --dataset cifar100 --NL 100 --alpha "$a" --K "$K" \
        --save_dir "saved_models/cifar100_a${a}_k${K}_s42" --groups "$G" --thrs 0.95 \
        --out "$out" > "logs/gm_k${K}_a${a}_G${G}.log" 2>&1
  echo "[$(date +%H:%M:%S)][g$gpu] done K=$K a=$a G=$G"
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
echo "FANOUT DONE: $(ls results/groupmerge_cifar100_*_G*_s42.json 2>/dev/null | wc -l) cells"
