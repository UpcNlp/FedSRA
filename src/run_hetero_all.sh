#!/bin/bash
# ============================================================
# 模型异构 16-cell 全量 launcher (只跑 Ours)
#   2 tier (mild/strong) x 2 dataset (cifar10/cifar100) x 4 alpha (0.05/0.1/0.3/0.5)
#   K=5 固定, seed=42 单 seed.
# GPU 列表通过环境变量传入, 默认 0..5; 每个 GPU 串行跑分到的 cell (--resume 安全).
# 全部完成后 git commit + push results.
#   用法: GPUS="0 1 2 3 4 5" nohup bash run_hetero_all.sh > logs/hetero_master.log 2>&1 &
# ============================================================
set -u
cd /public/home/dongshou/fedETF/ETF-pesuade
set +u; source /opt/dtk/env.sh 2>/dev/null || true; set -u   # 海光 DCU 运行时; env.sh 引用未定义 LD_LIBRARY_PATH, 需临时关 set -u
git config --global --add safe.directory /public/home/dongshou/fedETF 2>/dev/null || true
PY=/public/home/dongshou/anaconda/envs/ct/bin/python
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs

read -r -a GPU_ARR <<< "${GPUS:-0 1 2 3 4 5}"
NG=${#GPU_ARR[@]}
echo "===== hetero 16-cell launcher ($(date)) ====="
echo "GPUs: ${GPU_ARR[*]} (NG=$NG)"

# 16 个 cell: "tier dataset alpha"  (cifar100 较重, 与 cifar10 交错以均衡每条链)
CELLS=(
  "strong cifar10 0.05"  "mild cifar10 0.05"  "strong cifar100 0.05" "mild cifar100 0.05"
  "strong cifar10 0.1"   "mild cifar10 0.1"   "strong cifar100 0.1"  "mild cifar100 0.1"
  "strong cifar10 0.3"   "mild cifar10 0.3"   "strong cifar100 0.3"  "mild cifar100 0.3"
  "strong cifar10 0.5"   "mild cifar10 0.5"   "strong cifar100 0.5"  "mild cifar100 0.5"
)

run_cell() {
    local gpu=$1 tier=$2 ds=$3 a=$4
    local log="logs/hetero_${ds}_${tier}_a${a}_gpu${gpu}_${TS}.log"
    echo "[$(date +%H:%M:%S)] GPU$gpu START $tier $ds a$a -> $log"
    CUDA_VISIBLE_DEVICES=$gpu HIP_VISIBLE_DEVICES=$gpu $PY -u run_hetero_arch.py \
        --dataset "$ds" --tier "$tier" --alpha "$a" --seed 42 --resume > "$log" 2>&1
    echo "[$(date +%H:%M:%S)] GPU$gpu DONE  $tier $ds a$a rc=$?"
}

# round-robin 把 16 个 cell 分配到各 GPU; 每个 GPU 一个后台子 shell 串行执行其 cell
launch_chain() {
    local gpu=$1; shift
    for cell in "$@"; do
        set -- $cell
        run_cell "$gpu" "$1" "$2" "$3"
    done
    echo "[$(date +%H:%M:%S)] GPU$gpu CHAIN COMPLETE"
}

for gi in "${!GPU_ARR[@]}"; do
    gpu=${GPU_ARR[$gi]}
    chain=()
    for ci in "${!CELLS[@]}"; do
        if [ $((ci % NG)) -eq "$gi" ]; then
            chain+=("${CELLS[$ci]}")
        fi
    done
    echo "GPU$gpu chain: ${chain[*]}"
    launch_chain "$gpu" "${chain[@]}" &
done

wait
echo "===== ALL 16 CELLS DONE ($(date)) ====="
echo "completed result files:"
ls -1 results/hetero_*_s42.json 2>/dev/null | grep -v smoke

# commit + push results (in-flight 安全: 先 add 全部再 pull 之前已在外层完成)
git add -A
git commit -m "hetero-arch: 16-cell results (2 tier x 2 ds x 4 alpha, K=5, s42)" || true
git push && echo "pushed OK" || echo "git push FAILED (results committed locally on cluster)"
echo "===== launcher finished ====="
