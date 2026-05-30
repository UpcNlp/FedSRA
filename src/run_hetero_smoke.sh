#!/bin/bash
# 冒烟测试: 4 个组合 (mild/strong x cifar10/cifar100) 各跑 2 epoch, 单 GPU.
# 验证端到端管线 (异构训练 -> 保存 -> znorm 聚合 -> expert 融合 -> json).
# 用法: bash run_hetero_smoke.sh [GPU_ID]
set -e
cd /public/home/dongshou/fedETF/ETF-pesuade
source /opt/dtk/env.sh 2>/dev/null || true   # 海光 DCU 运行时 (galaxyhip/LD_LIBRARY_PATH)
PY=/public/home/dongshou/anaconda/envs/ct/bin/python
G=${1:-0}
mkdir -p logs

for combo in "strong cifar10" "mild cifar10" "strong cifar100" "mild cifar100"; do
    set -- $combo
    echo ""
    echo "================= SMOKE: tier=$1 dataset=$2 ================="
    CUDA_VISIBLE_DEVICES=$G HIP_VISIBLE_DEVICES=$G $PY -u run_hetero_arch.py \
        --dataset "$2" --tier "$1" --alpha 0.1 --seed 42 --smoke
done

echo ""
echo "================= SMOKE ALL PASSED ================="
ls -la results/hetero_*_smoke.json
