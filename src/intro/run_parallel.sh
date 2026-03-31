#!/bin/bash
set -e

# ===================== 配置 =====================
ALPHAS=(0.05 0.1 0.3 0.5 1.0)
TOTAL_TASKS=12               # 总并发：2张卡 × 6个任务 = 12
PYTHON_SCRIPT="intro/run_single_alpha.py"
LOG_DIR="logs"

mkdir -p outputs $LOG_DIR
rm -f outputs/result_*.json

# 循环生成 12 个任务（自动循环使用 alpha 列表）
for (( task_id=0; task_id<TOTAL_TASKS; task_id++ )); do
    # 循环取 alpha（5个alpha循环分配到12个任务）
    alpha_idx=$((task_id % ${#ALPHAS[@]}))
    alpha=${ALPHAS[$alpha_idx]}

    # ========== 核心：强制分配 GPU ==========
    if [ $task_id -lt 6 ]; then
        GPU=0
    else
        GPU=1
    fi

    LOG_FILE="$LOG_DIR/task_${task_id}_gpu${GPU}_alpha_${alpha}.log"

    echo "🚀 启动任务 $task_id | GPU $GPU | alpha $alpha | 日志 → $LOG_FILE"

    # 后台运行：独立GPU + 独立日志
    (
        export CUDA_VISIBLE_DEVICES=$GPU
        python "$PYTHON_SCRIPT" "$alpha" "$task_id" > "$LOG_FILE" 2>&1
        echo "✅ 完成任务 $task_id | GPU $GPU | alpha $alpha"
    ) &

    # 控制并发不超过 12
    while [ $(jobs -r | wc -l) -ge 12 ]; do
        sleep 0.1
    done
done

wait
echo -e "\n🎉 所有 12 个任务全部完成！"
python intro/plot_aggregate.py