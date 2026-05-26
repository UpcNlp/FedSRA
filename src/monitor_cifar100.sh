#!/bin/bash
# CIFAR-100 OURS scalability 监控脚本
# 用法: bash monitor_cifar100.sh
# 显示: 12 个 cell 当前状态 (done/running/queued/failed) + 已完成的 acc_dynamic

cd /public/home/dongshou/fedETF/ETF-pesuade

echo "================================================================"
echo "  CIFAR-100 OURS scalability 状态 ($(date))"
echo "================================================================"
echo ""

# 12 个目标 cell: (alpha, K)
CELLS=(
    "0.05:10" "0.05:20" "0.05:50"
    "0.1:10"  "0.1:20"  "0.1:50"
    "0.3:10"  "0.3:20"  "0.3:50"
    "0.5:10"  "0.5:20"  "0.5:50"
)

printf "%-12s %-8s %-12s %-12s %s\n" "α" "K" "状态" "acc_dynamic" "备注"
echo "----------------------------------------------------------------"

DONE=0
RUNNING=0
QUEUED=0
FAILED=0

for entry in "${CELLS[@]}"; do
    IFS=':' read -r alpha k <<< "$entry"
    json="results/znorm_cifar100_a${alpha}_k${k}_s42.json"
    log_pattern="logs/c100_a${alpha}_k${k}_s42_gpu*.log"

    if [ -f "$json" ]; then
        # 完成
        acc=$(python3 -c "import json; d=json.load(open('$json')); print(f\"{d.get('acc_dynamic', 0)*100:.2f}%\")" 2>/dev/null)
        af=$(python3 -c "import json; d=json.load(open('$json')); print(f\"αf={d.get('af_dynamic', 0):.3f}\")" 2>/dev/null)
        printf "%-12s %-8s %-12s %-12s %s\n" "$alpha" "$k" "✓ DONE" "$acc" "$af"
        DONE=$((DONE+1))
    else
        # 看 log 状态
        log=$(ls -t $log_pattern 2>/dev/null | head -1)
        if [ -z "$log" ]; then
            printf "%-12s %-8s %-12s %-12s %s\n" "$alpha" "$k" "  QUEUED" "-" "未启动"
            QUEUED=$((QUEUED+1))
        else
            # 检查日志最近 5 分钟有没有更新 (活跃)
            mtime=$(stat -c %Y "$log")
            now=$(date +%s)
            age=$((now - mtime))

            last=$(tail -1 "$log" 2>/dev/null)
            if echo "$last" | grep -q "Traceback\|Error\|OOM"; then
                err=$(grep -E "Error|OOM" "$log" | tail -1 | cut -c-50)
                printf "%-12s %-8s %-12s %-12s %s\n" "$alpha" "$k" "✗ FAILED" "-" "$err"
                FAILED=$((FAILED+1))
            elif [ $age -gt 600 ]; then
                printf "%-12s %-8s %-12s %-12s %s\n" "$alpha" "$k" "? STALE" "-" "日志 ${age}s 没更新"
                FAILED=$((FAILED+1))
            else
                # running, 提取最新进度
                prog=$(echo "$last" | grep -oE "(BB|Expert) [^,]+|eval [0-9]+/[0-9]+" | head -1)
                [ -z "$prog" ] && prog=$(echo "$last" | cut -c-40)
                printf "%-12s %-8s %-12s %-12s %s\n" "$alpha" "$k" "⏳ RUNNING" "-" "$prog"
                RUNNING=$((RUNNING+1))
            fi
        fi
    fi
done

echo "----------------------------------------------------------------"
echo "汇总: DONE=$DONE / RUNNING=$RUNNING / QUEUED=$QUEUED / FAILED=$FAILED  (total 12)"
echo ""

# GPU 状态
echo "=== GPU 状态 ==="
rocm-smi | grep -E "^[0-9]\s+" | awk '{printf "  GPU %s: %s%% util, %s%% VRAM, %s\n", $1, $7, $6, $9}'
echo ""

# 如果全部 done, 打印汇总表
if [ "$DONE" -eq 12 ]; then
    echo "===== 🎉 全部 12 cell 完成! Table III 数据 ====="
    echo ""
    echo "                K=10        K=20        K=50"
    for alpha in 0.05 0.1 0.3 0.5; do
        printf "α=%-6s" "$alpha"
        for k in 10 20 50; do
            json="results/znorm_cifar100_a${alpha}_k${k}_s42.json"
            v=$(python3 -c "import json; d=json.load(open('$json')); print(f\"{d['acc_dynamic']*100:.2f}\")" 2>/dev/null)
            printf "  %-10s" "$v"
        done
        echo ""
    done
    echo ""
    echo "+ K=5 列 (从现有 znorm_cifar100_a*_k5_s42.json):"
    echo "                K=5"
    for alpha in 0.05 0.1 0.3 0.5; do
        json="results/znorm_cifar100_a${alpha}_k5_s42.json"
        v=$(python3 -c "import json; d=json.load(open('$json')); print(f\"{d.get('acc_dynamic', d.get('best_acc', 0))*100:.2f}\")" 2>/dev/null)
        printf "α=%-6s  %s\n" "$alpha" "$v"
    done
fi
