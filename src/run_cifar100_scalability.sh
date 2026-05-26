#!/bin/bash
# ============================================================
# CIFAR-100 OURS scalability 完整 launcher
# 14233, 6 GPU (0/1/4/5/6/7), 每 GPU 跑 2 个 cell 串行, 共 12 cell
# 估计 wall time: ~14-18h
# ============================================================

set -e
cd /public/home/dongshou/fedETF/ETF-pesuade
mkdir -p logs

PY=/public/home/dongshou/anaconda/envs/ct/bin/python
TS=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="logs/c100_master_${TS}.log"

echo "===== CIFAR-100 OURS scalability launcher =====" | tee "$MASTER_LOG"
echo "Start: $(date)" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# ============================================================
# 任务分配 (固定):
# GPU 0: α=0.5 K=10 → α=0.05 K=20    (3.5+10 = 13.5h)
# GPU 1: α=0.05 K=50                  (14h, 独占)
# GPU 4: α=0.3 K=10 → α=0.1 K=50      (4+13 = 17h)
# GPU 5: α=0.5 K=20 → α=0.5 K=50      (7+10 = 17h)
# GPU 6: α=0.1 K=10 → α=0.3 K=50      (4.5+11 = 15.5h)
# GPU 7: α=0.05 K=10 → α=0.3 K=20 → α=0.1 K=20  (5+8+9 = 22h, 最长链)
# ============================================================

# 每个 GPU 一个 chain 函数: 串行跑多个 (alpha, K)
run_chain() {
    local gpu=$1
    shift
    local chain_log="logs/c100_gpu${gpu}_chain_${TS}.log"
    echo "[GPU $gpu] chain start at $(date)" > "$chain_log"

    while [ $# -gt 0 ]; do
        local alpha=$1
        local k=$2
        shift 2
        local cell_log="logs/c100_a${alpha}_k${k}_s42_gpu${gpu}_${TS}.log"

        echo "" >> "$chain_log"
        echo "===== [GPU $gpu] α=$alpha K=$k start $(date) =====" >> "$chain_log"

        HIP_VISIBLE_DEVICES=$gpu $PY run_znorm_cifar100.py \
            --alpha "$alpha" --n_clients "$k" --seed 42 \
            > "$cell_log" 2>&1

        local rc=$?
        if [ $rc -eq 0 ]; then
            local acc=$(grep -oE "acc_dynamic.*[0-9.]+" "$cell_log" | tail -1)
            echo "[GPU $gpu] α=$alpha K=$k DONE rc=0  $acc  at $(date)" >> "$chain_log"
        else
            echo "[GPU $gpu] α=$alpha K=$k FAILED rc=$rc at $(date)" >> "$chain_log"
        fi
    done

    echo "" >> "$chain_log"
    echo "[GPU $gpu] chain done at $(date)" >> "$chain_log"
}

export -f run_chain
export PY TS

# ============================================================
# 启动 6 个 chain (后台, 独立进程)
# ============================================================

# GPU 0: smoke test (α=0.5 K=10) 如果还在跑就跳过; 然后接 α=0.05 K=20
SMOKE_PID=$(cat .c100_smoke.pid 2>/dev/null || echo "")
if [ -n "$SMOKE_PID" ] && ps -p $SMOKE_PID > /dev/null 2>&1; then
    echo "[GPU 0] smoke test 还在跑 (pid=$SMOKE_PID), 等它跑完再接 α=0.05 K=20" | tee -a "$MASTER_LOG"
    nohup bash -c "
        while ps -p $SMOKE_PID > /dev/null 2>&1; do sleep 60; done
        run_chain 0 0.05 20
    " > "logs/c100_gpu0_wait_${TS}.log" 2>&1 &
    echo "  GPU 0 waiter pid=$!" | tee -a "$MASTER_LOG"
else
    echo "[GPU 0] smoke test 不在了, 起完整 chain" | tee -a "$MASTER_LOG"
    nohup bash -c "run_chain 0 0.5 10 0.05 20" > /dev/null 2>&1 &
    echo "  GPU 0 chain pid=$!" | tee -a "$MASTER_LOG"
fi

# GPU 1: α=0.05 K=50 独占
nohup bash -c "run_chain 1 0.05 50" > /dev/null 2>&1 &
echo "[GPU 1] chain pid=$!  (α=0.05 K=50)" | tee -a "$MASTER_LOG"

# GPU 4: α=0.3 K=10 → α=0.1 K=50
nohup bash -c "run_chain 4 0.3 10 0.1 50" > /dev/null 2>&1 &
echo "[GPU 4] chain pid=$!  (α=0.3 K=10 → α=0.1 K=50)" | tee -a "$MASTER_LOG"

# GPU 5: α=0.5 K=20 → α=0.5 K=50
nohup bash -c "run_chain 5 0.5 20 0.5 50" > /dev/null 2>&1 &
echo "[GPU 5] chain pid=$!  (α=0.5 K=20 → α=0.5 K=50)" | tee -a "$MASTER_LOG"

# GPU 6: α=0.1 K=10 → α=0.3 K=50
nohup bash -c "run_chain 6 0.1 10 0.3 50" > /dev/null 2>&1 &
echo "[GPU 6] chain pid=$!  (α=0.1 K=10 → α=0.3 K=50)" | tee -a "$MASTER_LOG"

# GPU 7: α=0.05 K=10 → α=0.3 K=20 → α=0.1 K=20 (最长链)
nohup bash -c "run_chain 7 0.05 10 0.3 20 0.1 20" > /dev/null 2>&1 &
echo "[GPU 7] chain pid=$!  (α=0.05 K=10 → α=0.3 K=20 → α=0.1 K=20)" | tee -a "$MASTER_LOG"

echo "" | tee -a "$MASTER_LOG"
echo "===== 全部 chain 已启动 =====" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "查看进度:"
echo "  tail -f logs/c100_gpu*_chain_${TS}.log"
echo ""
echo "查看某 cell 实时训练:"
echo "  tail -f logs/c100_a<α>_k<K>_s42_gpu<G>_${TS}.log"
echo ""
echo "60s 后初步验证..."
sleep 60

echo ""
echo "===== 60s 后状态 ====="
echo ""
echo "--- 各 chain 进度 ---"
for gpu in 0 1 4 5 6 7; do
    f="logs/c100_gpu${gpu}_chain_${TS}.log"
    if [ -f "$f" ]; then
        echo "[GPU $gpu]"
        tail -3 "$f" | sed 's/^/    /'
    fi
done

echo ""
echo "--- 当前 cell tqdm 进度 ---"
for log in logs/c100_a*_gpu*_${TS}.log; do
    [ -f "$log" ] || continue
    last=$(grep -E "BB c|Expert|eval" "$log" 2>/dev/null | tail -1)
    [ -n "$last" ] && echo "  $(basename $log): $last"
done

echo ""
echo "--- rocm-smi ---"
rocm-smi | grep -E "^[0-9]\s+"
