#!/bin/bash
# run.sh
# ======
# 用法：bash run.sh <机器名> <GPU_ID>
#
# 四个命令（四个终端分别执行）：
#   bash run.sh 2080 0
#   bash run.sh 2080 1
#   bash run.sh 3080 0
#   bash run.sh 3080 1

MACHINE=${1}
GPU=${2}

if [ -z "$MACHINE" ] || [ -z "$GPU" ]; then
    echo "Usage: bash run.sh <机器名> <GPU_ID>"
    echo "  示例: bash run.sh 2080 0"
    echo "  示例: bash run.sh 2080 1"
    echo "  示例: bash run.sh 3080 0"
    echo "  示例: bash run.sh 3080 1"
    exit 1
fi

SEED=42
mkdir -p results logs

python prefill_relational.py

run_job() {
    local alpha=$1
    local k=$2
    local pipeline=$3
    echo "------------------------------"
    echo "RUN: alpha=${alpha}  K=${k}  pipeline=${pipeline}  GPU=${GPU}"
    echo "------------------------------"
    python run_grid.py \
        --alpha     ${alpha} \
        --n_clients ${k} \
        --seed      ${SEED} \
        --gpu       ${GPU} \
        --pipeline  ${pipeline}
}

case "${MACHINE}_${GPU}" in

  2080_0)
    run_job 0.05 100 both
    run_job 0.5  100 both
    run_job 1.0  100 both
    ;;

  2080_1)
    run_job 0.1  100 intrinsic
    run_job 0.2  100 intrinsic
    run_job 0.3  100 intrinsic
    ;;

  3080_0)
    run_job 0.1  10 both
    run_job 0.2  10 both
    run_job 0.3  10 both
    run_job 0.5  10 both
    run_job 1.0  10 both
    ;;

  3080_1)
    run_job 1.0  20 both
    run_job 0.05 50 both
    run_job 0.1  50 intrinsic
    ;;

  *)
    echo "Unknown combination: machine=${MACHINE} gpu=${GPU}"
    echo "Valid: 2080 0 / 2080 1 / 3080 0 / 3080 1"
    exit 1
    ;;

esac

echo "=============================="
echo "Machine ${MACHINE} GPU ${GPU} all tasks done."
echo "=============================="