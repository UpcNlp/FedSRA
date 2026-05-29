#!/bin/bash
# run_rij_cell.sh <dataset> <K> <gpu> <alpha1> [alpha2 ...]
# Sources ROCm/DTK env (else torch ImportError libgalaxyhip), loops R+I, resume-aware.
# Launch with setsid so it survives ssh session disconnect:
#   setsid bash run_rij_cell.sh cifar100 5 1 0.05 0.1 0.3 0.5 </dev/null >logs/x.log 2>&1 &
set -u
cd /public/home/dongshou/fedETF/ETF-pesuade

# ---- ROCm / DTK environment (captured from working ct torch procs) ----
export ROCM_PATH=/opt/dtk
export HIP_PATH=/opt/dtk/hip
export PATH=/opt/dtk/bin:/opt/dtk/llvm/bin:/opt/dtk/hip/bin:/opt/conda/bin:$PATH
export LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/opencl/lib:${LD_LIBRARY_PATH:-}

PY=/public/home/dongshou/anaconda/envs/ct/bin/python
SEED=42
DS=$1; K=$2; GPU=$3; shift 3
ALPHAS="$@"
mkdir -p logs

echo "[$(date +%F_%T)] run_rij_cell ds=$DS K=$K gpu=$GPU alphas=$ALPHAS"
for a in $ALPHAS; do
  for loss in R I; do
    tag="${DS}_K${K}_a${a}_${loss}"
    echo "[$(date +%T)] start $tag"
    HIP_VISIBLE_DEVICES=$GPU CUDA_VISIBLE_DEVICES=$GPU \
      $PY -u run_ablation_RIJ.py --dataset $DS --loss_type $loss \
          --alpha $a --n_clients $K --seed $SEED --resume \
      > logs/ablation_RIJ_${tag}.log 2>&1
    rc=$?
    echo "[$(date +%T)] done $tag (rc=$rc)"
  done
done
echo "[$(date +%F_%T)] ALL DONE ds=$DS K=$K gpu=$GPU"
