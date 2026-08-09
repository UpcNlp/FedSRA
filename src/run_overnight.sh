#!/bin/bash
# Overnight orchestrator. Runs autonomously (launch with setsid). Chains:
#   1. wait for Group C (MedMNIST) to finish
#   2. A2 grouped-merge cost sweep (needs clean GPUs for latency timing)
#   3. B1 smoke gate -> extra-seed FedSRA training -> per-seed eval
# Everything is resume-safe; a re-launch continues where it stopped.
cd /public/home/dongshou/fedETF/ETF-pesuade
export LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/opencl/lib:$LD_LIBRARY_PATH
export HIP_PATH=/opt/dtk ROCM_PATH=/opt/dtk DTKROOT=/opt/dtk
PY=/public/home/dongshou/anaconda/envs/ct/bin/python
mkdir -p logs
log() { echo "[$(date '+%F %T')] $*" | tee -a logs/overnight.log; }

log "ORCHESTRATOR START"

# --- 1. wait for Group C (MedMNIST) ---
log "waiting for MedMNIST (Group C) to finish..."
while ps -eo cmd | grep -q '[m]edmnist_fedsra.py'; do sleep 120; done
log "Group C finished (medmnist procs = 0)"

# --- 2. A2 grouped-merge cost sweep ---
log "starting A2 grouped-merge cost sweep"
bash run_gmcost_fanout.sh >> logs/overnight.log 2>&1
log "A2 done"

# --- 3. B smoke gate ---
log "B smoke test (2 epochs, throwaway seed 999)"
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 $PY -u run_ablation_RIJ.py \
  --dataset cifar100 --loss_type J --alpha 0.5 --n_clients 5 --seed 999 \
  --epochs_bb 2 --skip_experts > logs/bsmoke.log 2>&1
if [ -f saved_models/ablation_cifar100/J_a0.5_k5_s999/client_0/backbone.pt ]; then
  log "B smoke OK; launching full B seed sweep"
  rm -rf saved_models/ablation_cifar100/J_a0.5_k5_s999
  bash run_bseeds_fanout.sh >> logs/overnight.log 2>&1
  log "B training done; evaluating per-seed accuracy"
  for seed in 0 123; do for K in 5 10 20; do for a in 0.05 0.1 0.3 0.5; do
    ev="results/ablation_RIJ_eval_cifar100_a${a}_k${K}_s${seed}.json"
    [ -f "$ev" ] && continue
    [ -f "saved_models/ablation_cifar100/J_a${a}_k${K}_s${seed}/client_0/backbone.pt" ] || continue
    HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 $PY -u eval_ablation_RIJ.py \
      --dataset cifar100 --alpha "$a" --n_clients "$K" --seed "$seed" \
      >> logs/beval.log 2>&1
  done; done; done
  log "B eval done"
else
  log "B SMOKE FAILED -- skipping B (see logs/bsmoke.log)"
fi

log "ORCHESTRATOR COMPLETE"
