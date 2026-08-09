#!/bin/bash
# Overnight orchestrator v2 (re-scoped after A2's large-K latency timing proved
# impractically slow). Order by value/tractability:
#   1. multi-round ETF curve (R1-D4/R3-W7) -- cheap, high value, was being starved
#   2. lean A2 grouped-merge cost (K=10, fast, clean latency) -- R2-D2 Pareto
#   3. B seeds smoke -> train -> eval (R2-W4/D3)
# Launch with setsid. Everything resume-safe.
cd /public/home/dongshou/fedETF/ETF-pesuade
export LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/opencl/lib:$LD_LIBRARY_PATH
export HIP_PATH=/opt/dtk ROCM_PATH=/opt/dtk DTKROOT=/opt/dtk
PY=/public/home/dongshou/anaconda/envs/ct/bin/python
mkdir -p logs
log() { echo "[$(date '+%F %T')] $*" | tee -a logs/overnight2.log; }

log "ORCHESTRATOR v2 START"

log "phase 1: multi-round ETF curve"
bash run_multiround_fanout.sh >> logs/overnight2.log 2>&1
log "phase 1 done"

log "phase 2: lean A2 grouped-merge cost (K=10)"
bash run_gmcost_lean.sh >> logs/overnight2.log 2>&1
log "phase 2 done"

log "phase 3: B smoke test"
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 $PY -u run_ablation_RIJ.py \
  --dataset cifar100 --loss_type J --alpha 0.5 --n_clients 5 --seed 999 \
  --epochs_bb 2 --skip_experts > logs/bsmoke.log 2>&1
if [ -f saved_models/ablation_cifar100/J_a0.5_k5_s999/client_0/backbone.pt ]; then
  log "B smoke OK; launching B seed sweep"
  rm -rf saved_models/ablation_cifar100/J_a0.5_k5_s999
  bash run_bseeds_fanout.sh >> logs/overnight2.log 2>&1
  log "B training done; per-seed eval"
  for seed in 0 123; do for K in 5 10 20; do for a in 0.05 0.1 0.3 0.5; do
    ev="results/ablation_RIJ_eval_cifar100_a${a}_k${K}_s${seed}.json"
    [ -f "$ev" ] && continue
    [ -f "saved_models/ablation_cifar100/J_a${a}_k${K}_s${seed}/client_0/backbone.pt" ] || continue
    HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 $PY -u eval_ablation_RIJ.py \
      --dataset cifar100 --alpha "$a" --n_clients "$K" --seed "$seed" >> logs/beval.log 2>&1
  done; done; done
  log "B eval done"
else
  log "B SMOKE FAILED -- skipping B"
fi
log "ORCHESTRATOR v2 COMPLETE"
