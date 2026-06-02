#!/bin/bash
set +u; source /opt/dtk/env.sh 2>/dev/null || true; set -u
cd /public/home/dongshou/fedETF/ETF-pesuade

PYTHON=/public/home/dongshou/anaconda/envs/ct/bin/python
LOGDIR=logs/section_a_complete
mkdir -p $LOGDIR

# Wait for GPA variants dispatch to leave GPUs free
echo "Waiting for GPA variants to finish first..."
while pgrep -af ablation_aggregation_v2 > /dev/null 2>&1; do sleep 30; done
echo "GPA variants done, dispatching Section A inference"

# Section A missing cells:
# (a) cifar100 K=50 CE+Ens / CE+GPA × 4 α (use eval_noetf_geom.py)
# (b) cifar100 K=20 α=.1/.5 RIJ eval (the only missing I cells in K=5/10/20)
# (c) cifar100 K=50 α=.05 RIJ eval (have I backbone only at α=.05)

JOBS=()
# (a) K=50 CE+GPA inference
for a in 0.05 0.1 0.3 0.5; do
  JOBS+=("ce_gpa cifar100 50 300 $a")
done
# (b) K=20 α=.1/.5 RIJ eval
for a in 0.1 0.5; do
  JOBS+=("rij cifar100 20 - $a")
done
# (c) K=50 α=.05 RIJ eval (have I only at this α)
JOBS+=("rij cifar100 50 - 0.05")

NUM_GPU=4
i=0
for job in "${JOBS[@]}"; do
  read kind ds K le a <<< "$job"
  gpu=$((i % NUM_GPU))
  LOG=$LOGDIR/${kind}_${ds}_K${K}_a${a}.log
  echo "[GPU $gpu] $kind $ds K=$K α=$a -> $LOG"
  if [ "$kind" = "ce_gpa" ]; then
    HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
      $PYTHON /public/home/dongshou/fedETF/Co-Boosting-PP-master/eval_noetf_geom.py \
        --dataset $ds --alpha $a --le $le --n_clients $K --seed 42 \
        > $LOG 2>&1 &
  else
    HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
      $PYTHON eval_ablation_RIJ.py \
        --dataset $ds --alpha $a --n_clients $K --seed 42 \
        > $LOG 2>&1 &
  fi
  i=$((i+1))
  if (( i % NUM_GPU == 0 )); then wait; fi
done
wait
echo "===== SECTION A COMPLETE ====="
