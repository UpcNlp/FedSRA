#!/bin/bash
set +u; source /opt/dtk/env.sh 2>/dev/null || true; set -u
cd /public/home/dongshou/fedETF/ETF-pesuade
PYTHON=/public/home/dongshou/anaconda/envs/ct/bin/python
LOGDIR=logs/section_a_direct
mkdir -p $LOGDIR

# (a) CIFAR-100 K=50 CE+Ens / CE+GPA × 4 α
# (b) CIFAR-100 K=20 α=.1/.5 RIJ eval
# (c) CIFAR-100 K=50 α=.05 RIJ eval (I@α=.05 backbone exists)

JOBS=(
  "ce_gpa cifar100 50 0.05"
  "ce_gpa cifar100 50 0.1"
  "ce_gpa cifar100 50 0.3"
  "ce_gpa cifar100 50 0.5"
  "rij    cifar100 20 0.1"
  "rij    cifar100 20 0.5"
  "rij    cifar100 50 0.05"
)

NUM_GPU=4
i=0
for job in "${JOBS[@]}"; do
  read kind ds K a <<< "$job"
  gpu=$((i % NUM_GPU))
  LOG=$LOGDIR/${kind}_${ds}_K${K}_a${a}.log
  echo "[GPU $gpu] $kind $ds K=$K α=$a"
  if [ "$kind" = "ce_gpa" ]; then
    HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
      $PYTHON /public/home/dongshou/fedETF/Co-Boosting-PP-master/eval_noetf_geom.py \
        --dataset $ds --alpha $a --le 300 --n_clients $K --seed 42 \
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
echo "===== ALL DONE ====="
