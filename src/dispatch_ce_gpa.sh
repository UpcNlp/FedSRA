#!/bin/bash
set -e
set +u; source /opt/dtk/env.sh 2>/dev/null || true; set -u

PYTHON=/public/home/dongshou/anaconda/envs/ct/bin/python
SCRIPT=/public/home/dongshou/fedETF/Co-Boosting-PP-master/eval_noetf_geom.py
LOGDIR=/public/home/dongshou/fedETF/ETF-pesuade/logs/ce_gpa
RESULT=/public/home/dongshou/fedETF/ETF-pesuade/results/ce_gpa_grid.json
mkdir -p $LOGDIR

JOBS=(
  "cifar10:5:300"
  "cifar10:10:700"
  "cifar10:20:300"
  "cifar100:10:300"
  "cifar100:20:300"
)

NUM_GPU=4
i=0
for cell in "${JOBS[@]}"; do
  IFS=":" read ds K le <<< "$cell"
  for a in 0.05 0.1 0.3 0.5; do
    gpu=$((i % NUM_GPU))
    LOG=$LOGDIR/${ds}_K${K}_a${a}.log
    echo "[GPU $gpu] $ds K=$K a=$a Le=$le -> $LOG"
    HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
      $PYTHON $SCRIPT --dataset $ds --alpha $a --le $le --n_clients $K --seed 42 \
      > $LOG 2>&1 &
    i=$((i+1))
    if (( i % NUM_GPU == 0 )); then wait; fi
  done
done
wait

# Parse all logs into JSON
$PYTHON - << PYEOF
import json, re, glob, os
out = {}
for f in sorted(glob.glob("/*.log")):
    name = os.path.basename(f).replace(".log","")
    with open(f) as fp: text = fp.read()
    m = re.search(r"FD=\d+\s+ENSEMBLE\(CE head\)=([\d.]+)\s+GEOMETRIC\(znorm->ETF\d+\)=([\d.]+)", text)
    if m:
        out[name] = {"ens": float(m.group(1)), "geo": float(m.group(2))}
    else:
        out[name] = {"error": True, "tail": text[-200:]}
with open("", "w") as fp:
    json.dump(out, fp, indent=2)
print("RESULTS:")
for k,v in out.items(): print(f"  {k}: {v}")
PYEOF

echo "ALL DONE - results in "
