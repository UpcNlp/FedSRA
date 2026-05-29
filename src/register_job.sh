#!/bin/bash
# register_job.sh — declare a job's serial queue so the dashboard can show
# done / running / pending. Run this ON THE HOST where the job runs, from the
# repo root (or anywhere — it locates the repo via REPO env or its own path).
#
# Usage:
#   register_job.sh <label> <gpu> <tag1> [tag2 ...]
#
# Example (a launcher calls this once before its loop):
#   register_job.sh "RIJ c100 K10" 4 \
#       cifar100_K10_a0.05_R cifar100_K10_a0.05_I \
#       cifar100_K10_a0.1_R  cifar100_K10_a0.1_I  ...
#
# Tags are LOG-style: <dataset>_K<k>_a<alpha>_<loss>  (e.g. cifar100_K10_a0.05_R)
# The dashboard maps each tag to results/ablation_RIJ_meta_<dataset>_<loss>_a<alpha>_k<k>_s42.json
# to decide "done", and to a live process to decide "running".
#
# Writes:  $REPO/exp_registry/<host>__<label-slug>.json
# No shared FS assumption: each host keeps its own registry; the 'host' field
# lets the dashboard dedup entries on cluster nodes that share /public.

set -u

LABEL="${1:?usage: register_job.sh <label> <gpu> <tag1> [tag2 ...]}"
GPU="${2:?need gpu id}"
shift 2
QUEUE=("$@")

# locate repo: env REPO, else assume we're inside it (find ETF-pesuade marker)
if [ -n "${REPO:-}" ]; then
    R="$REPO"
elif [ -f "run_ablation_RIJ.py" ]; then
    R="$(pwd)"
elif [ -f "ETF-pesuade/run_ablation_RIJ.py" ]; then
    R="$(pwd)/ETF-pesuade"
else
    echo "cannot locate repo; set REPO=/path/to/ETF-pesuade" >&2
    exit 1
fi

# Host tag must match the 'name' field in the dashboard's hosts.json.
# Override with HOSTTAG=... when hostname doesn't encode it (e.g. standalone boxes).
HOST="${HOSTTAG:-$(hostname | sed 's/.*dongshou-//; s/notebook-//')}"
SLUG=$(echo "$LABEL" | tr ' /' '__' | tr -cd 'A-Za-z0-9_.-')
mkdir -p "$R/exp_registry"
OUT="$R/exp_registry/${HOST}__${SLUG}.json"

# build queue JSON array
QJSON=""
for t in "${QUEUE[@]}"; do
    QJSON="${QJSON}\"$t\","
done
QJSON="[${QJSON%,}]"

cat > "$OUT" <<EOF
{
  "host": "$HOST",
  "label": "$LABEL",
  "experiment": "$LABEL",
  "gpu": "$GPU",
  "queue": $QJSON,
  "started": "$(date '+%Y-%m-%d %H:%M:%S')",
  "n_queue": ${#QUEUE[@]}
}
EOF

echo "registered: $OUT"
echo "  host=$HOST gpu=$GPU n_queue=${#QUEUE[@]}"
