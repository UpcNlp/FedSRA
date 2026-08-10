#!/usr/bin/env bash
set -euo pipefail

ROOT=/public/home/dongshou/fedETF
OUT="$ROOT/realfed_out"
PY=/public/home/dongshou/anaconda/envs/ct/bin/python
SUMMARY="$ROOT/review_response/realfed_summary.csv"

expected=()
for method in fedsra ce fafi coboost; do
  for seed in 0 42 123; do
    expected+=("realfed_binary_${method}_heldout-none_s${seed}.json")
  done
done
for method in fedsra ce fafi; do
  for seed in 0 42 123; do
    expected+=("realfed_binary_${method}_heldout-mbrset_s${seed}.json")
  done
done

while true; do
  missing=0
  for file in "${expected[@]}"; do
    [[ -f "$OUT/results/$file" ]] || missing=$((missing + 1))
  done
  echo "[$(date '+%F %T')] completed=$(( ${#expected[@]} - missing ))/${#expected[@]}"
  [[ $missing -eq 0 ]] && break
  sleep 120
done

cd "$ROOT"
"$PY" review_response/experiments/validate_realfed_results.py \
  --results "$OUT/results"
"$PY" review_response/experiments/summarize_realfed.py \
  --results "$OUT/results" --output "$SUMMARY"
echo "[$(date '+%F %T')] summary complete: $SUMMARY"
