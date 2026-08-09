#!/usr/bin/env python
"""Summarize the grouped-merge G-sweep (serving-cost knob) into one table.

Reads results/groupmerge_cifar100_a*_k*_G*_s*.json (one cell per file, thr=0.95),
groups by (alpha, K), and reports for each serving-cost point G:
  n_models, accuracy, total params, params vs full K-ensemble, params vs one backbone.

G = K  -> full ensemble (top accuracy, most serving cost).
G = 1  -> single merged model (cheapest serving, O(1) in K at inference).

Writes results/summary_groupmerge_cifar100.csv and prints a markdown view.
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

rows = []
for fp in glob.glob(os.path.join(RES, "groupmerge_cifar100_a*_k*_G*_s*.json")):
    with open(fp) as f:
        d = json.load(f)
    ens = d["ensemble_params"]
    single = d["single_backbone_params"]
    for r in d["results"]:
        rows.append({
            "alpha": d["alpha"],
            "K": d["K"],
            "G": r["G"],
            "n_models": r["n_models"],
            "acc": r["acc"],
            "thr": r["thr"],
            "tot_params": r["tot_params"],
            "vs_ensemble": r["tot_params"] / ens,
            "vs_single": r["tot_params"] / single,
        })

rows.sort(key=lambda x: (x["alpha"], x["K"], x["G"]))

# CSV
csv_path = os.path.join(RES, "summary_groupmerge_cifar100.csv")
cols = ["alpha", "K", "G", "n_models", "acc", "tot_params", "vs_ensemble", "vs_single", "thr"]
with open(csv_path, "w") as f:
    f.write(",".join(cols) + "\n")
    for r in rows:
        f.write(",".join(str(r[c]) for c in cols) + "\n")

# Markdown, grouped by (alpha, K)
print(f"# Grouped-merge G-sweep (CIFAR-100, thr=0.95, seed 42) — {len(rows)} points\n")
last = None
for r in rows:
    key = (r["alpha"], r["K"])
    if key != last:
        if last is not None:
            print()
        print(f"## alpha={r['alpha']}  K={r['K']}  "
              f"(1 backbone = {r['tot_params']/r['vs_single']/1e6:.1f}M, "
              f"full ensemble = {r['tot_params']/r['vs_ensemble']/1e6:.0f}M)")
        print("| G | n_models | acc | params(M) | x ensemble | x 1-backbone |")
        print("|---|---|---|---|---|---|")
        last = key
    print(f"| {r['G']} | {r['n_models']} | {r['acc']*100:.2f}% | "
          f"{r['tot_params']/1e6:.0f} | {r['vs_ensemble']:.2f}x | {r['vs_single']:.1f}x |")

print(f"\nCSV written: {csv_path}")
