#!/usr/bin/env python3
"""Fed-ISIC2019 schema inspection + natural-skew GATE (CPU-only, seconds).

Purpose (two jobs in one, per REBUTTAL_EXECUTION_SPEC.md 4.1):
  1. Print the parquet schema so the training script can be written against
     the *real* column names (image / label / center) with zero guesswork.
  2. Measure the natural label skew across the 6 FLamby centers -- the GATE
     that decides whether Fed-ISIC is a strong-skew in-regime dataset worth
     betting the make-or-break on.

No torch, no GPU. Only pandas + pyarrow (+ numpy). Nothing is decoded from the
image column; we only inspect its type.

Usage (on the cluster):
  python fedisic_inspect.py \
    --train /public/home/dongshou/fedETF/realfed_data/Fed-ISIC2019/hf_flower/data/train-00000-of-00001.parquet \
    --test  /public/home/dongshou/fedETF/realfed_data/Fed-ISIC2019/hf_flower/data/test-00000-of-00001.parquet \
    --reference /public/home/dongshou/fedETF/realfed_data/Fed-ISIC2019/flamby_reference/
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


def describe_value(v):
    """Human-readable type of a cell without dumping image bytes."""
    if isinstance(v, dict):
        keys = list(v.keys())
        shapes = {}
        for k in keys:
            vv = v[k]
            if isinstance(vv, (bytes, bytearray)):
                shapes[k] = f"bytes[{len(vv)}]"
            else:
                shapes[k] = type(vv).__name__
        return f"dict{keys} -> {shapes}"
    if isinstance(v, (bytes, bytearray)):
        return f"bytes[{len(v)}]"
    if isinstance(v, (np.ndarray, list)):
        return f"{type(v).__name__}[len={len(v)}]"
    s = str(v)
    return f"{type(v).__name__}={s[:60]}"


def print_schema(df: pd.DataFrame, name: str) -> None:
    print(f"\n{'='*70}\n[{name}] shape={df.shape}\n{'='*70}")
    print("columns / dtype / sample value:")
    row0 = df.iloc[0]
    for col in df.columns:
        print(f"  - {col:<24s} {str(df[col].dtype):<12s} {describe_value(row0[col])}")


def candidate_columns(df: pd.DataFrame):
    """Guess which column is the class label and which is the center/client."""
    label_cands, center_cands = [], []
    for col in df.columns:
        if col.lower() in ("image", "img", "pixels"):
            continue
        try:
            nun = df[col].nunique(dropna=True)
        except TypeError:
            continue  # unhashable (e.g. image dicts)
        # class label: ~2..40 unique, integer-ish, name hints
        if 2 <= nun <= 40:
            name_hint = any(h in col.lower() for h in ("label", "target", "class", "diagnos", "y"))
            center_hint = any(h in col.lower() for h in ("center", "client", "site", "hospital", "domain", "partition", "dataset", "source"))
            if center_hint or nun <= 8:
                center_cands.append((col, nun))
            if name_hint or (pd.api.types.is_integer_dtype(df[col]) and nun <= 20):
                label_cands.append((col, nun))
    return label_cands, center_cands


def skew_report(df: pd.DataFrame, center_col: str, label_col: str) -> None:
    print(f"\n{'#'*70}\n# GATE: natural label skew  (center='{center_col}', label='{label_col}')\n{'#'*70}")
    ct = pd.crosstab(df[center_col], df[label_col])
    print("\nper-center class counts (rows=center, cols=class):")
    print(ct.to_string())

    classes = ct.columns.tolist()
    C = len(classes)
    global_dist = ct.sum(0) / ct.values.sum()
    print(f"\n#classes={C}  global class fractions:")
    print("  " + "  ".join(f"{c}:{global_dist[c]*100:5.1f}%" for c in classes))

    print("\nper-center skew metrics:")
    print(f"  {'center':<12s} {'n':>7s} {'ncls>=1%':>9s} {'norm_H':>7s} {'L1_to_global':>13s}")
    eff_alphas = []
    for center in ct.index:
        counts = ct.loc[center].to_numpy(dtype=float)
        n = counts.sum()
        p = counts / max(n, 1)
        cov = int((p >= 0.01).sum())                       # classes with >=1% mass
        H = -sum(pi * math.log(pi) for pi in p if pi > 0)  # entropy (nats)
        normH = H / math.log(C) if C > 1 else 0.0          # 0=one class, 1=uniform
        l1 = float(np.abs(p - global_dist.to_numpy()).sum())
        eff_alphas.append(normH)
        print(f"  {str(center):<12s} {int(n):>7d} {cov:>9d} {normH:>7.3f} {l1:>13.3f}")

    mean_normH = float(np.mean(eff_alphas))
    print(f"\n  mean normalized entropy across centers = {mean_normH:.3f}")
    print("  interpretation:")
    print("    normH -> 0.0  = each center sees ~1 class  (extreme incomplete coverage, best for FedSRA)")
    print("    normH -> 1.0  = each center sees all classes uniformly (no label skew; FedSRA has no edge)")
    verdict = (
        "STRONG skew -> in-regime, worth the make-or-break" if mean_normH < 0.6 else
        "MODERATE skew -> borderline; check coverage/feature-shift before betting" if mean_normH < 0.8 else
        "WEAK skew -> likely DomainNet-redux; do NOT bet make-or-break here"
    )
    print(f"\n  GATE VERDICT (heuristic): {verdict}")
    print("  (final call also needs feature-shift assessment; dermoscopy is same-modality so expected mild.)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--test", type=Path, default=None)
    ap.add_argument("--reference", type=Path, default=None)
    ap.add_argument("--center_col", default=None, help="override auto-detected center column")
    ap.add_argument("--label_col", default=None, help="override auto-detected label column")
    args = ap.parse_args()

    print(f"pandas={pd.__version__}")
    df = pd.read_parquet(args.train)
    print_schema(df, f"TRAIN {args.train.name}")
    if args.test and args.test.exists():
        dft = pd.read_parquet(args.test, columns=None)
        print_schema(dft, f"TEST {args.test.name}")

    label_cands, center_cands = candidate_columns(df)
    print("\n--- auto-detected candidates ---")
    print(f"  label candidates : {label_cands}")
    print(f"  center candidates: {center_cands}")

    # also dump value_counts of every low-cardinality column so nothing is missed
    print("\n--- value_counts for all low-cardinality (<=12) columns ---")
    for col in df.columns:
        try:
            nun = df[col].nunique(dropna=True)
        except TypeError:
            continue
        if nun <= 12:
            vc = df[col].value_counts(dropna=False).sort_index()
            print(f"  [{col}] ({nun} unique): {dict(vc)}")

    center_col = args.center_col or (center_cands[0][0] if center_cands else None)
    label_col = args.label_col or (label_cands[0][0] if label_cands else None)
    if center_col and label_col:
        skew_report(df, center_col, label_col)
    else:
        print("\n[!] could not auto-detect center/label columns; re-run with "
              "--center_col and --label_col once you see the schema above.")

    if args.reference and args.reference.exists():
        print(f"\n--- flamby_reference/ contents ({args.reference}) ---")
        for p in sorted(args.reference.rglob("*")):
            if p.is_file():
                print(f"  {p.relative_to(args.reference)}  ({p.stat().st_size} B)")


if __name__ == "__main__":
    main()
