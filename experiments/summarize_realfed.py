#!/usr/bin/env python3
"""Aggregate completed RealFed cells into mean +/- std tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


VARIANTS: Dict[str, List[Tuple[str, str]]] = {
    "fedsra": [
        ("FedSRA (local moments)", "rga_client_local_moments"),
        ("FedSRA (full-batch diagnostic)", "rga_full_batch_diagnostic"),
        ("FedSRA (per-sample LN)", "rga_per_sample_layernorm"),
    ],
    "ce": [
        ("O-FedAvg", "one_shot_fedavg"),
        ("CE ensemble", "uniform_logit_ensemble"),
        ("CE ensemble (sqrt)", "sqrt_weighted_logit_ensemble"),
    ],
    "fafi": [("FAFI", "fafi_weighted_feature_ensemble")],
    "coboost": [("Co-Boosting", "coboost_student")],
}
METRICS = [
    "balanced_accuracy",
    "auroc",
    "auprc",
    "macro_f1",
    "sensitivity",
    "specificity",
]
SEEDS = (0, 42, 123)


def collect(results_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    records = []
    missing = []
    for heldout in ("none", "mbrset"):
        methods = ("fedsra", "ce", "fafi", "coboost")
        for method in methods:
            if heldout != "none" and method == "coboost":
                continue
            for seed in SEEDS:
                tag = f"realfed_binary_{method}_heldout-{heldout}_s{seed}"
                path = results_dir / f"{tag}.json"
                if not path.exists():
                    missing.append(tag)
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                for display_name, variant in VARIANTS[method]:
                    for domain, domain_results in payload["evaluation"].items():
                        if variant not in domain_results:
                            missing.append(f"{tag}:{variant}")
                            continue
                        metric_values = domain_results[variant]
                        record = {
                            "setting": "three-source" if heldout == "none" else "held-out mBRSET",
                            "heldout": heldout,
                            "method": display_name,
                            "base_method": method,
                            "variant": variant,
                            "seed": seed,
                            "domain": domain,
                            "elapsed_s": payload.get("elapsed_s", np.nan),
                            "gpu_peak_mb": payload.get("gpu_peak_mb", np.nan),
                        }
                        for metric in METRICS:
                            record[metric] = metric_values.get(metric, np.nan)
                        records.append(record)
                    if heldout == "none":
                        participating = payload.get("clients", ["brset", "mbrset", "odir"])
                        per_source = [
                            payload["evaluation"][source][variant]["balanced_accuracy"]
                            for source in participating
                        ]
                        record = {
                            "setting": "three-source",
                            "heldout": heldout,
                            "method": display_name,
                            "base_method": method,
                            "variant": variant,
                            "seed": seed,
                            "domain": "worst-source",
                            "elapsed_s": payload.get("elapsed_s", np.nan),
                            "gpu_peak_mb": payload.get("gpu_peak_mb", np.nan),
                        }
                        for metric in METRICS:
                            record[metric] = np.nan
                        record["balanced_accuracy"] = min(per_source)
                        records.append(record)
    return pd.DataFrame(records), missing


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    grouped = raw.groupby(
        ["setting", "heldout", "method", "base_method", "variant", "domain"],
        sort=False,
        dropna=False,
    )
    rows = []
    for keys, frame in grouped:
        row = dict(
            zip(
                ["setting", "heldout", "method", "base_method", "variant", "domain"],
                keys,
            )
        )
        row["n_seeds"] = int(frame["seed"].nunique())
        for metric in METRICS:
            values = frame[metric].dropna().to_numpy(dtype=float) * 100.0
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else np.nan
            )
        elapsed = frame["elapsed_s"].dropna().to_numpy(dtype=float)
        peak = frame["gpu_peak_mb"].dropna().to_numpy(dtype=float)
        row["elapsed_min_mean"] = float(elapsed.mean() / 60.0) if len(elapsed) else np.nan
        row["gpu_peak_mb_mean"] = float(peak.mean()) if len(peak) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def format_pm(mean: float, std: float, n: int) -> str:
    if pd.isna(mean):
        return "--"
    if n <= 1 or pd.isna(std):
        return f"{mean:.2f}"
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def markdown_table(summary: pd.DataFrame, setting: str, domain: str) -> str:
    if summary.empty:
        return f"No completed rows for {setting}/{domain}."
    selected = summary.loc[
        (summary["setting"] == setting) & (summary["domain"] == domain)
    ].copy()
    if selected.empty:
        return f"No completed rows for {setting}/{domain}."
    lines = [
        "| Method | Seeds | BA | AUROC | AUPRC | Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in selected.iterrows():
        n = int(row["n_seeds"])
        fields = []
        for metric in ("balanced_accuracy", "auroc", "auprc", "macro_f1"):
            fields.append(
                format_pm(row[f"{metric}_mean"], row[f"{metric}_std"], n)
            )
        lines.append(
            f"| {row['method']} | {n} | " + " | ".join(fields) + " |"
        )
    return "\n".join(lines)


def per_domain_ba_table(summary: pd.DataFrame) -> str:
    domains = ("brset", "mbrset", "odir", "pooled", "worst-source")
    if summary.empty:
        return "No completed three-source rows."
    selected = summary.loc[summary["setting"] == "three-source"].copy()
    if selected.empty:
        return "No completed three-source rows."
    lines = [
        "| Method | BRSET | mBRSET | ODIR-5K | Pooled | Worst source |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in selected["method"].drop_duplicates():
        fields = []
        method_rows = selected.loc[selected["method"] == method]
        for domain in domains:
            rows = method_rows.loc[method_rows["domain"] == domain]
            if rows.empty:
                fields.append("--")
                continue
            row = rows.iloc[0]
            fields.append(
                format_pm(
                    row["balanced_accuracy_mean"],
                    row["balanced_accuracy_std"],
                    int(row["n_seeds"]),
                )
            )
        lines.append(f"| {method} | " + " | ".join(fields) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    raw, missing = collect(args.results)
    summary = aggregate(raw)
    raw_path = args.output.with_name(args.output.stem + "_raw.csv")
    raw.to_csv(raw_path, index=False)
    summary.to_csv(args.output, index=False)

    report = ["# RealFed summary", ""]
    report.extend(
        [
            "## Three-source cross-silo, pooled test",
            "",
            markdown_table(summary, "three-source", "pooled"),
            "",
            "## Three-source balanced accuracy by source",
            "",
            per_domain_ba_table(summary),
            "",
            "## Held-out mBRSET domain shift, mBRSET test",
            "",
            markdown_table(summary, "held-out mBRSET", "mbrset"),
            "",
            f"Completed raw rows: {len(raw)}; missing cells/variants: {len(missing)}.",
        ]
    )
    if missing:
        report.extend(["", "Missing:", ""] + [f"- {item}" for item in missing])
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    json_path = args.output.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "missing": missing,
                "summary": summary.to_dict(orient="records"),
            },
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    print(f"raw={len(raw)} summary={len(summary)} missing={len(missing)}")
    print(markdown_path)


if __name__ == "__main__":
    main()
