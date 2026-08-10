#!/usr/bin/env python3
"""Validate RealFed result completeness and metric integrity before reporting."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SEEDS = (0, 42, 123)
DOMAINS = ("brset", "mbrset", "odir", "pooled")
METHODS = {
    "fedsra": (
        "rga_client_local_moments",
        "rga_full_batch_diagnostic",
        "rga_per_sample_layernorm",
    ),
    "ce": (
        "one_shot_fedavg",
        "uniform_logit_ensemble",
        "sqrt_weighted_logit_ensemble",
    ),
    "fafi": ("fafi_weighted_feature_ensemble",),
    "coboost": ("coboost_student",),
}
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "auroc",
    "auprc",
    "sensitivity",
    "specificity",
)


def expected_cells():
    for heldout in ("none", "mbrset"):
        for method, variants in METHODS.items():
            if heldout == "mbrset" and method == "coboost":
                continue
            for seed in SEEDS:
                yield method, heldout, seed, variants


def validate_cell(path: Path, method: str, heldout: str, seed: int, variants) -> list[str]:
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"{path.name}: invalid JSON: {exc}"]

    expected_tag = f"realfed_binary_{method}_heldout-{heldout}_s{seed}"
    if payload.get("tag") != expected_tag:
        errors.append(f"{path.name}: tag={payload.get('tag')!r}, expected {expected_tag!r}")
    cell = payload.get("cell", {})
    if cell.get("seed") != seed:
        errors.append(f"{path.name}: seed={cell.get('seed')!r}, expected {seed}")
    expected_clients = ["brset", "mbrset", "odir"]
    if heldout == "mbrset":
        expected_clients.remove("mbrset")
    if payload.get("clients") != expected_clients:
        errors.append(
            f"{path.name}: clients={payload.get('clients')!r}, expected {expected_clients!r}"
        )

    evaluation = payload.get("evaluation", {})
    for domain in DOMAINS:
        if domain not in evaluation:
            errors.append(f"{path.name}: missing domain {domain}")
            continue
        for variant in variants:
            metrics = evaluation[domain].get(variant)
            if metrics is None:
                errors.append(f"{path.name}: {domain} missing variant {variant}")
                continue
            for metric in METRICS:
                value = metrics.get(metric)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    errors.append(f"{path.name}: {domain}/{variant}/{metric} is {value!r}")
                elif not 0.0 <= float(value) <= 1.0:
                    errors.append(f"{path.name}: {domain}/{variant}/{metric}={value} outside [0,1]")
            if not isinstance(metrics.get("n"), int) or metrics["n"] <= 0:
                errors.append(f"{path.name}: {domain}/{variant}/n={metrics.get('n')!r}")

    for field in ("elapsed_s", "gpu_peak_mb", "worst_participating_domain_balanced_accuracy"):
        value = payload.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            errors.append(f"{path.name}: {field}={value!r}")
    if not payload.get("code_revision"):
        errors.append(f"{path.name}: missing code_revision")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--allow_incomplete", action="store_true")
    args = parser.parse_args()

    missing: list[str] = []
    errors: list[str] = []
    checked = 0
    for method, heldout, seed, variants in expected_cells():
        tag = f"realfed_binary_{method}_heldout-{heldout}_s{seed}"
        path = args.results / f"{tag}.json"
        if not path.exists():
            missing.append(path.name)
            continue
        checked += 1
        errors.extend(validate_cell(path, method, heldout, seed, variants))

    print(f"checked={checked}/21 missing={len(missing)} errors={len(errors)}")
    for item in missing:
        print(f"MISSING {item}")
    for item in errors:
        print(f"ERROR {item}")
    if errors or (missing and not args.allow_incomplete):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
