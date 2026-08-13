#!/usr/bin/env python3
"""Offline accuracy-vs-epoch curve for Fed-ISIC2019 from milestone checkpoints.

Given per-center checkpoints saved by fedisic_fedsra.py at several epochs
(``center_{c}.ep{E}.pt`` for E<epochs and ``center_{c}.pt`` for the final epoch),
this rebuilds the K=6 federation at each milestone and evaluates on the pooled
test set -- so one training run yields the whole convergence curve and we can
pick the right epoch empirically instead of guessing.

For ``fedsra`` the client-local feature moments are recomputed per milestone from
that center's train images (cheap eval pass). For ``ce`` only the models are
needed.

Usage:
  python fedisic_eval_curve.py --method fedsra --seed 42 --epochs 20,40,60,80,100 \
    --train_parquet ... --test_parquet ... --output realfed_out/fedisic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from medmnist_fedsra import atomic_json_dump, generate_etf, seed_everything
from fedisic_fedsra import (
    N_CLASSES,
    Backbone,
    CEModel,
    IsicParquetDataset,
    build_transform,
    ce_logits,
    fedsra_logits,
    make_loader,
    multiclass_metrics,
)


def ckpt_path(ckpt_dir: Path, center: int, epoch: int, final_epoch: int) -> Path:
    if epoch == final_epoch:
        return ckpt_dir / f"center_{center}.pt"
    return ckpt_dir / f"center_{center}.ep{epoch}.pt"


@torch.no_grad()
def compute_moments(model, loader, device):
    sum_x = sum_x2 = None
    n = 0
    model.to(device).eval()
    for x, _ in loader:
        raw = model.forward_raw(x.to(device, non_blocking=True)).float().cpu()
        sum_x = raw.double().sum(0) if sum_x is None else sum_x + raw.double().sum(0)
        sum_x2 = raw.double().square().sum(0) if sum_x2 is None else sum_x2 + raw.double().square().sum(0)
        n += len(raw)
    model.cpu()
    mu = (sum_x / n).float()
    sd = (sum_x2 / n - (sum_x / n).square()).clamp_min(1e-8).sqrt().float()
    return mu, sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["fedsra", "ce"], required=True)
    ap.add_argument("--train_parquet", type=Path, required=True)
    ap.add_argument("--test_parquet", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=str, required=True, help="comma list, e.g. 20,40,60,80,100")
    ap.add_argument("--final_epoch", type=int, default=100)
    ap.add_argument("--image_size", type=int, default=144)
    ap.add_argument("--feature_dim", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    epochs = [int(e) for e in args.epochs.split(",")]
    ckpt_dir = args.output / "checkpoints" / f"{args.method}_s{args.seed}"
    train_df = pd.read_parquet(args.train_parquet)
    test_df = pd.read_parquet(args.test_parquet)
    centers = sorted(train_df["center"].unique())
    sample_counts = [int((train_df["center"] == c).sum()) for c in centers]
    etf = generate_etf(N_CLASSES, args.feature_dim, 42)

    test_ds = IsicParquetDataset(test_df.reset_index(drop=True), build_transform(args.image_size, False))
    test_loader = make_loader(test_ds, max(64, args.batch_size), args.workers)

    curve = {}
    for E in epochs:
        paths = [ckpt_path(ckpt_dir, int(c), E, args.final_epoch) for c in centers]
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"[epoch {E}] SKIP - missing {len(missing)} ckpt(s), e.g. {missing[0].name}", flush=True)
            continue
        models_in = []
        moments = []
        for c, p in zip(centers, paths):
            saved = torch.load(p, map_location="cpu", weights_only=False)
            if args.method == "fedsra":
                m = Backbone(args.feature_dim, pretrained=False)
                m.load_state_dict(saved["model"])
                rows = train_df[train_df["center"] == c].reset_index(drop=True)
                ml = make_loader(IsicParquetDataset(rows, build_transform(args.image_size, False)),
                                 max(64, args.batch_size), args.workers)
                moments.append(compute_moments(m, ml, device))
            else:
                m = CEModel(args.feature_dim, pretrained=False)
                m.load_state_dict(saved["model"])
            models_in.append(m)
        if args.method == "fedsra":
            outputs, labels = fedsra_logits(models_in, moments, np.asarray(sample_counts), etf, test_loader, device)
        else:
            outputs, labels = ce_logits(models_in, np.asarray(sample_counts), test_loader, device)
        curve[E] = {name: multiclass_metrics(lg, labels) for name, lg in outputs.items()}
        # concise console line for the key methods
        key = "rga_client_local_moments" if args.method == "fedsra" else "uniform_logit_ensemble"
        m = curve[E].get(key, {})
        print(f"[epoch {E:3d}] {key}: BA={m.get('balanced_accuracy',0)*100:6.2f}  AUC={m.get('macro_auc_ovr',0)*100:6.2f}", flush=True)

    out = args.output / "results" / f"curve_{args.method}_s{args.seed}.json"
    atomic_json_dump({"method": args.method, "seed": args.seed, "epochs": epochs, "curve": curve}, out)
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
