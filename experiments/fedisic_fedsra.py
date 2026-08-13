#!/usr/bin/env python3
"""Fed-ISIC2019 natural-federation experiment for the FedSRA rebuttal (make-or-break).

Natural clients = the 6 FLamby acquisition centers (column ``center``, 0..5).
Task = 8-class skin-lesion classification (column ``label``, 0..7).
Images live inside the parquet as an HF Image feature: {'bytes', 'path'}.

This mirrors ``realfed_fundus.py`` (real independent CE clients with a SHARED
initialization -> legitimate one-shot FedAvg + CE ensembles) but upgraded to the
multi-class machinery of ``medmnist_fedsra.py`` (RGA aggregation + macro metrics).

Two independently trained methods, one shared data split:
  * fedsra: ImageNet-init ResNet-18 + fixed 8-way ETF, trained with joint_etf_loss;
            served by RGA (client-local moments / full-batch diagnostic / per-sample LN)
            over the K=6 client backbones  [O(K) tier].
  * ce:     ImageNet-init ResNet-18 + linear head, cross-entropy, SHARED init;
            served by one-shot FedAvg [single-model tier] and uniform / sqrt-size
            logit ensembles [O(K) tier].

Fair tiered comparison (see REBUTTAL_EXECUTION_SPEC.md 2):
  Tier-1 single model : FedSRA is not single-model here; compare O-FedAvg vs best-single.
  Tier-2 O(K)         : FedSRA-RGA  vs  CE uniform/sqrt ensemble  (same inference budget).

FAFI is a separate pipeline (adapt realfed_fafi.py next).

Usage (cluster), single seed first:
  python fedisic_fedsra.py --method both --seed 42 \
    --train_parquet realfed_data/Fed-ISIC2019/hf_flower/data/train-00000-of-00001.parquet \
    --test_parquet  realfed_data/Fed-ISIC2019/hf_flower/data/test-00000-of-00001.parquet \
    --output realfed_out/fedisic --epochs 30 --image_size 144
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

from medmnist_fedsra import (
    aggregate_logits,
    atomic_json_dump,
    atomic_torch_save,
    generate_etf,
    joint_etf_loss,
    seed_everything,
    state_to_fp16_cpu,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

N_CLASSES = 8
N_CENTERS = 6


@dataclass(frozen=True)
class Config:
    method: str
    seed: int
    epochs: int
    feature_dim: int
    image_size: int
    pretrained: bool
    snapshot_every: int = 0

    @property
    def tag(self) -> str:
        return f"fedisic_{self.method}_s{self.seed}"


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "standalone"


# ── data ───────────────────────────────────────────────────────────────────

class IsicParquetDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform) -> None:
        self.images = frame["image"].tolist()      # list of {'bytes','path'}
        self.labels = frame["label"].astype(int).to_numpy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        rec = self.images[index]
        with Image.open(io.BytesIO(rec["bytes"])) as image:
            image = image.convert("RGB")
        return self.transform(image), int(self.labels[index])


def build_transform(image_size: int, train: bool):
    if train:
        ops = [
            transforms.RandomResizedCrop(image_size, scale=(0.70, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.15, 0.15, 0.10, 0.03),
        ]
    else:
        ops = [transforms.Resize((image_size, image_size), antialias=True)]
    return transforms.Compose(
        ops + [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def make_loader(dataset: Dataset, batch_size: int, workers: int,
                train: bool = False, balanced: bool = False) -> DataLoader:
    sampler = None
    shuffle = train
    if balanced:
        labels = np.asarray(dataset.labels)
        counts = np.bincount(labels, minlength=N_CLASSES).clip(min=1)
        weights = 1.0 / counts[labels]
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double), len(labels), replacement=True
        )
        shuffle = False
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=train and len(dataset) >= batch_size,
    )


# ── models ─────────────────────────────────────────────────────────────────

class Backbone(nn.Module):
    """ImageNet-pretrained ResNet-18 encoder + projection to feature_dim."""

    def __init__(self, feature_dim: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        net.fc = nn.Identity()
        self.encoder = net
        self.projection = nn.Linear(512, feature_dim)

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.encoder(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.forward_raw(x), dim=1)


class CEModel(nn.Module):
    def __init__(self, feature_dim: int, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = Backbone(feature_dim, pretrained)
        self.classifier = nn.Linear(feature_dim, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone.forward_raw(x))


def average_models(models_in: Sequence[nn.Module], sample_counts: np.ndarray) -> nn.Module:
    """Sample-size-weighted one-shot FedAvg over identically initialized clients."""
    averaged = copy.deepcopy(models_in[0]).cpu()
    states = [m.cpu().state_dict() for m in models_in]
    weights = torch.as_tensor(sample_counts, dtype=torch.float64)
    weights = weights / weights.sum()
    merged = {}
    for key, first in states[0].items():
        if first.is_floating_point():
            merged[key] = sum(
                float(w) * s[key].float() for w, s in zip(weights, states)
            ).to(first.dtype)
        else:
            merged[key] = first.clone()
    averaged.load_state_dict(merged)
    return averaged


# ── metrics (multi-class, imbalance-aware) ───────────────────────────────────

def multiclass_metrics(logits: torch.Tensor, labels: np.ndarray) -> Dict[str, object]:
    logits = logits.float()
    pred = logits.argmax(1).numpy()
    probs = logits.softmax(1).numpy()
    per_class = []
    for c in range(N_CLASSES):
        mask = labels == c
        per_class.append(float((pred[mask] == c).mean()) if mask.any() else None)
    valid = [x for x in per_class if x is not None]
    try:
        auc = float(roc_auc_score(labels, probs, labels=np.arange(N_CLASSES),
                                  multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float((pred == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "macro_auc_ovr": auc,
        "worst_class_accuracy": min(valid) if valid else None,
        "per_class_accuracy": per_class,
        "n": int(len(labels)),
    }


# ── training ─────────────────────────────────────────────────────────────────

def optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _train_loop(model, loader, device, epochs, lr, loss_fn, checkpoint, tag, save_every, milestone_every=0):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    last = checkpoint.with_suffix(".last.pt")
    start_epoch = 0
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        optimizer_to(optimizer, device); scheduler.load_state_dict(saved["scheduler"])
        start_epoch = int(saved["epoch"])
        print(f"  {tag}: resume at {start_epoch}/{epochs}", flush=True)
    amp = device.type == "cuda"
    for epoch in range(start_epoch, epochs):
        model.train(); total = seen = 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                loss = loss_fn(model, x, y)
            loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(y); seen += len(y)
        scheduler.step()
        if epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"  {tag}: epoch {epoch+1}/{epochs} loss={total/max(seen,1):.4f}", flush=True)
        if (epoch + 1) % save_every == 0 and epoch + 1 < epochs:
            atomic_torch_save({"model": state_to_fp16_cpu(model.state_dict()),
                               "optimizer": optimizer.state_dict(),
                               "scheduler": scheduler.state_dict(), "epoch": epoch + 1}, last)
        # permanent milestone snapshot (model only) for the accuracy-vs-epoch curve
        if milestone_every and (epoch + 1) % milestone_every == 0 and epoch + 1 < epochs:
            mp = checkpoint.with_name(f"{checkpoint.stem}.ep{epoch+1}.pt")
            atomic_torch_save({"model": state_to_fp16_cpu(model.state_dict()), "epoch": epoch + 1}, mp)
    if last.exists():
        last.unlink()
    return model


def train_fedsra_client(cfg, rows, etf, checkpoint, device, batch_size, workers, lr, save_every):
    model = Backbone(cfg.feature_dim, pretrained=cfg.pretrained)
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        return model, saved["mu"].float(), saved["sd"].float(), saved["meta"]
    train_ds = IsicParquetDataset(rows, build_transform(cfg.image_size, True))
    loader = make_loader(train_ds, batch_size, workers, train=True, balanced=True)
    etf_dev = etf.to(device)
    t0 = time.time()
    model = _train_loop(model, loader, device, cfg.epochs, lr,
                        lambda m, x, y: joint_etf_loss(m.forward_raw(x), y, etf_dev, 0.1),
                        checkpoint, "fedsra", save_every, milestone_every=cfg.snapshot_every)
    # client-local uploaded feature moments (deployable RGA statistics)
    eval_ds = IsicParquetDataset(rows, build_transform(cfg.image_size, False))
    ml = make_loader(eval_ds, max(64, batch_size), workers)
    sum_x = sum_x2 = None; n = 0; model.eval()
    with torch.no_grad():
        for x, _ in ml:
            raw = model.forward_raw(x.to(device, non_blocking=True)).float().cpu()
            sum_x = raw.double().sum(0) if sum_x is None else sum_x + raw.double().sum(0)
            sum_x2 = raw.double().square().sum(0) if sum_x2 is None else sum_x2 + raw.double().square().sum(0)
            n += len(raw)
    mu = (sum_x / n).float()
    sd = (sum_x2 / n - (sum_x / n).square()).clamp_min(1e-8).sqrt().float()
    meta = {"n": int(len(rows)), "elapsed_s": time.time() - t0}
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save({"model": state_to_fp16_cpu(model.state_dict()),
                       "mu": mu.half(), "sd": sd.half(), "meta": meta}, checkpoint)
    return model.cpu(), mu, sd, meta


def train_ce_client(cfg, rows, checkpoint, device, batch_size, workers, lr, save_every, init_state):
    model = CEModel(cfg.feature_dim, pretrained=False)
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        return model, saved["meta"]
    if init_state is None:
        raise ValueError("CE clients require a shared initialization for one-shot FedAvg")
    model.load_state_dict(init_state)
    train_ds = IsicParquetDataset(rows, build_transform(cfg.image_size, True))
    loader = make_loader(train_ds, batch_size, workers, train=True, balanced=True)
    t0 = time.time()
    model = _train_loop(model, loader, device, cfg.epochs, lr,
                        lambda m, x, y: F.cross_entropy(m(x), y),
                        checkpoint, "ce", save_every, milestone_every=cfg.snapshot_every)
    meta = {"n": int(len(rows)), "elapsed_s": time.time() - t0}
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save({"model": state_to_fp16_cpu(model.state_dict()), "meta": meta}, checkpoint)
    return model.cpu(), meta


# ── evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def fedsra_logits(models_in, moments, sample_counts, etf, loader, device):
    raws = []; labels = None
    for i, model in enumerate(models_in):
        model.to(device).eval(); feats = []; ys = []
        for x, y in loader:
            feats.append(model.forward_raw(x.to(device, non_blocking=True)).float().cpu())
            if i == 0:
                ys.append(y)
        raws.append(torch.cat(feats))
        if i == 0:
            labels = torch.cat(ys).numpy()
        model.cpu()
    return aggregate_logits(raws, moments, np.sqrt(sample_counts), etf), labels


@torch.no_grad()
def ce_logits(models_in, sample_counts, loader, device):
    logits_all = []; labels = None
    for i, model in enumerate(models_in):
        model.to(device).eval(); outs = []; ys = []
        for x, y in loader:
            outs.append(model(x.to(device, non_blocking=True)).float().cpu())
            if i == 0:
                ys.append(y)
        logits_all.append(torch.cat(outs))
        if i == 0:
            labels = torch.cat(ys).numpy()
        model.cpu()
    w = np.sqrt(sample_counts); w = w / w.sum()
    result = {
        "uniform_logit_ensemble": torch.stack(logits_all).mean(0),
        "sqrt_weighted_logit_ensemble": sum(float(wi) * x for wi, x in zip(w, logits_all)),
    }
    averaged = average_models(models_in, sample_counts).to(device).eval()
    outs = []
    for x, _ in loader:
        outs.append(averaged(x.to(device, non_blocking=True)).float().cpu())
    result["one_shot_fedavg"] = torch.cat(outs)
    # best single client (diagnostic lower bound): report each client's own logits
    for i, lg in enumerate(logits_all):
        result[f"single_client_{i}"] = lg
    return result, labels


def per_center_audit(train_df: pd.DataFrame) -> Dict[str, object]:
    ct = pd.crosstab(train_df["center"], train_df["label"])
    audit = {"per_center_class_counts": {int(c): {int(k): int(v) for k, v in ct.loc[c].items()} for c in ct.index}}
    normH = {}
    for c in ct.index:
        p = ct.loc[c].to_numpy(float); p = p / max(p.sum(), 1)
        H = -sum(pi * np.log(pi) for pi in p if pi > 0)
        normH[int(c)] = float(H / np.log(N_CLASSES))
    audit["per_center_normalized_entropy"] = normH
    audit["mean_normalized_entropy"] = float(np.mean(list(normH.values())))
    return audit


def evaluate_method(method, trained, moments, sample_counts, etf, test_df, cfg, device, batch_size, workers):
    evaluations = {}
    domains = {"pooled": test_df}
    for c in sorted(test_df["center"].unique()):
        domains[f"center_{int(c)}"] = test_df[test_df["center"] == c]
    for domain, df in domains.items():
        if len(df) == 0:
            continue
        ds = IsicParquetDataset(df.reset_index(drop=True), build_transform(cfg.image_size, False))
        loader = make_loader(ds, max(64, batch_size), workers)
        if method == "fedsra":
            outputs, labels = fedsra_logits(trained, moments, np.asarray(sample_counts), etf, loader, device)
        else:
            outputs, labels = ce_logits(trained, np.asarray(sample_counts), loader, device)
        evaluations[domain] = {name: multiclass_metrics(lg, labels) for name, lg in outputs.items()}
    return evaluations


def run_method(method, cfg, train_df, test_df, etf, device, args):
    ckpt_dir = args.output / "checkpoints" / f"{method}_s{cfg.seed}"
    centers = sorted(train_df["center"].unique())
    init_state = None
    if method == "ce":
        seed_everything(cfg.seed)
        init_model = CEModel(cfg.feature_dim, pretrained=cfg.pretrained)
        init_state = {k: v.detach().cpu().clone() for k, v in init_model.state_dict().items()}
        del init_model
    trained = []; moments = []; metas = []; sample_counts = []
    for ci, center in enumerate(centers):
        seed_everything(cfg.seed * 1000 + ci + 17)
        rows = train_df[train_df["center"] == center].reset_index(drop=True)
        if args.limit_per_center > 0 and len(rows) > args.limit_per_center:
            rows = rows.sample(args.limit_per_center, random_state=cfg.seed).reset_index(drop=True)
        sample_counts.append(len(rows))
        ckpt = ckpt_dir / f"center_{int(center)}.pt"
        print(f"[{method}] center {int(center)} ({len(rows)} imgs)", flush=True)
        if method == "fedsra":
            model, mu, sd, meta = train_fedsra_client(cfg, rows, etf, ckpt, device, args.batch_size, args.workers, args.lr, args.save_every)
            moments.append((mu, sd))
        else:
            model, meta = train_ce_client(cfg, rows, ckpt, device, args.batch_size, args.workers, args.lr, args.save_every, init_state)
        trained.append(model); metas.append(meta)
    evaluations = evaluate_method(method, trained, moments, sample_counts, etf, test_df, cfg, device, args.batch_size, args.workers)
    return {"clients": [int(c) for c in centers], "sample_counts": sample_counts,
            "client_meta": metas, "evaluation": evaluations}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["fedsra", "ce", "both"], default="both")
    ap.add_argument("--train_parquet", type=Path, required=True)
    ap.add_argument("--test_parquet", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--feature_dim", type=int, default=256)
    ap.add_argument("--image_size", type=int, default=144)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--limit_per_center", type=int, default=0,
                    help="cap train images per center for a fast smoke (0 = no cap)")
    ap.add_argument("--pretrained", action="store_true",
                    help="use ImageNet-pretrained init (default: from-scratch, matches the paper)")
    ap.add_argument("--snapshot_every", type=int, default=0,
                    help="save a permanent per-center checkpoint every N epochs for the acc-vs-epoch curve")
    args = ap.parse_args()

    (args.output / "results").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"device={device}", flush=True)

    train_df = pd.read_parquet(args.train_parquet)
    test_df = pd.read_parquet(args.test_parquet)
    audit = per_center_audit(train_df)
    print(f"[audit] mean normalized entropy across centers = {audit['mean_normalized_entropy']:.3f}", flush=True)

    etf = generate_etf(N_CLASSES, args.feature_dim, 42)
    methods = ["ce", "fedsra"] if args.method == "both" else [args.method]
    t0 = time.time()
    per_method = {}
    for m in methods:
        cfg = Config(m, args.seed, args.epochs, args.feature_dim, args.image_size, args.pretrained, args.snapshot_every)
        seed_everything(cfg.seed)
        per_method[m] = run_method(m, cfg, train_df, test_df, etf, device, args)

    result = {
        "schema_version": 1,
        "dataset": "fed-isic2019",
        "n_classes": N_CLASSES, "n_centers": N_CENTERS,
        "seed": args.seed, "epochs": args.epochs, "image_size": args.image_size,
        "feature_dim": args.feature_dim,
        "code_revision": git_revision(),
        "natural_skew_audit": audit,
        "methods": per_method,
        "elapsed_s": time.time() - t0,
        "gpu_peak_mb": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
    }
    out_path = args.output / "results" / f"fedisic_{args.method}_s{args.seed}.json"
    atomic_json_dump(result, out_path)

    # ── console head-to-head on pooled test (the make-or-break read) ──
    print("\n" + "=" * 72)
    print(f"Fed-ISIC2019 pooled test  (seed={args.seed}, balanced-acc / macro-AUC / worst-class)")
    print("=" * 72)
    def show(method, key, label):
        try:
            m = per_method[method]["evaluation"]["pooled"][key]
        except KeyError:
            return
        wc = m["worst_class_accuracy"]
        print(f"  {label:<34s} BA={m['balanced_accuracy']*100:6.2f}  AUC={m['macro_auc_ovr']*100:6.2f}  worst={100*wc if wc is not None else float('nan'):6.2f}")
    if "ce" in per_method:
        print("-- single-model tier --")
        show("ce", "one_shot_fedavg", "O-FedAvg (real CE, shared init)")
        print("-- O(K) tier --")
        show("ce", "uniform_logit_ensemble", "CE ensemble (uniform)")
        show("ce", "sqrt_weighted_logit_ensemble", "CE ensemble (sqrt-n)")
    if "fedsra" in per_method:
        show("fedsra", "rga_client_local_moments", "FedSRA-RGA (client-local moments*)")
        show("fedsra", "rga_full_batch_diagnostic", "FedSRA-RGA (inference-batch)")
        show("fedsra", "rga_per_sample_layernorm", "FedSRA-RGA (per-sample LN)")
    print("  (* = deployable default;  FAFI to be added separately)")
    print(f"\nSaved: {out_path}  elapsed={(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
