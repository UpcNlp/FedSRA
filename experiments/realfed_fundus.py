#!/usr/bin/env python3
"""Real-source cross-silo fundus experiment for the FedSRA rebuttal.

The three clinical sources are the clients: BRSET, mBRSET, and ODIR-5K.  The
shared single-label task is binary diabetes-related retinal-disease recognition. Precomputed
60/20/20 patient-disjoint train/validation/test splits are fixed across methods. Two independently
trained methods are supported:

* ``fedsra``: ImageNet-initialized ResNet-18 clients with a fixed binary ETF,
  followed by RGA and inference-only aggregation ablations.
* ``ce``: ImageNet-initialized ResNet-18 clients with cross entropy, followed by
  O-FedAvg and uniform/sqrt-size logit ensembles.

Every completed cell writes a result JSON and resumable per-client checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
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

SOURCES = {
    "brset": {
        "csv": "BRSET/dataset_multilabel_split.csv",
        "path": "view_0",
        "patient": "patient_id",
        "label": "final_icdr_binary",
    },
    "mbrset": {
        "csv": "mBRSET/dataset_multilabel_split.csv",
        "path": "file",
        "patient": "patient",
        "label": "final_icdr_binary",
    },
    "odir": {
        "csv": "ODIR-5K/dataset_multilabel_split.csv",
        "path": "file",
        "patient": "ID",
        "label": "final_icdr_binary",
    },
}


@dataclass(frozen=True)
class Config:
    method: str
    seed: int
    epochs: int
    feature_dim: int
    image_size: int
    heldout: str

    @property
    def clients(self) -> List[str]:
        return [x for x in SOURCES if x != self.heldout]

    @property
    def tag(self) -> str:
        h = self.heldout if self.heldout else "none"
        return f"realfed_binary_{self.method}_heldout-{h}_s{self.seed}"


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "standalone"


class FundusDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, transform) -> None:
        self.paths = rows["_path"].astype(str).tolist()
        self.labels = rows["_label"].astype(int).to_numpy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
        return self.transform(image), int(self.labels[index])


def build_transform(image_size: int, train: bool):
    if train:
        ops = [
            transforms.RandomResizedCrop(image_size, scale=(0.70, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(0.15, 0.15, 0.10, 0.03),
        ]
    else:
        ops = [transforms.Resize((image_size, image_size), antialias=True)]
    return transforms.Compose(
        ops
        + [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )


class FundusBackbone(nn.Module):
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
        self.backbone = FundusBackbone(feature_dim, pretrained)
        self.classifier = nn.Linear(feature_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone.forward_raw(x))


def load_sources(root: Path) -> Dict[str, pd.DataFrame]:
    result = {}
    for source, spec in SOURCES.items():
        csv_path = root / spec["csv"]
        frame = pd.read_csv(csv_path)
        frame = frame.loc[frame[spec["label"]].notna()].copy()
        base = csv_path.parent
        frame["_path"] = frame[spec["path"]].map(
            lambda x: str((base / str(x)).resolve())
        )
        frame["_label"] = frame[spec["label"]].astype(int)
        frame["_patient"] = frame[spec["patient"]].astype(str)
        if frame["_path"].duplicated().any():
            duplicate = frame.loc[frame["_path"].duplicated(), "_path"].iloc[0]
            raise ValueError(f"{source}: duplicate image path, e.g. {duplicate}")
        missing = [p for p in frame["_path"] if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(f"{source}: {len(missing)} missing images, e.g. {missing[0]}")
        split_patients = {
            s: set(frame.loc[frame["split"] == s, "_patient"])
            for s in ("train", "val", "test")
        }
        if any(
            split_patients[a] & split_patients[b]
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        ):
            raise ValueError(f"{source}: patient leakage across configured splits")
        result[source] = frame
    return result


def subset_rows(
    frame: pd.DataFrame, split: str, limit: int, seed: int
) -> pd.DataFrame:
    rows = frame.loc[frame["split"] == split].copy()
    if limit > 0 and len(rows) > limit:
        rows = rows.sample(limit, random_state=seed)
    return rows.reset_index(drop=True)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    train: bool = False,
    balanced: bool = False,
) -> DataLoader:
    sampler = None
    shuffle = train
    if balanced:
        labels = np.asarray(dataset.labels)
        counts = np.bincount(labels, minlength=2).clip(min=1)
        weights = 1.0 / counts[labels]
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double), len(labels), replacement=True
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=train and len(dataset) >= batch_size,
    )


def binary_metrics(logits: torch.Tensor, labels: np.ndarray) -> Dict[str, float]:
    probs = logits.float().softmax(1).numpy()[:, 1]
    pred = (probs >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(labels, probs)),
        "auprc": float(average_precision_score(labels, probs)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "n": int(len(labels)),
        "positive_n": int(labels.sum()),
    }


def average_models(
    models_in: Sequence[nn.Module], sample_counts: np.ndarray
) -> nn.Module:
    """Sample-size-weighted one-shot FedAvg over identically initialized clients."""
    averaged = copy.deepcopy(models_in[0]).cpu()
    states = [m.cpu().state_dict() for m in models_in]
    weights = torch.as_tensor(sample_counts, dtype=torch.float64)
    weights = weights / weights.sum()
    merged = {}
    for key, first in states[0].items():
        if first.is_floating_point():
            merged[key] = sum(
                float(weight) * state[key].float()
                for weight, state in zip(weights, states)
            ).to(first.dtype)
        else:
            merged[key] = first.clone()
    averaged.load_state_dict(merged)
    return averaged


def save_checkpoint(model: nn.Module, path: Path, meta: Dict[str, object]) -> None:
    atomic_torch_save({"model": state_to_fp16_cpu(model.state_dict()), "meta": meta}, path)


def optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_or_train_fedsra(
    source: str,
    cfg: Config,
    train_rows: pd.DataFrame,
    etf: torch.Tensor,
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
    lr: float,
    save_every: int,
) -> Tuple[FundusBackbone, torch.Tensor, torch.Tensor, Dict[str, object]]:
    model = FundusBackbone(cfg.feature_dim, pretrained=not checkpoint.exists())
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        return model, saved["mu"].float(), saved["sd"].float(), saved["meta"]
    model.to(device)
    train_ds = FundusDataset(train_rows, build_transform(cfg.image_size, True))
    eval_ds = FundusDataset(train_rows, build_transform(cfg.image_size, False))
    loader = make_loader(train_ds, batch_size, workers, train=True, balanced=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    last_checkpoint = checkpoint.with_suffix(".last.pt")
    start_epoch = 0
    if last_checkpoint.exists():
        saved = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        optimizer_to(optimizer, device)
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch = int(saved["epoch"])
        print(f"  {source}: resume at epoch {start_epoch}/{cfg.epochs}", flush=True)
    amp = device.type == "cuda"
    start = time.time()
    etf_dev = etf.to(device)
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        total = 0.0
        seen = 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                loss = joint_etf_loss(model.forward_raw(x), y, etf_dev, 0.1)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(y)
            seen += len(y)
        scheduler.step()
        if epoch == 0 or (epoch + 1) % max(1, cfg.epochs // 5) == 0:
            print(f"  {source}: epoch {epoch+1}/{cfg.epochs} loss={total/max(seen,1):.4f}", flush=True)
        if (epoch + 1) % save_every == 0 and epoch + 1 < cfg.epochs:
            atomic_torch_save(
                {
                    "model": state_to_fp16_cpu(model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                },
                last_checkpoint,
            )
    moment_loader = make_loader(eval_ds, max(64, batch_size), workers)
    sum_x = sum_x2 = None
    n = 0
    model.eval()
    with torch.no_grad():
        for x, _ in moment_loader:
            raw = model.forward_raw(x.to(device, non_blocking=True)).float().cpu()
            sum_x = raw.double().sum(0) if sum_x is None else sum_x + raw.double().sum(0)
            sum_x2 = raw.double().square().sum(0) if sum_x2 is None else sum_x2 + raw.double().square().sum(0)
            n += len(raw)
    mu = (sum_x / n).float()
    sd = (sum_x2 / n - (sum_x / n).square()).clamp_min(1e-8).sqrt().float()
    meta = {"source": source, "n": len(train_rows), "positive": int(train_rows["_label"].sum()), "elapsed_s": time.time()-start}
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save({"model": state_to_fp16_cpu(model.state_dict()), "mu": mu.half(), "sd": sd.half(), "meta": meta}, checkpoint)
    if last_checkpoint.exists():
        last_checkpoint.unlink()
    return model.cpu(), mu, sd, meta


def load_or_train_ce(
    source: str,
    cfg: Config,
    train_rows: pd.DataFrame,
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
    lr: float,
    save_every: int,
    initial_state: Optional[Dict[str, torch.Tensor]],
) -> Tuple[CEModel, Dict[str, object]]:
    model = CEModel(
        cfg.feature_dim,
        pretrained=not checkpoint.exists() and initial_state is None,
    )
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        return model, saved["meta"]
    if initial_state is None:
        raise ValueError("CE clients require a shared initialization for one-shot FedAvg")
    model.load_state_dict(initial_state)
    model.to(device)
    ds = FundusDataset(train_rows, build_transform(cfg.image_size, True))
    loader = make_loader(ds, batch_size, workers, train=True, balanced=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    last_checkpoint = checkpoint.with_suffix(".last.pt")
    start_epoch = 0
    if last_checkpoint.exists():
        saved = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        optimizer_to(optimizer, device)
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch = int(saved["epoch"])
        print(f"  {source}: resume at epoch {start_epoch}/{cfg.epochs}", flush=True)
    amp = device.type == "cuda"
    start = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        total = 0.0
        seen = 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(y)
            seen += len(y)
        scheduler.step()
        if epoch == 0 or (epoch + 1) % max(1, cfg.epochs // 5) == 0:
            print(f"  {source}: epoch {epoch+1}/{cfg.epochs} loss={total/max(seen,1):.4f}", flush=True)
        if (epoch + 1) % save_every == 0 and epoch + 1 < cfg.epochs:
            atomic_torch_save(
                {
                    "model": state_to_fp16_cpu(model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                },
                last_checkpoint,
            )
    meta = {"source": source, "n": len(train_rows), "positive": int(train_rows["_label"].sum()), "elapsed_s": time.time()-start}
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, checkpoint, meta)
    if last_checkpoint.exists():
        last_checkpoint.unlink()
    return model.cpu(), meta


@torch.no_grad()
def fedsra_logits(
    models_in: Sequence[FundusBackbone],
    moments: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    sample_counts: np.ndarray,
    etf: torch.Tensor,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
    raws = []
    labels = None
    for index, model in enumerate(models_in):
        model.to(device).eval()
        feats, ys = [], []
        for x, y in loader:
            feats.append(model.forward_raw(x.to(device, non_blocking=True)).float().cpu())
            if index == 0:
                ys.append(y)
        raws.append(torch.cat(feats))
        if index == 0:
            labels = torch.cat(ys).numpy()
        model.cpu()
    return aggregate_logits(raws, moments, np.sqrt(sample_counts), etf), labels


@torch.no_grad()
def ce_logits(
    models_in: Sequence[CEModel],
    sample_counts: np.ndarray,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
    logits_all = []
    labels = None
    for index, model in enumerate(models_in):
        model.to(device).eval()
        outs, ys = [], []
        for x, y in loader:
            outs.append(model(x.to(device, non_blocking=True)).float().cpu())
            if index == 0:
                ys.append(y)
        logits_all.append(torch.cat(outs))
        if index == 0:
            labels = torch.cat(ys).numpy()
        model.cpu()
    weights = np.sqrt(sample_counts)
    weights = weights / weights.sum()
    result = {
        "uniform_logit_ensemble": torch.stack(logits_all).mean(0),
        "sqrt_weighted_logit_ensemble": sum(float(w) * x for w, x in zip(weights, logits_all)),
    }
    averaged = average_models(models_in, sample_counts).to(device).eval()
    outs = []
    for x, _ in loader:
        outs.append(averaged(x.to(device, non_blocking=True)).float().cpu())
    result["one_shot_fedavg"] = torch.cat(outs)
    return result, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["fedsra", "ce"], required=True)
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--feature_dim", type=int, default=256)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--heldout", choices=["", *SOURCES], default="")
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_test", type=int, default=0)
    ap.add_argument("--validate_only", action="store_true")
    args = ap.parse_args()
    cfg = Config(args.method, args.seed, args.epochs, args.feature_dim, args.image_size, args.heldout)
    result_path = args.output / "results" / f"{cfg.tag}.json"
    if result_path.exists():
        print(f"SKIP {result_path}")
        return
    seed_everything(cfg.seed)
    frames = load_sources(args.data_root)
    audit = {}
    for source, frame in frames.items():
        audit[source] = {
            split: {"n": len(rows := subset_rows(frame, split, 0, cfg.seed)), "positive": int(rows["_label"].sum()), "patients": int(rows["_patient"].nunique())}
            for split in ("train", "val", "test")
        }
    if args.validate_only:
        print(json.dumps(audit, indent=2))
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    checkpoint_dir = args.output / "checkpoints" / cfg.tag
    etf = generate_etf(2, cfg.feature_dim, 42)
    ce_initial_state = None
    if cfg.method == "ce":
        # O-FedAvg is only well-defined when every local optimizer starts from
        # exactly the same model.  The per-client seeds below affect sampling
        # and augmentation, not the shared initialization.
        initial_model = CEModel(cfg.feature_dim, pretrained=True)
        ce_initial_state = {
            key: value.detach().cpu().clone()
            for key, value in initial_model.state_dict().items()
        }
        del initial_model
    trained = []
    moments = []
    metas = []
    sample_counts = []
    for client_index, source in enumerate(cfg.clients):
        seed_everything(cfg.seed * 1000 + client_index + 17)
        rows = subset_rows(frames[source], "train", args.limit_train, cfg.seed)
        sample_counts.append(len(rows))
        checkpoint = checkpoint_dir / f"{source}.pt"
        if cfg.method == "fedsra":
            model, mu, sd, meta = load_or_train_fedsra(source, cfg, rows, etf, checkpoint, device, args.batch_size, args.workers, args.lr, args.save_every)
            moments.append((mu, sd))
        else:
            model, meta = load_or_train_ce(
                source,
                cfg,
                rows,
                checkpoint,
                device,
                args.batch_size,
                args.workers,
                args.lr,
                args.save_every,
                ce_initial_state,
            )
        trained.append(model)
        metas.append(meta)
    evaluations = {}
    test_sets = {}
    for source, frame in frames.items():
        rows = subset_rows(frame, "test", args.limit_test, cfg.seed)
        test_sets[source] = FundusDataset(rows, build_transform(cfg.image_size, False))
    test_sets["pooled"] = ConcatDataset(list(test_sets.values()))
    for domain, ds in test_sets.items():
        loader = make_loader(ds, max(64, args.batch_size), args.workers)
        if cfg.method == "fedsra":
            outputs, labels = fedsra_logits(trained, moments, np.asarray(sample_counts), etf, loader, device)
        else:
            outputs, labels = ce_logits(trained, np.asarray(sample_counts), loader, device)
        evaluations[domain] = {name: binary_metrics(logits, labels) for name, logits in outputs.items()}
    primary = "rga_client_local_moments" if cfg.method == "fedsra" else "one_shot_fedavg"
    participating = cfg.clients
    worst_domain = min(evaluations[d][primary]["balanced_accuracy"] for d in participating)
    result = {
        "schema_version": 1,
        "cell": asdict(cfg),
        "tag": cfg.tag,
        "code_revision": git_revision(),
        "data_audit": audit,
        "clients": cfg.clients,
        "client_meta": metas,
        "evaluation": evaluations,
        "primary_method": primary,
        "worst_participating_domain_balanced_accuracy": worst_domain,
        "elapsed_s": time.time() - start,
        "gpu_peak_mb": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0,
    }
    atomic_json_dump(result, result_path)
    atomic_json_dump(
        {"status": "complete", "tag": cfg.tag, "completed_at": datetime.now(timezone.utc).isoformat(), "argv": sys.argv, "result": str(result_path)},
        args.output / "results" / f"{cfg.tag}.meta.json",
    )
    print(f"COMPLETE {cfg.tag}: pooled BA={evaluations['pooled'][primary]['balanced_accuracy']*100:.2f}% elapsed={(time.time()-start)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
