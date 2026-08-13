#!/usr/bin/env python3
"""FAFI (ICML 2025) on Fed-ISIC2019 natural federation (8-class, 6 centers).

Domain adaptation of the authors' official objective (learnable prototypes,
four-term local loss, unweighted global prototype average, size-weighted feature
ensemble), matched to the same from-scratch ResNet-18 and training budget used by
``fedisic_fedsra.py`` so the comparison is apples-to-apples. Data comes from the
FLamby parquet (image bytes + center + label), partitioned by the 6 real centers.

Usage:
  python fedisic_fafi.py --seed 42 --epochs 100 --image_size 144 \
    --train_parquet ... --test_parquet ... --output realfed_out/fedisic
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from medmnist_fedsra import (
    atomic_json_dump,
    atomic_torch_save,
    seed_everything,
    state_to_fp16_cpu,
)
from fedisic_fedsra import (
    N_CLASSES,
    N_CENTERS,
    IsicParquetDataset,
    build_transform,
    git_revision,
    make_loader,
    multiclass_metrics,
    optimizer_to,
    per_center_audit,
)

# Official authors' objective, bundled in the repo (do not reimplement).
REPO_ROOT = Path(__file__).resolve().parents[2]
FAFI_ROOT = REPO_ROOT / "FAFI_ICML25-master-orgin"
sys.path.insert(0, str(FAFI_ROOT))
from oneshot_algorithms.ours.unsupervised_loss import (  # noqa: E402
    Contrastive_proto_feature_loss,
    Contrastive_proto_loss,
    SupConLoss,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass(frozen=True)
class Config:
    seed: int
    epochs: int
    image_size: int
    pretrained: bool

    @property
    def tag(self) -> str:
        return f"fedisic_fafi_s{self.seed}"


class TwoViewIsicDataset(Dataset):
    """Two independently augmented views per image (required by FAFI)."""

    def __init__(self, frame: pd.DataFrame, image_size: int) -> None:
        self.images = frame["image"].tolist()
        self.labels = frame["label"].astype(int).to_numpy()
        self.transform = build_transform(image_size, True)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        rec = self.images[index]
        with Image.open(io.BytesIO(rec["bytes"])) as image:
            image = image.convert("RGB")
            view1 = self.transform(image)
            view2 = self.transform(image)
        return view1, view2, int(self.labels[index])


class FAFIModel(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        encoder = models.resnet18(weights=weights)
        encoder.fc = nn.Identity()
        self.encoder = encoder
        self.learnable_proto = nn.Parameter(torch.randn(N_CLASSES, 512))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = F.normalize(self.encoder(x), p=2, dim=1, eps=1e-12)
        logits = feature @ self.learnable_proto.t()
        return logits, feature


def train_client(cfg, rows, shared_init, checkpoint, device, batch_size, workers, lr, save_every):
    model = FAFIModel(pretrained=False)
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        return model, saved["meta"]
    model.load_state_dict(shared_init)
    model.to(device)
    loader = make_loader(TwoViewIsicDataset(rows, cfg.image_size), batch_size, workers, train=True, balanced=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    ce_loss = nn.CrossEntropyLoss()
    supcon = SupConLoss(temperature=0.07)
    proto_feat = Contrastive_proto_feature_loss(temperature=1.0)
    proto_reg = Contrastive_proto_loss(temperature=1.0)

    last = checkpoint.with_suffix(".last.pt")
    start_epoch = 0
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        optimizer_to(optimizer, device); scheduler.load_state_dict(saved["scheduler"])
        start_epoch = int(saved["epoch"])
        print(f"  fafi: resume {start_epoch}/{cfg.epochs}", flush=True)
    amp = device.type == "cuda"
    t0 = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        model.train(); total = seen = 0
        for view1, view2, labels in loader:
            view1 = view1.to(device, non_blocking=True); view2 = view2.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images = torch.cat([view1, view2], 0); doubled = torch.cat([labels, labels], 0)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                logits, features = model(images)
                b = len(labels)
                f1, f2 = torch.split(features, [b, b], 0)
                paired = torch.stack([f1, f2], 1)
                loss = (ce_loss(logits, doubled) + supcon(paired, labels)
                        + proto_feat(features, model.learnable_proto, doubled)
                        + proto_reg(model.learnable_proto))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * b; seen += b
        scheduler.step()
        if epoch == 0 or (epoch + 1) % max(1, cfg.epochs // 5) == 0:
            print(f"  fafi: epoch {epoch+1}/{cfg.epochs} loss={total/max(seen,1):.4f}", flush=True)
        if (epoch + 1) % save_every == 0 and epoch + 1 < cfg.epochs:
            atomic_torch_save({"model": state_to_fp16_cpu(model.state_dict()),
                               "optimizer": optimizer.state_dict(),
                               "scheduler": scheduler.state_dict(), "epoch": epoch + 1}, last)
    meta = {"n": int(len(rows)), "elapsed_s": time.time() - t0}
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save({"model": state_to_fp16_cpu(model.state_dict()), "meta": meta}, checkpoint)
    if last.exists():
        last.unlink()
    return model.cpu(), meta


@torch.no_grad()
def fafi_logits(models_in, sample_counts, loader, device):
    w = torch.as_tensor(sample_counts, dtype=torch.float32); w = w / w.sum()
    global_proto = torch.stack([m.learnable_proto.detach().float().cpu() for m in models_in]).mean(0).to(device)
    feats = []; labels = []
    for x, y in loader:
        x = x.to(device, non_blocking=True); fused = None
        for wi, m in zip(w, models_in):
            m.to(device).eval()
            f = m.encoder(x).float()
            fused = float(wi) * f if fused is None else fused + float(wi) * f
        feats.append(F.normalize(fused, p=2, dim=1).cpu()); labels.append(y)
    for m in models_in:
        m.cpu()
    logits = torch.cat(feats) @ global_proto.cpu().t()
    return logits, torch.cat(labels).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_parquet", type=Path, required=True)
    ap.add_argument("--test_parquet", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--image_size", type=int, default=144)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save_every", type=int, default=20)
    ap.add_argument("--limit_per_center", type=int, default=0)
    ap.add_argument("--pretrained", action="store_true")
    args = ap.parse_args()

    cfg = Config(args.seed, args.epochs, args.image_size, args.pretrained)
    (args.output / "results").mkdir(parents=True, exist_ok=True)
    out_path = args.output / "results" / f"{cfg.tag}.json"
    if out_path.exists():
        print(f"SKIP {out_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    seed_everything(cfg.seed)
    train_df = pd.read_parquet(args.train_parquet)
    test_df = pd.read_parquet(args.test_parquet)
    audit = per_center_audit(train_df)
    print(f"[audit] mean normalized entropy = {audit['mean_normalized_entropy']:.3f}", flush=True)

    seed_everything(cfg.seed)
    init_model = FAFIModel(pretrained=cfg.pretrained)
    shared_init = {k: v.detach().cpu().clone() for k, v in init_model.state_dict().items()}
    del init_model

    ckpt_dir = args.output / "checkpoints" / cfg.tag
    centers = sorted(train_df["center"].unique())
    trained = []; metas = []; sample_counts = []
    t0 = time.time()
    for ci, center in enumerate(centers):
        seed_everything(cfg.seed * 1000 + ci + 17)
        rows = train_df[train_df["center"] == center].reset_index(drop=True)
        if args.limit_per_center > 0 and len(rows) > args.limit_per_center:
            rows = rows.sample(args.limit_per_center, random_state=cfg.seed).reset_index(drop=True)
        sample_counts.append(len(rows))
        print(f"[fafi] center {int(center)} ({len(rows)} imgs)", flush=True)
        model, meta = train_client(cfg, rows, shared_init, ckpt_dir / f"center_{int(center)}.pt",
                                   device, args.batch_size, args.workers, args.lr, args.save_every)
        trained.append(model); metas.append(meta)

    evaluations = {}
    domains = {"pooled": test_df}
    for c in sorted(test_df["center"].unique()):
        domains[f"center_{int(c)}"] = test_df[test_df["center"] == c]
    for domain, df in domains.items():
        if len(df) == 0:
            continue
        ds = IsicParquetDataset(df.reset_index(drop=True), build_transform(cfg.image_size, False))
        loader = make_loader(ds, max(64, args.batch_size), args.workers)
        logits, labels = fafi_logits(trained, np.asarray(sample_counts), loader, device)
        evaluations[domain] = {"fafi_weighted_feature_ensemble": multiclass_metrics(logits, labels)}

    result = {
        "schema_version": 1, "dataset": "fed-isic2019", "method": "FAFI",
        "method_source": "official ICML 2025 objective and aggregation",
        "adaptation": "from-scratch ResNet-18, matched budget",
        "n_classes": N_CLASSES, "n_centers": N_CENTERS,
        "cell": asdict(cfg), "tag": cfg.tag, "code_revision": git_revision(),
        "natural_skew_audit": audit, "clients": [int(c) for c in centers],
        "sample_counts": sample_counts, "client_meta": metas,
        "evaluation": evaluations, "primary_method": "fafi_weighted_feature_ensemble",
        "elapsed_s": time.time() - t0,
        "gpu_peak_mb": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0,
        "completed_at": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
    }
    atomic_json_dump(result, out_path)
    m = evaluations["pooled"]["fafi_weighted_feature_ensemble"]
    print(f"COMPLETE {cfg.tag}: pooled BA={m['balanced_accuracy']*100:.2f}% "
          f"AUC={m['macro_auc_ovr']*100:.2f}% elapsed={(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
