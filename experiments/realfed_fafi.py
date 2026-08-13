#!/usr/bin/env python3
"""FAFI (ICML 2025) on the three-source real fundus benchmark.

This is a domain adaptation of the authors' official implementation.  It keeps
FAFI's learnable-prototype model, four-term local objective, unweighted global
prototype average, and data-size-weighted feature ensemble.  The CIFAR-specific
encoder is replaced by the same ImageNet-initialized ResNet-18 used by the
FedSRA/CE real-data runs, and the local optimizer/budget are matched as well.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import models

from realfed_fundus import (
    SOURCES,
    FundusDataset,
    atomic_json_dump,
    atomic_torch_save,
    binary_metrics,
    build_transform,
    git_revision,
    load_sources,
    make_loader,
    optimizer_to,
    seed_everything,
    state_to_fp16_cpu,
    subset_rows,
)

# Import the objective directly from the published authors' code bundled in
# this repository; do not silently reimplement or simplify the baseline.
REPO_ROOT = Path(__file__).resolve().parents[1]
FAFI_ROOT = REPO_ROOT / "baselines" / "FAFI"
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
    heldout: str

    @property
    def clients(self) -> List[str]:
        return [source for source in SOURCES if source != self.heldout]

    @property
    def tag(self) -> str:
        heldout = self.heldout if self.heldout else "none"
        return f"realfed_binary_fafi_heldout-{heldout}_s{self.seed}"


class TwoViewFundusDataset(Dataset):
    """Return two independently augmented views, as required by FAFI."""

    def __init__(self, rows: pd.DataFrame, image_size: int) -> None:
        self.paths = rows["_path"].astype(str).tolist()
        self.labels = rows["_label"].astype(int).to_numpy()
        self.transform = build_transform(image_size, True)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            view1 = self.transform(image)
            view2 = self.transform(image)
        return view1, view2, int(self.labels[index])


class FAFIModel(nn.Module):
    """Official LearnableProtoResNet geometry with a 224px ResNet-18."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        encoder = models.resnet18(weights=weights)
        encoder.fc = nn.Identity()
        self.encoder = encoder
        self.learnable_proto = nn.Parameter(torch.randn(2, 512))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = F.normalize(self.encoder(x), p=2, dim=1, eps=1e-12)
        logits = feature @ self.learnable_proto.t()
        return logits, feature


def train_client(
    source: str,
    cfg: Config,
    train_rows: pd.DataFrame,
    shared_initial_state: Dict[str, torch.Tensor],
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
    lr: float,
    save_every: int,
) -> Tuple[FAFIModel, Dict[str, object]]:
    model = FAFIModel(pretrained=False)
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        return model, saved["meta"]
    model.load_state_dict(shared_initial_state)
    model.to(device)

    dataset = TwoViewFundusDataset(train_rows, cfg.image_size)
    loader = make_loader(dataset, batch_size, workers, train=True, balanced=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    ce_loss = nn.CrossEntropyLoss()
    supcon_loss = SupConLoss(temperature=0.07)
    proto_feature_loss = Contrastive_proto_feature_loss(temperature=1.0)
    proto_loss = Contrastive_proto_loss(temperature=1.0)

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
        for view1, view2, labels in loader:
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images = torch.cat([view1, view2], dim=0)
            doubled_labels = torch.cat([labels, labels], dim=0)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=amp
            ):
                logits, features = model(images)
                batch = len(labels)
                feature1, feature2 = torch.split(features, [batch, batch], dim=0)
                paired_features = torch.stack([feature1, feature2], dim=1)
                loss = (
                    ce_loss(logits, doubled_labels)
                    + supcon_loss(paired_features, labels)
                    + proto_feature_loss(
                        features, model.learnable_proto, doubled_labels
                    )
                    + proto_loss(model.learnable_proto)
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * batch
            seen += batch
        scheduler.step()
        if epoch == 0 or (epoch + 1) % max(1, cfg.epochs // 5) == 0:
            print(
                f"  {source}: epoch {epoch+1}/{cfg.epochs} "
                f"loss={total/max(seen, 1):.4f}",
                flush=True,
            )
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

    meta = {
        "source": source,
        "n": len(train_rows),
        "positive": int(train_rows["_label"].sum()),
        "elapsed_s": time.time() - start,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        {"model": state_to_fp16_cpu(model.state_dict()), "meta": meta}, checkpoint
    )
    if last_checkpoint.exists():
        last_checkpoint.unlink()
    return model.cpu(), meta


@torch.no_grad()
def fafi_logits(
    models_in: Sequence[FAFIModel],
    sample_counts: np.ndarray,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, np.ndarray]:
    weights = torch.as_tensor(sample_counts, dtype=torch.float32)
    weights = weights / weights.sum()
    global_proto = torch.stack(
        [model.learnable_proto.detach().float().cpu() for model in models_in]
    ).mean(0).to(device)

    all_features = []
    labels = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        fused = None
        for weight, model in zip(weights, models_in):
            model.to(device).eval()
            feature = model.encoder(x).float()
            fused = float(weight) * feature if fused is None else fused + float(weight) * feature
        all_features.append(F.normalize(fused, p=2, dim=1).cpu())
        labels.append(y)
    for model in models_in:
        model.cpu()
    features = torch.cat(all_features)
    logits = features @ global_proto.cpu().t()
    return logits, torch.cat(labels).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--heldout", choices=["", *SOURCES], default="")
    parser.add_argument("--limit_train", type=int, default=0)
    parser.add_argument("--limit_test", type=int, default=0)
    args = parser.parse_args()

    cfg = Config(args.seed, args.epochs, args.image_size, args.heldout)
    result_path = args.output / "results" / f"{cfg.tag}.json"
    if result_path.exists():
        print(f"SKIP {result_path}")
        return

    seed_everything(cfg.seed)
    frames = load_sources(args.data_root)
    audit = {
        source: {
            split: {
                "n": len(rows := subset_rows(frame, split, 0, cfg.seed)),
                "positive": int(rows["_label"].sum()),
                "patients": int(rows["_patient"].nunique()),
            }
            for split in ("train", "val", "test")
        }
        for source, frame in frames.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.time()

    initial_model = FAFIModel(pretrained=True)
    shared_initial_state = {
        key: value.detach().cpu().clone()
        for key, value in initial_model.state_dict().items()
    }
    del initial_model

    checkpoint_dir = args.output / "checkpoints" / cfg.tag
    trained = []
    client_meta = []
    sample_counts = []
    for client_index, source in enumerate(cfg.clients):
        seed_everything(cfg.seed * 1000 + client_index + 17)
        rows = subset_rows(frames[source], "train", args.limit_train, cfg.seed)
        sample_counts.append(len(rows))
        model, meta = train_client(
            source,
            cfg,
            rows,
            shared_initial_state,
            checkpoint_dir / f"{source}.pt",
            device,
            args.batch_size,
            args.workers,
            args.lr,
            args.save_every,
        )
        trained.append(model)
        client_meta.append(meta)

    test_sets: Dict[str, Dataset] = {}
    for source, frame in frames.items():
        rows = subset_rows(frame, "test", args.limit_test, cfg.seed)
        test_sets[source] = FundusDataset(
            rows, build_transform(cfg.image_size, False)
        )
    test_sets["pooled"] = ConcatDataset(list(test_sets.values()))

    evaluations = {}
    for domain, dataset in test_sets.items():
        loader = make_loader(dataset, max(64, args.batch_size), args.workers)
        logits, labels = fafi_logits(
            trained, np.asarray(sample_counts), loader, device
        )
        evaluations[domain] = {
            "fafi_weighted_feature_ensemble": binary_metrics(logits, labels)
        }

    primary = "fafi_weighted_feature_ensemble"
    worst_domain = min(
        evaluations[source][primary]["balanced_accuracy"]
        for source in cfg.clients
    )
    result = {
        "schema_version": 1,
        "method": "FAFI",
        "method_source": "official ICML 2025 objective and aggregation",
        "adaptation": "ImageNet ResNet-18 and matched local training budget",
        "cell": asdict(cfg),
        "tag": cfg.tag,
        "code_revision": git_revision(),
        "data_audit": audit,
        "clients": cfg.clients,
        "client_meta": client_meta,
        "evaluation": evaluations,
        "primary_method": primary,
        "worst_participating_domain_balanced_accuracy": worst_domain,
        "elapsed_s": time.time() - start,
        "gpu_peak_mb": (
            torch.cuda.max_memory_allocated() / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
    }
    atomic_json_dump(result, result_path)
    atomic_json_dump(
        {
            "status": "complete",
            "tag": cfg.tag,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "argv": sys.argv,
            "result": str(result_path),
        },
        args.output / "results" / f"{cfg.tag}.meta.json",
    )
    print(
        f"COMPLETE {cfg.tag}: pooled BA="
        f"{evaluations['pooled'][primary]['balanced_accuracy']*100:.2f}% "
        f"elapsed={(time.time()-start)/60:.1f}min",
        flush=True,
    )


if __name__ == "__main__":
    main()
