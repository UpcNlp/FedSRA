#!/usr/bin/env python3
"""Fresh reviewer-response experiments for FedSRA on MedMNIST.

The script is intentionally self-contained so it can be copied to a clean directory on
each RTX 5090 host. It trains independent ETF-anchored ResNet-18 client backbones, saves
per-client resumable checkpoints, uploads client-local feature moments, and evaluates
several server aggregation rules without using validation/test labels for calibration.

The optional corruption evaluation implements the common PIL-only subset of the official
MedMNIST-C registry (pixelation, JPEG compression, brightness, and contrast) with the exact
severity parameters published by the MedMNIST-C project:
https://github.com/francescodisalvo05/medmnistc-api
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


DATASET_INFO = {
    "pathmnist": {"classes": 9, "rgb": True},
    "organamnist": {"classes": 11, "rgb": False},
}

# Exact common-subset values from medmnistc/corruptions/registry.py.
MEDMNISTC_COMMON = {
    "pathmnist": {
        "pixelate": [0.80, 0.60, 0.40, 0.30, 0.25],
        "jpeg_compression": [50, 30, 15, 10, 7],
        "brightness_up": [1.10, 1.15, 1.20, 1.22, 1.25],
        "brightness_down": [0.85, 0.80, 0.75, 0.72, 0.70],
        "contrast_up": [1.10, 1.20, 1.30, 1.40, 1.60],
        "contrast_down": [0.80, 0.70, 0.60, 0.55, 0.50],
    },
    "organamnist": {
        "pixelate": [0.70, 0.60, 0.50, 0.40, 0.35],
        "jpeg_compression": [50, 30, 15, 10, 7],
        "brightness_up": [1.20, 1.30, 1.40, 1.50, 1.60],
        "brightness_down": [0.80, 0.75, 0.70, 0.65, 0.60],
        "contrast_up": [1.30, 1.40, 1.60, 1.70, 1.80],
        "contrast_down": [0.80, 0.70, 0.60, 0.55, 0.50],
    },
}


@dataclass(frozen=True)
class CellConfig:
    dataset: str
    alpha: float
    n_clients: int
    noise_rate: float
    seed: int
    epochs: int
    feature_dim: int

    @property
    def tag(self) -> str:
        a = str(self.alpha).replace(".", "p")
        q = str(self.noise_rate).replace(".", "p")
        return f"{self.dataset}_a{a}_k{self.n_clients}_noise{q}_s{self.seed}"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json_dump(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch_save(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def file_sha256(path: Path, block_size: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return os.environ.get("FED_SRA_CODE_REVISION", "standalone")


class MedArrayDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: Optional[Sequence[int]],
        transform,
        corruptor=None,
    ) -> None:
        self.images = images
        self.labels = np.asarray(labels).reshape(-1).astype(np.int64)
        self.indices = np.arange(len(self.labels)) if indices is None else np.asarray(indices)
        self.transform = transform
        self.corruptor = corruptor

    def __len__(self) -> int:
        return int(len(self.indices))

    @staticmethod
    def _pil(arr: np.ndarray) -> Image.Image:
        if arr.ndim == 2:
            return Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
        if arr.ndim == 3 and arr.shape[-1] == 1:
            return Image.fromarray(arr[..., 0].astype(np.uint8), mode="L").convert("RGB")
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        img = self._pil(self.images[idx])
        if self.corruptor is not None:
            img = self.corruptor(img)
        return self.transform(img), int(self.labels[idx])


def build_transform(dataset: str, train: bool):
    if train and dataset == "pathmnist":
        aug = [
            transforms.Resize((36, 36), antialias=True),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05),
        ]
    elif train:
        # Avoid flips for axial organ slices: they can change laterality semantics.
        aug = [
            transforms.Resize((32, 32), antialias=True),
            transforms.RandomRotation(8),
        ]
    else:
        aug = [transforms.Resize((32, 32), antialias=True)]
    return transforms.Compose(
        aug
        + [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


class RegistryCorruptor:
    def __init__(self, dataset: str, name: str, severity: int):
        if name not in MEDMNISTC_COMMON[dataset]:
            raise ValueError(f"Unsupported corruption {name!r} for {dataset}")
        if severity not in (1, 2, 3, 4, 5):
            raise ValueError("severity must be in 1..5")
        self.name = name
        self.value = MEDMNISTC_COMMON[dataset][name][severity - 1]

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.name == "pixelate":
            w, h = img.size
            rw = max(1, int(w * self.value))
            rh = max(1, int(h * self.value))
            return img.resize((rw, rh), Image.Resampling.BOX).resize(
                (w, h), Image.Resampling.BOX
            )
        if self.name == "jpeg_compression":
            buf = BytesIO()
            img.save(buf, "JPEG", quality=int(self.value))
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        if self.name.startswith("brightness_"):
            return ImageEnhance.Brightness(img).enhance(float(self.value))
        if self.name.startswith("contrast_"):
            return ImageEnhance.Contrast(img).enhance(float(self.value))
        raise AssertionError(self.name)


def dirichlet_partition(
    labels: np.ndarray,
    n_clients: int,
    alpha: float,
    n_classes: int,
    seed: int,
    min_client_size: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    labels = np.asarray(labels).reshape(-1)
    for attempt in range(200):
        rng = np.random.RandomState(seed + attempt * 1009)
        clients: List[List[int]] = [[] for _ in range(n_clients)]
        counts = np.zeros((n_clients, n_classes), dtype=np.int64)
        for c in range(n_classes):
            idx = np.flatnonzero(labels == c)
            rng.shuffle(idx)
            props = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(props)[:-1] * len(idx)).astype(int)
            splits = np.split(idx, cuts)
            for k, part in enumerate(splits):
                clients[k].extend(part.tolist())
                counts[k, c] = len(part)
        sizes = np.asarray([len(x) for x in clients])
        if len(sizes) > 0 and sizes.min() >= min_client_size:
            out = []
            for x in clients:
                a = np.asarray(x, dtype=np.int64)
                rng.shuffle(a)
                out.append(a)
            return out, counts
    raise RuntimeError(
        f"Could not obtain min client size {min_client_size} after 200 Dirichlet draws"
    )


def inject_symmetric_noise(
    labels: np.ndarray,
    client_indices: Sequence[np.ndarray],
    rate: float,
    n_classes: int,
    seed: int,
) -> Tuple[np.ndarray, List[int]]:
    noisy = np.asarray(labels).reshape(-1).astype(np.int64).copy()
    changed: List[int] = []
    if rate <= 0:
        return noisy, [0 for _ in client_indices]
    for k, idx in enumerate(client_indices):
        rng = np.random.RandomState(seed + 7919 * (k + 1))
        n = int(round(rate * len(idx)))
        picked = rng.choice(idx, size=n, replace=False)
        old = noisy[picked]
        offset = rng.randint(1, n_classes, size=n)
        noisy[picked] = (old + offset) % n_classes
        changed.append(int(n))
    return noisy, changed


def generate_etf(n_classes: int, feature_dim: int, seed: int = 42) -> torch.Tensor:
    if feature_dim < n_classes:
        raise ValueError("feature_dim must be >= n_classes")
    g = torch.Generator().manual_seed(seed)
    m = math.sqrt(n_classes / (n_classes - 1)) * (
        torch.eye(n_classes) - torch.ones(n_classes, n_classes) / n_classes
    )
    if feature_dim > n_classes:
        q, _ = torch.linalg.qr(torch.randn(feature_dim, n_classes, generator=g))
        m = m @ q.T
    return F.normalize(m, dim=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class ResNet18Backbone(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._layer(64, 64, 2, 1)
        self.layer2 = self._layer(64, 128, 2, 2)
        self.layer3 = self._layer(128, 256, 2, 2)
        self.layer4 = self._layer(256, 512, 2, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, feature_dim)

    @staticmethod
    def _layer(in_channels: int, channels: int, blocks: int, stride: int):
        layers = [BasicBlock(in_channels, channels, stride)]
        layers.extend(BasicBlock(channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.forward_raw(x), dim=1)


def joint_etf_loss(
    raw: torch.Tensor, labels: torch.Tensor, etf: torch.Tensor, temperature: float
) -> torch.Tensor:
    features = F.normalize(raw, dim=1)
    logits = features @ etf.T / temperature
    proto_ce = F.cross_entropy(logits, labels)
    alignment = (1.0 - (features * etf[labels]).sum(1)).mean()

    contrastive = raw.new_zeros(())
    batch = len(labels)
    if batch > 1:
        eye = torch.eye(batch, device=labels.device, dtype=torch.bool)
        same = labels[:, None].eq(labels[None, :]) & ~eye
        valid = same.any(1)
        if valid.any():
            sim = features @ features.T / temperature
            sim = sim - sim.max(1, keepdim=True).values.detach()
            exp_sim = torch.exp(sim).masked_fill(eye, 0.0)
            log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
            contrastive = -(
                (same.float() * log_prob).sum(1)[valid]
                / same.sum(1)[valid].float()
            ).mean()
    return proto_ce + 0.5 * contrastive + 0.5 * alignment


def state_to_fp16_cpu(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in state.items():
        value = value.detach().cpu()
        out[key] = value.half() if value.is_floating_point() else value
    return out


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def local_feature_moments(
    model: ResNet18Backbone, loader: DataLoader, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    model.eval()
    total = None
    total_sq = None
    n = 0
    for x, _ in loader:
        raw = model.forward_raw(x.to(device, non_blocking=True)).float().cpu()
        if total is None:
            total = raw.double().sum(0)
            total_sq = raw.double().square().sum(0)
        else:
            total += raw.double().sum(0)
            total_sq += raw.double().square().sum(0)
        n += len(raw)
    if n == 0:
        raise RuntimeError("Cannot compute feature moments from an empty client")
    mu = total / n
    var = (total_sq / n - mu.square()).clamp_min(1e-8)
    return mu.float(), var.sqrt().float(), n


def train_or_load_client(
    client_id: int,
    cfg: CellConfig,
    images: np.ndarray,
    noisy_labels: np.ndarray,
    clean_labels: np.ndarray,
    indices: np.ndarray,
    etf: torch.Tensor,
    cell_dir: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
    lr: float,
    save_every: int,
) -> Tuple[ResNet18Backbone, torch.Tensor, torch.Tensor, Dict[str, object]]:
    final_path = cell_dir / f"client_{client_id}.pt"
    last_path = cell_dir / f"client_{client_id}.last.pt"
    model = ResNet18Backbone(cfg.feature_dim).to(device)

    if final_path.exists():
        saved = torch.load(final_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        return model, saved["moment_mu"].float(), saved["moment_sd"].float(), saved["meta"]

    train_ds = MedArrayDataset(
        images, noisy_labels, indices, build_transform(cfg.dataset, train=True)
    )
    moment_ds = MedArrayDataset(
        images, clean_labels, indices, build_transform(cfg.dataset, train=False)
    )
    train_loader = make_loader(train_ds, batch_size, True, workers)
    moment_loader = make_loader(moment_ds, max(batch_size, 512), False, workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    start_epoch = 0
    elapsed_before = 0.0
    if last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch = int(saved["epoch"])
        elapsed_before = float(saved.get("elapsed_s", 0.0))
        print(f"  client {client_id}: resume at epoch {start_epoch}/{cfg.epochs}", flush=True)

    etf_dev = etf.to(device)
    amp_enabled = device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    t0 = time.time()
    model.train()
    last_loss = float("nan")
    for epoch in range(start_epoch, cfg.epochs):
        running = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
            ):
                raw = model.forward_raw(x)
                loss = joint_etf_loss(raw, y, etf_dev, temperature=0.1)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(y)
            seen += len(y)
        scheduler.step()
        last_loss = running / max(seen, 1)
        ep = epoch + 1
        if ep == 1 or ep % max(1, cfg.epochs // 5) == 0 or ep == cfg.epochs:
            print(
                f"  client {client_id}: epoch {ep:03d}/{cfg.epochs} loss={last_loss:.5f}",
                flush=True,
            )
        if ep % save_every == 0 or ep == cfg.epochs:
            atomic_torch_save(
                {
                    "model": state_to_fp16_cpu(model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": ep,
                    "elapsed_s": elapsed_before + time.time() - t0,
                },
                last_path,
            )

    mu, sd, moment_n = local_feature_moments(model, moment_loader, device)
    classes, class_counts = np.unique(clean_labels[indices], return_counts=True)
    meta = {
        "client_id": client_id,
        "n_samples": int(len(indices)),
        "classes": [int(x) for x in classes],
        "class_counts": {str(int(c)): int(n) for c, n in zip(classes, class_counts)},
        "epochs": cfg.epochs,
        "last_loss": last_loss,
        "train_elapsed_s": elapsed_before + time.time() - t0,
        "moment_n": int(moment_n),
    }
    atomic_torch_save(
        {
            "model": state_to_fp16_cpu(model.state_dict()),
            "moment_mu": mu.half(),
            "moment_sd": sd.half(),
            "meta": meta,
        },
        final_path,
    )
    if last_path.exists():
        last_path.unlink()
    return model, mu, sd, meta


@torch.no_grad()
def extract_all_raw(
    models: Sequence[ResNet18Backbone],
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[torch.Tensor], np.ndarray]:
    all_raw: List[torch.Tensor] = []
    labels_out: Optional[np.ndarray] = None
    for k, model in enumerate(models):
        model.to(device).eval()
        feats: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        for x, y in loader:
            feats.append(model.forward_raw(x.to(device, non_blocking=True)).float().cpu())
            if k == 0:
                labels.append(y.cpu())
        all_raw.append(torch.cat(feats, 0))
        if k == 0:
            labels_out = torch.cat(labels).numpy()
        model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assert labels_out is not None
    return all_raw, labels_out


def aggregate_logits(
    all_raw: Sequence[torch.Tensor],
    moments: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    weights: np.ndarray,
    etf: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    n = len(all_raw[0])
    fd = all_raw[0].shape[1]
    sum_w = float(weights.sum())
    batch = torch.zeros(n, fd)
    local = torch.zeros(n, fd)
    sample = torch.zeros(n, fd)
    raw_post = torch.zeros(n, fd)
    pre_l2 = torch.zeros(n, fd)
    uniform_pre_l2 = torch.zeros(n, fd)
    logit_uniform = torch.zeros(n, len(etf))
    logit_sqrt = torch.zeros(n, len(etf))

    for k, raw in enumerate(all_raw):
        w = float(weights[k])
        batch_sd = raw.std(0, keepdim=True, unbiased=False).clamp_min(1e-6)
        batch += (raw - raw.mean(0, keepdim=True)) / batch_sd * w
        mu, sd = moments[k]
        local += (raw - mu.float()[None, :]) / sd.float().clamp_min(1e-6)[None, :] * w
        sample += F.layer_norm(raw, (fd,)) * w
        raw_post += raw * w
        norm = F.normalize(raw, dim=1)
        pre_l2 += norm * w
        uniform_pre_l2 += norm / len(all_raw)
        logits_k = norm @ etf.T
        logit_uniform += logits_k / len(all_raw)
        logit_sqrt += logits_k * (w / sum_w)

    return {
        "rga_full_batch_diagnostic": F.normalize(batch / sum_w, dim=1) @ etf.T,
        "rga_client_local_moments": F.normalize(local / sum_w, dim=1) @ etf.T,
        "rga_per_sample_layernorm": F.normalize(sample / sum_w, dim=1) @ etf.T,
        "raw_sum_post_l2": F.normalize(raw_post / sum_w, dim=1) @ etf.T,
        "pre_l2_feature_average": F.normalize(pre_l2 / sum_w, dim=1) @ etf.T,
        "uniform_feature_average": F.normalize(uniform_pre_l2, dim=1) @ etf.T,
        "uniform_logit_ensemble": logit_uniform,
        "sqrt_weighted_logit_ensemble": logit_sqrt,
    }


def weighted_parameter_average(
    models: Sequence[ResNet18Backbone], sample_weights: np.ndarray
) -> ResNet18Backbone:
    """One-shot FedAvg diagnostic for the same-architecture client backbones."""
    if len(models) == 0:
        raise ValueError("at least one client model is required")
    weights = np.asarray(sample_weights, dtype=np.float64)
    weights = weights / weights.sum()
    states = [model.state_dict() for model in models]
    merged: Dict[str, torch.Tensor] = {}
    for key, first in states[0].items():
        if first.is_floating_point():
            value = torch.zeros_like(first, dtype=torch.float32)
            for weight, state in zip(weights, states):
                value.add_(state[key].float(), alpha=float(weight))
            merged[key] = value.to(first.dtype)
        else:
            # BatchNorm's num_batches_tracked is bookkeeping, not a model parameter.
            merged[key] = first.clone()
    averaged = ResNet18Backbone(models[0].fc.out_features)
    averaged.load_state_dict(merged)
    return averaged


def metrics_from_logits(
    logits: torch.Tensor, labels: np.ndarray, n_classes: int
) -> Dict[str, object]:
    logits = logits.float()
    pred = logits.argmax(1).numpy()
    probs = logits.softmax(1).numpy()
    acc = float((pred == labels).mean())
    per_class = []
    for c in range(n_classes):
        mask = labels == c
        per_class.append(float((pred[mask] == c).mean()) if mask.any() else None)
    try:
        auc = float(
            roc_auc_score(
                labels,
                probs,
                labels=np.arange(n_classes),
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        auc = float("nan")
    valid_pc = [x for x in per_class if x is not None]
    return {
        "accuracy": acc,
        "macro_auc_ovr": auc,
        "per_class_accuracy": per_class,
        "worst_class_accuracy": min(valid_pc) if valid_pc else None,
    }


def evaluate_loader(
    models: Sequence[ResNet18Backbone],
    moments: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    weights: np.ndarray,
    etf: torch.Tensor,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
    parameter_average_model: Optional[ResNet18Backbone] = None,
) -> Dict[str, Dict[str, object]]:
    raw, labels = extract_all_raw(models, loader, device)
    method_logits = aggregate_logits(raw, moments, weights, etf)
    single_metrics = [
        metrics_from_logits(F.normalize(r, dim=1) @ etf.T, labels, n_classes)
        for r in raw
    ]
    result = {
        name: metrics_from_logits(logits, labels, n_classes)
        for name, logits in method_logits.items()
    }
    result["single_clients"] = single_metrics
    result["best_single_client_diagnostic"] = max(
        single_metrics, key=lambda x: float(x["accuracy"])
    )
    if parameter_average_model is not None:
        averaged_raw, averaged_labels = extract_all_raw(
            [parameter_average_model], loader, device
        )
        result["one_shot_parameter_average"] = metrics_from_logits(
            F.normalize(averaged_raw[0], dim=1) @ etf.T,
            averaged_labels,
            n_classes,
        )
    return result


def load_npz(path: Path, limit_train: int, limit_test: int):
    data = np.load(path, allow_pickle=False)
    train_images = data["train_images"]
    train_labels = data["train_labels"].reshape(-1)
    test_images = data["test_images"]
    test_labels = data["test_labels"].reshape(-1)
    if limit_train > 0:
        train_images = train_images[:limit_train]
        train_labels = train_labels[:limit_train]
    if limit_test > 0:
        test_images = test_images[:limit_test]
        test_labels = test_labels[:limit_test]
    return train_images, train_labels, test_images, test_labels


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASET_INFO))
    ap.add_argument("--data", type=Path, required=True, help="official MedMNIST .npz")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--n_clients", type=int, default=5)
    ap.add_argument("--noise_rate", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--feature_dim", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument(
        "--validate_only",
        action="store_true",
        help="load data and validate the deterministic partition without training",
    )
    ap.add_argument("--corruptions", default="")
    ap.add_argument("--severities", default="1,3,5")
    ap.add_argument("--limit_train", type=int, default=0, help="smoke-test only")
    ap.add_argument("--limit_test", type=int, default=0, help="smoke-test only")
    args = ap.parse_args()

    if not 0.0 <= args.noise_rate < 1.0:
        raise ValueError("noise_rate must be in [0,1)")
    cfg = CellConfig(
        dataset=args.dataset,
        alpha=args.alpha,
        n_clients=args.n_clients,
        noise_rate=args.noise_rate,
        seed=args.seed,
        epochs=args.epochs,
        feature_dim=args.feature_dim,
    )
    result_path = args.output / "results" / f"{cfg.tag}.json"
    meta_path = args.output / "results" / f"{cfg.tag}.meta.json"
    if result_path.exists() and not args.eval_only:
        print(f"SKIP completed cell: {result_path}", flush=True)
        return

    seed_everything(cfg.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    n_classes = DATASET_INFO[cfg.dataset]["classes"]
    start = time.time()
    train_images, train_labels, test_images, test_labels = load_npz(
        args.data, args.limit_train, args.limit_test
    )
    clients, counts = dirichlet_partition(
        train_labels,
        cfg.n_clients,
        cfg.alpha,
        n_classes,
        cfg.seed,
        min_client_size=min(32, max(2, len(train_labels) // (20 * cfg.n_clients))),
    )
    noisy_labels, changed = inject_symmetric_noise(
        train_labels, clients, cfg.noise_rate, n_classes, cfg.seed
    )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "dataset": cfg.dataset,
                    "train_n": int(len(train_labels)),
                    "test_n": int(len(test_labels)),
                    "classes_present": [int(x) for x in np.unique(train_labels)],
                    "client_sizes": [int(len(x)) for x in clients],
                    "class_counts": counts.tolist(),
                    "changed_labels": changed,
                },
                indent=2,
            )
        )
        return
    etf = generate_etf(n_classes, cfg.feature_dim, seed=42)
    cell_dir = args.output / "checkpoints" / cfg.tag
    cell_dir.mkdir(parents=True, exist_ok=True)

    models: List[ResNet18Backbone] = []
    moments: List[Tuple[torch.Tensor, torch.Tensor]] = []
    client_meta: List[Dict[str, object]] = []
    for k, idx in enumerate(clients):
        seed_everything(cfg.seed * 1000 + k + 17)
        if args.eval_only and not (cell_dir / f"client_{k}.pt").exists():
            raise FileNotFoundError(cell_dir / f"client_{k}.pt")
        model, mu, sd, meta = train_or_load_client(
            k,
            cfg,
            train_images,
            noisy_labels,
            train_labels,
            idx,
            etf,
            cell_dir,
            device,
            args.batch_size,
            args.workers,
            args.lr,
            args.save_every,
        )
        models.append(model.cpu())
        moments.append((mu.cpu(), sd.cpu()))
        client_meta.append(meta)

    sample_counts = np.asarray([len(x) for x in clients], dtype=np.float64)
    weights = np.sqrt(sample_counts)
    parameter_average_model = weighted_parameter_average(models, sample_counts)
    test_ds = MedArrayDataset(
        test_images, test_labels, None, build_transform(cfg.dataset, train=False)
    )
    test_loader = make_loader(test_ds, max(args.batch_size, 512), False, args.workers)
    clean = evaluate_loader(
        models,
        moments,
        weights,
        etf,
        test_loader,
        device,
        n_classes,
        parameter_average_model,
    )

    corr_results: Dict[str, object] = {}
    corruption_names = parse_csv(args.corruptions)
    severities = [int(x) for x in parse_csv(args.severities)]
    for name in corruption_names:
        for severity in severities:
            key = f"{name}_s{severity}"
            print(f"  corruption eval: {key}", flush=True)
            ds = MedArrayDataset(
                test_images,
                test_labels,
                None,
                build_transform(cfg.dataset, train=False),
                RegistryCorruptor(cfg.dataset, name, severity),
            )
            loader = make_loader(ds, max(args.batch_size, 512), False, args.workers)
            corr_results[key] = evaluate_loader(
                models,
                moments,
                weights,
                etf,
                loader,
                device,
                n_classes,
                parameter_average_model,
            )

    corr_summary: Dict[str, object] = {}
    if corr_results:
        method_names = [k for k in clean if k not in ("single_clients",)]
        for method in method_names:
            vals = []
            for row in corr_results.values():
                metric = row.get(method)
                if isinstance(metric, dict) and "accuracy" in metric:
                    vals.append(float(metric["accuracy"]))
            if vals:
                corr_summary[method] = {
                    "mean_accuracy": float(np.mean(vals)),
                    "std_across_corruptions": float(np.std(vals)),
                    "worst_accuracy": float(np.min(vals)),
                }

    result = {
        "schema_version": 1,
        "cell": asdict(cfg),
        "tag": cfg.tag,
        "code_revision": git_revision(),
        "data": {
            "path": str(args.data),
            "sha256": file_sha256(args.data),
            "train_n": int(len(train_labels)),
            "test_n": int(len(test_labels)),
            "official_split": True,
        },
        "partition": {
            "counts": counts.tolist(),
            "client_sizes": [int(len(x)) for x in clients],
            "changed_labels": changed,
            "actual_noise_rate": float(sum(changed) / len(train_labels)),
        },
        "client_meta": client_meta,
        "normalization_note": (
            "Primary deployment candidate uses client-local feature moments uploaded with "
            "the model; full-batch statistics are diagnostic only."
        ),
        "moment_upload_overhead_bytes_per_client_fp32": int(2 * cfg.feature_dim * 4),
        "clean": clean,
        "corruptions": corr_results,
        "corruption_summary": corr_summary,
    }
    atomic_json_dump(result, result_path)

    elapsed = time.time() - start
    peak_mb = (
        torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
    )
    meta = {
        "status": "complete",
        "tag": cfg.tag,
        "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "pid": os.getpid(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": elapsed,
        "gpu_peak_mb": peak_mb,
        "python": sys.version,
        "torch": torch.__version__,
        "device": str(device),
        "argv": sys.argv,
        "result": str(result_path),
    }
    atomic_json_dump(meta, meta_path)
    print(
        f"COMPLETE {cfg.tag}: local-moments acc="
        f"{clean['rga_client_local_moments']['accuracy'] * 100:.2f}% "
        f"full-batch diagnostic={clean['rga_full_batch_diagnostic']['accuracy'] * 100:.2f}% "
        f"elapsed={elapsed / 60:.1f}min peak={peak_mb:.0f}MB",
        flush=True,
    )


if __name__ == "__main__":
    main()
