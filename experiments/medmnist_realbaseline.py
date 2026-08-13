#!/usr/bin/env python3
"""MedMNIST with REAL baselines (fixes the aggregation-only comparison).

The original medmnist_fedsra.py computes "O-FedAvg" / "logit ensemble" by
re-aggregating the SAME ETF backbones -- i.e. they are FedSRA ablations, not
independent baselines. This script trains genuinely independent models on the
identical incomplete-coverage Dirichlet partition:

  * fedsra : ETF backbone + RGA (reuses joint_etf_loss + aggregate_logits)
  * ce     : ResNet18 + linear head, cross-entropy, SHARED init
             -> real one-shot FedAvg (parameter average) + CE logit ensembles
  * fafi   : official FAFI objective (learnable proto, four-term loss)

All three use the from-scratch small-image ResNet18Backbone, the same partition,
epochs, and seeds, so the comparison is apples-to-apples.

Usage:
  python medmnist_realbaseline.py --method ce --dataset pathmnist \
    --data medmnist_data/pathmnist.npz --alpha 0.05 --seed 42 --epochs 100 \
    --output medmnist_out/realbaseline
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from medmnist_fedsra import (
    DATASET_INFO,
    MedArrayDataset,
    ResNet18Backbone,
    aggregate_logits,
    atomic_json_dump,
    atomic_torch_save,
    build_transform,
    dirichlet_partition,
    generate_etf,
    git_revision,
    inject_symmetric_noise,
    joint_etf_loss,
    load_npz,
    make_loader,
    metrics_from_logits,
    seed_everything,
    state_to_fp16_cpu,
)

FAFI_ROOT = Path(__file__).resolve().parents[2] / "FAFI_ICML25-master-orgin"
sys.path.insert(0, str(FAFI_ROOT))
from oneshot_algorithms.ours.unsupervised_loss import (  # noqa: E402
    Contrastive_proto_feature_loss,
    Contrastive_proto_loss,
    SupConLoss,
)


@dataclass(frozen=True)
class Config:
    method: str
    dataset: str
    alpha: float
    n_clients: int
    noise_rate: float
    seed: int
    epochs: int
    feature_dim: int

    @property
    def tag(self) -> str:
        return f"mmreal_{self.method}_{self.dataset}_a{self.alpha}_n{self.noise_rate}_s{self.seed}"


class CEModel(nn.Module):
    def __init__(self, feature_dim: int, n_classes: int) -> None:
        super().__init__()
        self.backbone = ResNet18Backbone(feature_dim)
        self.classifier = nn.Linear(feature_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone.forward_raw(x))


class FAFIModel(nn.Module):
    def __init__(self, feature_dim: int, n_classes: int) -> None:
        super().__init__()
        self.backbone = ResNet18Backbone(feature_dim)
        self.learnable_proto = nn.Parameter(torch.randn(n_classes, feature_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = F.normalize(self.backbone.forward_raw(x), p=2, dim=1, eps=1e-12)
        return feat @ self.learnable_proto.t(), feat


class TwoViewMed(Dataset):
    def __init__(self, base: MedArrayDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        # MedArrayDataset applies a (random) transform each call -> two views
        v1, y = self.base[i]
        v2, _ = self.base[i]
        return v1, v2, y


def average_state(models_in: Sequence[nn.Module], weights: np.ndarray) -> nn.Module:
    averaged = copy.deepcopy(models_in[0]).cpu()
    states = [m.cpu().state_dict() for m in models_in]
    w = torch.as_tensor(weights, dtype=torch.float64); w = w / w.sum()
    merged = {}
    for key, first in states[0].items():
        if first.is_floating_point():
            merged[key] = sum(float(wi) * s[key].float() for wi, s in zip(w, states)).to(first.dtype)
        else:
            merged[key] = first.clone()
    averaged.load_state_dict(merged)
    return averaged


def cosine(opt, epochs):
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)


def train_client(method, cfg, base_ds, etf, n_classes, device, batch_size, workers, lr, init_state):
    if method == "fedsra":
        model = ResNet18Backbone(cfg.feature_dim)
    elif method == "ce":
        model = CEModel(cfg.feature_dim, n_classes); model.load_state_dict(init_state)
    else:
        model = FAFIModel(cfg.feature_dim, n_classes); model.load_state_dict(init_state)
    model.to(device)
    if method == "fafi":
        loader = make_loader(TwoViewMed(base_ds), batch_size, True, workers, drop_last=False)
        supcon = SupConLoss(temperature=0.07)
        proto_feat = Contrastive_proto_feature_loss(temperature=1.0)
        proto_reg = Contrastive_proto_loss(temperature=1.0)
    else:
        loader = make_loader(base_ds, batch_size, True, workers, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = cosine(opt, cfg.epochs)
    etf_dev = etf.to(device)
    amp = device.type == "cuda"
    for epoch in range(cfg.epochs):
        model.train()
        for batch in loader:
            if (len(batch[-1]) < 2):  # skip size<2 batches (BatchNorm needs >=2)
                continue
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                if method == "fedsra":
                    x, y = batch; x, y = x.to(device), y.to(device)
                    loss = joint_etf_loss(model.forward_raw(x), y, etf_dev, 0.1)
                elif method == "ce":
                    x, y = batch; x, y = x.to(device), y.to(device)
                    loss = F.cross_entropy(model(x), y)
                else:
                    v1, v2, y = batch
                    v1, v2, y = v1.to(device), v2.to(device), y.to(device)
                    images = torch.cat([v1, v2], 0); doubled = torch.cat([y, y], 0)
                    logits, feats = model(images)
                    b = len(y); f1, f2 = torch.split(feats, [b, b], 0)
                    loss = (F.cross_entropy(logits, doubled) + supcon(torch.stack([f1, f2], 1), y)
                            + proto_feat(feats, model.learnable_proto, doubled)
                            + proto_reg(model.learnable_proto))
            loss.backward()
            if method == "fafi":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
    model.eval()
    mu = sd = None
    if method == "fedsra":
        el = make_loader(base_ds, max(256, batch_size), False, workers)
        sx = sx2 = None; n = 0
        with torch.no_grad():
            for x, _ in el:
                raw = model.forward_raw(x.to(device)).float().cpu()
                sx = raw.double().sum(0) if sx is None else sx + raw.double().sum(0)
                sx2 = raw.double().square().sum(0) if sx2 is None else sx2 + raw.double().square().sum(0)
                n += len(raw)
        mu = (sx / n).float(); sd = (sx2 / n - (sx / n).square()).clamp_min(1e-8).sqrt().float()
    return model.cpu(), mu, sd


@torch.no_grad()
def ce_eval(models_in, sample_counts, loader, device, n_classes):
    logits_all = []; labels = None
    for i, m in enumerate(models_in):
        m.to(device).eval(); outs = []; ys = []
        for x, y in loader:
            outs.append(m(x.to(device)).float().cpu())
            if i == 0: ys.append(y)
        logits_all.append(torch.cat(outs))
        if i == 0: labels = torch.cat(ys).numpy()
        m.cpu()
    w = np.sqrt(sample_counts); w = w / w.sum()
    res = {"uniform_logit_ensemble": torch.stack(logits_all).mean(0),
           "sqrt_weighted_logit_ensemble": sum(float(wi) * x for wi, x in zip(w, logits_all))}
    avg = average_state(models_in, sample_counts).to(device).eval()
    outs = []
    for x, _ in loader:
        outs.append(avg(x.to(device)).float().cpu())
    res["one_shot_fedavg"] = torch.cat(outs)
    return {k: metrics_from_logits(v, labels, n_classes) for k, v in res.items()}


@torch.no_grad()
def fedsra_eval(models_in, moments, sample_counts, etf, loader, device, n_classes):
    raws = []; labels = None
    for i, m in enumerate(models_in):
        m.to(device).eval(); feats = []; ys = []
        for x, y in loader:
            feats.append(m.forward_raw(x.to(device)).float().cpu())
            if i == 0: ys.append(y)
        raws.append(torch.cat(feats))
        if i == 0: labels = torch.cat(ys).numpy()
        m.cpu()
    out = aggregate_logits(raws, moments, np.sqrt(sample_counts), etf)
    return {k: metrics_from_logits(v, labels, n_classes) for k, v in out.items()}


@torch.no_grad()
def fafi_eval(models_in, sample_counts, loader, device, n_classes):
    w = torch.as_tensor(sample_counts, dtype=torch.float32); w = w / w.sum()
    gproto = torch.stack([m.learnable_proto.detach().float().cpu() for m in models_in]).mean(0).to(device)
    feats = []; labels = []
    for x, y in loader:
        x = x.to(device); fused = None
        for wi, m in zip(w, models_in):
            m.to(device).eval(); f = m.backbone.forward_raw(x).float()
            fused = float(wi) * f if fused is None else fused + float(wi) * f
        feats.append(F.normalize(fused, dim=1).cpu()); labels.append(y)
    for m in models_in: m.cpu()
    logits = torch.cat(feats) @ gproto.cpu().t()
    return {"fafi_weighted_feature_ensemble": metrics_from_logits(logits, torch.cat(labels).numpy(), n_classes)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["fedsra", "ce", "fafi"], required=True)
    ap.add_argument("--dataset", required=True, choices=sorted(DATASET_INFO))
    ap.add_argument("--data", type=Path, required=True)
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
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_test", type=int, default=0)
    args = ap.parse_args()

    cfg = Config(args.method, args.dataset, args.alpha, args.n_clients,
                 args.noise_rate, args.seed, args.epochs, args.feature_dim)
    (args.output / "results").mkdir(parents=True, exist_ok=True)
    out_path = args.output / "results" / f"{cfg.tag}.json"
    if out_path.exists():
        print(f"SKIP {out_path}"); return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = DATASET_INFO[cfg.dataset]["classes"]
    seed_everything(cfg.seed)
    tr_imgs, tr_lbls, te_imgs, te_lbls = load_npz(args.data, args.limit_train, args.limit_test)
    clients, counts = dirichlet_partition(tr_lbls, cfg.n_clients, cfg.alpha, n_classes, cfg.seed,
                                          min_client_size=min(32, max(2, len(tr_lbls) // (20 * cfg.n_clients))))
    noisy, _ = inject_symmetric_noise(tr_lbls, clients, cfg.noise_rate, n_classes, cfg.seed)
    coverage = [(counts[k] > 0).sum() for k in range(cfg.n_clients)]
    print(f"[partition] a={cfg.alpha} client class-coverage: {coverage} / {n_classes}", flush=True)

    etf = generate_etf(n_classes, cfg.feature_dim, 42)
    init_state = None
    if cfg.method in ("ce", "fafi"):
        seed_everything(cfg.seed)
        init = CEModel(cfg.feature_dim, n_classes) if cfg.method == "ce" else FAFIModel(cfg.feature_dim, n_classes)
        init_state = {k: v.detach().cpu().clone() for k, v in init.state_dict().items()}
        del init

    trained = []; moments = []; sample_counts = []
    t0 = time.time()
    for k, idx in enumerate(clients):
        seed_everything(cfg.seed * 1000 + k + 17)
        sample_counts.append(len(idx))
        base = MedArrayDataset(tr_imgs, noisy, idx, build_transform(cfg.dataset, train=True))
        print(f"[{cfg.method}] client {k} ({len(idx)} imgs)", flush=True)
        model, mu, sd = train_client(cfg.method, cfg, base, etf, n_classes, device,
                                     args.batch_size, args.workers, args.lr, init_state)
        trained.append(model)
        if cfg.method == "fedsra":
            moments.append((mu, sd))

    test_ds = MedArrayDataset(te_imgs, te_lbls, None, build_transform(cfg.dataset, train=False))
    loader = make_loader(test_ds, max(256, args.batch_size), False, args.workers)
    sc = np.asarray(sample_counts)
    if cfg.method == "fedsra":
        evaluation = fedsra_eval(trained, moments, sc, etf, loader, device, n_classes)
        key = "rga_client_local_moments"
    elif cfg.method == "ce":
        evaluation = ce_eval(trained, sc, loader, device, n_classes)
        key = "one_shot_fedavg"
    else:
        evaluation = fafi_eval(trained, sc, loader, device, n_classes)
        key = "fafi_weighted_feature_ensemble"

    result = {
        "schema_version": 1, "dataset": cfg.dataset, "method": cfg.method,
        "cell": asdict(cfg), "tag": cfg.tag, "code_revision": git_revision(),
        "n_classes": n_classes, "client_class_coverage": [int(c) for c in coverage],
        "sample_counts": sample_counts, "evaluation": evaluation,
        "elapsed_s": time.time() - t0,
        "gpu_peak_mb": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0,
        "completed_at": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
    }
    atomic_json_dump(result, out_path)
    m = evaluation[key]
    print(f"COMPLETE {cfg.tag}: {key} acc={m['accuracy']*100:.2f}% "
          f"AUC={m['macro_auc_ovr']*100:.2f}% worst={100*(m['worst_class_accuracy'] or 0):.2f}% "
          f"elapsed={(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
