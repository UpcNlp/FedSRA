"""
Pipeline: SSL Feature Extraction + Normalizing Flow Density Estimation

Paradigm shift:
  - Neural network (VICReg) does ONLY feature extraction
  - Density estimation uses statistical/generative methods, NOT neural network training objectives
  - Per-class normalizing flow trained by MAXIMUM LIKELIHOOD
    → the correct objective for single-class density estimation
    → no negatives needed, no collapse possible, more data = better

Why normalizing flows work for single-class learning:
  1. Invertible by construction → cannot collapse
  2. Trained by exact log-likelihood → directly optimizes density quality
  3. Single-class sufficient → only models p(features | class c)
  4. More data → better likelihood estimates → monotonically improving
  5. Adapts to ANY feature distribution → works regardless of SSL's uniformity

Architecture:
  Phase 1: VICReg backbone (no L2 norm) → augmentation-invariant features
  Phase 2: PCA reduction (256 → 64 dim) → tractable density estimation
  Phase 3: RealNVP flow per class → exact log-likelihood scoring
"""

import os
import json
import time
import math
import random
import argparse
import warnings
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms

warnings.filterwarnings("ignore")

# ============================================================
# Global Config
# ============================================================
N_CLIENTS = 5
N_CLASSES = 10

# Backbone training
BATCH_SIZE = 256
BB_LR = 1e-3
BB_WD = 1e-4
BB_EPOCHS = 300
FEAT_DIM = 256

# VICReg weights (standard, no modifications)
LAMBDA_INV = 25.0
MU_VAR = 25.0
NU_COV = 1.0

# Flow training
FLOW_PCA_DIM = 64          # PCA reduction before flow
FLOW_N_LAYERS = 6          # coupling layers
FLOW_HIDDEN = 128          # hidden dim in coupling networks
FLOW_LR = 5e-4
FLOW_EPOCHS = 200
FLOW_BATCH = 256

MIN_SAMPLES = 30
CALIB_RATIO = 0.10
EPS_STD = 1e-6
MAX_RELIABILITY = 6.0

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


# ============================================================
# Utils
# ============================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def dirichlet_split(targets, n_clients, alpha, seed=42):
    rng = np.random.RandomState(seed)
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[int(label)].append(idx)

    client_indices = defaultdict(list)
    client_class_counts = defaultdict(lambda: defaultdict(int))

    for c in range(N_CLASSES):
        idxs = np.array(class_indices[c])
        rng.shuffle(idxs)
        props = rng.dirichlet([alpha] * n_clients)
        counts = (props * len(idxs)).astype(int)
        counts[-1] = len(idxs) - counts[:-1].sum()
        start = 0
        for k in range(n_clients):
            end = start + counts[k]
            if end > start:
                part = idxs[start:end].tolist()
                client_indices[k].extend(part)
                client_class_counts[k][c] = len(part)
            start = end
    return dict(client_indices), dict(client_class_counts)


def split_client_train_calib(base_targets, indices, calib_ratio=0.1, seed=42):
    rng = np.random.RandomState(seed)
    by_class = defaultdict(list)
    for idx in indices:
        c = int(base_targets[idx])
        by_class[c].append(idx)

    train_idx, calib_idx = [], []
    for c, idxs in by_class.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        if len(idxs) <= 2:
            train_idx.extend(idxs)
            continue
        n_calib = max(1, int(round(len(idxs) * calib_ratio)))
        n_calib = min(n_calib, len(idxs) - 1)
        calib_idx.extend(idxs[:n_calib])
        train_idx.extend(idxs[n_calib:])
    rng.shuffle(train_idx)
    rng.shuffle(calib_idx)
    return train_idx, calib_idx


# ============================================================
# Transforms & Datasets
# ============================================================
def get_ssl_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def get_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


class DualViewDataset(Dataset):
    def __init__(self, base_dataset, indices):
        self.data = base_dataset.data
        self.targets = np.array(base_dataset.targets)
        self.indices = list(indices)
        self.transform = get_ssl_transform()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx % len(self.indices)]
        img = Image.fromarray(self.data[real_idx])
        return self.transform(img), self.transform(img), int(self.targets[real_idx])


class IndexedDataset(Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.data = base_dataset.data
        self.targets = np.array(base_dataset.targets)
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img = Image.fromarray(self.data[real_idx])
        return self.transform(img), int(self.targets[real_idx])


# ============================================================
# Backbone (standard ResNet-like, NO L2 normalization)
# ============================================================
class GNBasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        gn = min(8, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(gn, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(gn, out_ch)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(gn, out_ch)
            )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)), inplace=True)
        out = self.gn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class Backbone(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(GNBasicBlock(64, 64), GNBasicBlock(64, 64))
        self.layer2 = nn.Sequential(GNBasicBlock(64, 128, 2), GNBasicBlock(128, 128))
        self.layer3 = nn.Sequential(GNBasicBlock(128, 256, 2), GNBasicBlock(256, 256))
        self.layer4 = nn.Sequential(GNBasicBlock(256, 512, 2), GNBasicBlock(512, 512))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, feat_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)   # NO L2 normalization


class Projector(nn.Module):
    def __init__(self, in_dim=256, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim))

    def forward(self, x):
        return self.net(x)


class SSLModel(nn.Module):
    def __init__(self, feat_dim=256, proj_dim=256):
        super().__init__()
        self.backbone = Backbone(feat_dim)
        self.projector = Projector(feat_dim, proj_dim)

    def encode(self, x):
        return self.backbone(x)

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return h, z


# ============================================================
# VICReg Loss (standard)
# ============================================================
def vicreg_loss(z1, z2, lam=LAMBDA_INV, mu=MU_VAR, nu=NU_COV):
    inv = F.mse_loss(z1, z2)

    std1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    var = torch.mean(F.relu(1.0 - std1)) + torch.mean(F.relu(1.0 - std2))

    z1c = z1 - z1.mean(0)
    z2c = z2 - z2.mean(0)
    N = z1.shape[0]
    cov1 = (z1c.T @ z1c) / max(N - 1, 1)
    cov2 = (z2c.T @ z2c) / max(N - 1, 1)
    cov = off_diagonal(cov1).pow(2).mean() + off_diagonal(cov2).pow(2).mean()

    return lam * inv + mu * var + nu * cov, {
        "inv": inv.item(), "var": var.item(), "cov": cov.item()
    }


# ============================================================
# Phase 1: Train VICReg Backbone
# ============================================================
def train_backbone(client_id, base_dataset, indices, device,
                   epochs=BB_EPOCHS, lr=BB_LR, wd=BB_WD):
    dataset = DualViewDataset(base_dataset, indices)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4, pin_memory=True,
                        drop_last=len(dataset) >= BATCH_SIZE,
                        persistent_workers=True)

    model = SSLModel(FEAT_DIM, FEAT_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

    best_state, best_q = None, -1e9

    print(f"\n  Client {client_id}: VICReg on {len(indices)} samples")

    for ep in range(epochs):
        model.train()
        loss_sum, n_batch = 0.0, 0
        stats_sum = defaultdict(float)

        for x1, x2, _ in loader:
            x1, x2 = x1.to(device, non_blocking=True), x2.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                _, z1 = model(x1)
                _, z2 = model(x2)
                loss, stats = vicreg_loss(z1, z2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            loss_sum += loss.item()
            for k, v in stats.items():
                stats_sum[k] += v
            n_batch += 1

        sch.step()

        model.eval()
        with torch.no_grad():
            batch = next(iter(loader))
            probe = batch[0][:128].to(device)
            feats = model.encode(probe)
            feat_std = feats.std(dim=0).mean().item()
            active = (feats.std(dim=0) > 0.01).float().mean().item()
            q = active + 10 * feat_std

        if q > best_q:
            best_q = q
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (ep + 1) % 50 == 0 or ep == 0:
            a = lambda name: stats_sum[name] / max(n_batch, 1)
            print(f"    ep {ep+1:3d}/{epochs}  loss={loss_sum/max(n_batch,1):.4f}  "
                  f"inv={a('inv'):.4f}  var={a('var'):.4f}  cov={a('cov'):.4f}  "
                  f"feat_std={feat_std:.4f}  active={active:.1%}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval().cpu()
    return model


# ============================================================
# Feature Extraction
# ============================================================
@torch.no_grad()
def extract_features(model, loader, device):
    model = model.to(device).eval()
    feats, labels = [], []
    for x, y in loader:
        feats.append(model.encode(x.to(device, non_blocking=True)).cpu())
        labels.append(y)
    model.cpu()
    return torch.cat(feats), torch.cat(labels)


# ============================================================
# PCA
# ============================================================
class PCATransform:
    """Fit PCA on training data, transform any features."""
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X):
        """X: (N, D) tensor"""
        self.mean = X.mean(dim=0)
        Xc = X - self.mean
        # Use SVD for numerical stability
        U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
        self.components = Vt[:self.n_components]  # (n_components, D)
        explained = (S[:self.n_components] ** 2).sum() / (S ** 2).sum()
        return explained.item()

    def transform(self, X):
        """X: (N, D) → (N, n_components)"""
        return (X - self.mean) @ self.components.T

    def to(self, device):
        self.mean = self.mean.to(device)
        self.components = self.components.to(device)
        return self


# ============================================================
# RealNVP Normalizing Flow
# ============================================================
class CouplingLayer(nn.Module):
    """Affine coupling layer for RealNVP."""
    def __init__(self, dim, hidden, mask):
        super().__init__()
        self.register_buffer("mask", mask)
        self.s_net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, dim), nn.Tanh())  # Tanh for stability
        self.t_net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, dim))

    def forward(self, x):
        """Forward: x → z, returns z and log_det"""
        x_masked = x * self.mask
        s = self.s_net(x_masked) * (1 - self.mask)
        t = self.t_net(x_masked) * (1 - self.mask)
        z = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det = s.sum(dim=1)
        return z, log_det

    def inverse(self, z):
        """Inverse: z → x"""
        z_masked = z * self.mask
        s = self.s_net(z_masked) * (1 - self.mask)
        t = self.t_net(z_masked) * (1 - self.mask)
        x = z_masked + (1 - self.mask) * (z - t) * torch.exp(-s)
        return x


class RealNVP(nn.Module):
    """
    RealNVP normalizing flow.
    Learns an invertible mapping from data distribution to standard Gaussian.
    Training objective: maximum likelihood (= minimize negative log-likelihood).

    This IS the correct objective for single-class density estimation:
    - No negatives needed
    - No collapse possible (invertible)
    - More data → better likelihood estimates
    """
    def __init__(self, dim, n_layers=6, hidden=128):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()

        for i in range(n_layers):
            # Alternating masks
            mask = torch.zeros(dim)
            if i % 2 == 0:
                mask[:dim // 2] = 1.0
            else:
                mask[dim // 2:] = 1.0
            self.layers.append(CouplingLayer(dim, hidden, mask))

    def forward(self, x):
        """x → z, with log_det for likelihood computation"""
        log_det_total = 0
        z = x
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_total += log_det
        return z, log_det_total

    def log_prob(self, x):
        """
        Compute log p(x) = log p_base(f(x)) + log|det(df/dx)|

        This is EXACT log-likelihood under the learned distribution.
        """
        z, log_det = self.forward(x)
        # Base distribution: standard Gaussian
        log_pz = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=1)
        return log_pz + log_det


def train_flow(features, device, n_layers=FLOW_N_LAYERS, hidden=FLOW_HIDDEN,
               lr=FLOW_LR, epochs=FLOW_EPOCHS, batch_size=FLOW_BATCH):
    """
    Train a normalizing flow on feature vectors.
    Pure maximum likelihood — the mathematically correct objective
    for density estimation from single-class data.
    """
    dim = features.shape[1]
    N = features.shape[0]

    flow = RealNVP(dim, n_layers=n_layers, hidden=hidden).to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    dataset = TensorDataset(features)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=N >= batch_size)

    best_nll = float("inf")
    best_state = None

    for ep in range(epochs):
        flow.train()
        nll_sum, n_batch = 0.0, 0
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            log_p = flow.log_prob(batch)
            loss = -log_p.mean()  # negative log-likelihood

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            opt.step()

            nll_sum += loss.item()
            n_batch += 1

        sch.step()
        avg_nll = nll_sum / max(n_batch, 1)

        if avg_nll < best_nll:
            best_nll = avg_nll
            best_state = {k: v.cpu().clone() for k, v in flow.state_dict().items()}

    if best_state:
        flow.load_state_dict(best_state)
    flow.eval().cpu()
    return flow, best_nll


# ============================================================
# Phase 2 & 3: Per-class density models
# ============================================================
@torch.no_grad()
def build_class_models(client_id, backbone, base_dataset,
                       train_indices, calib_indices,
                       class_counts, device):
    """
    For each class with enough data:
      1. Extract features
      2. PCA reduce
      3. Train normalizing flow (maximum likelihood)
      4. Calibrate using held-out data
    Also fits a full-covariance Gaussian as a simpler baseline.
    """
    # Extract all training features
    train_ds = IndexedDataset(base_dataset, train_indices, get_test_transform())
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False,
                              num_workers=4, pin_memory=True)
    all_feats, all_labels = extract_features(backbone, train_loader, device)

    # Extract calibration features
    calib_feats, calib_labels = None, None
    if len(calib_indices) > 0:
        calib_ds = IndexedDataset(base_dataset, calib_indices, get_test_transform())
        calib_loader = DataLoader(calib_ds, batch_size=256, shuffle=False,
                                  num_workers=4, pin_memory=True)
        calib_feats, calib_labels = extract_features(backbone, calib_loader, device)

    models = {}

    for c in sorted(class_counts.keys()):
        n = class_counts[c]
        if n < MIN_SAMPLES:
            continue

        mask = (all_labels == c)
        fc = all_feats[mask]
        if len(fc) < MIN_SAMPLES:
            continue

        # --- Full-covariance Gaussian (baseline) ---
        mu = fc.mean(dim=0)
        Xc = fc - mu
        cov = (Xc.T @ Xc) / max(len(fc) - 1, 1)
        # Ledoit-Wolf shrinkage
        trace_cov = torch.trace(cov)
        alpha_shrink = min(0.3, FEAT_DIM / max(len(fc), 1))
        cov_reg = (1 - alpha_shrink) * cov + alpha_shrink * (trace_cov / FEAT_DIM) * torch.eye(FEAT_DIM)

        # Robust inverse via pseudo-inverse
        try:
            cov_inv = torch.linalg.pinv(cov_reg)
        except:
            cov_inv = torch.eye(FEAT_DIM)

        # --- PCA ---
        pca = PCATransform(FLOW_PCA_DIM)
        explained = pca.fit(fc)

        fc_pca = pca.transform(fc)

        # Standardize for flow stability
        pca_mean = fc_pca.mean(dim=0)
        pca_std = fc_pca.std(dim=0).clamp(min=1e-6)
        fc_pca_norm = (fc_pca - pca_mean) / pca_std

        # --- Train Normalizing Flow ---
        print(f"    c{c}: training flow on {len(fc)} samples "
              f"(PCA {FEAT_DIM}→{FLOW_PCA_DIM}, explained={explained:.1%})")

        flow, best_nll = train_flow(
            fc_pca_norm, device,
            n_layers=FLOW_N_LAYERS, hidden=FLOW_HIDDEN,
            lr=FLOW_LR, epochs=FLOW_EPOCHS
        )

        pack = {
            "n": int(n),
            # Gaussian model
            "mu": mu,
            "cov_inv": cov_inv,
            # Flow model
            "flow": flow,
            "pca": pca,
            "pca_mean": pca_mean,
            "pca_std": pca_std,
            "flow_nll": best_nll,
        }

        # --- Calibration ---
        # Compute flow log-probs on fit set (for reference)
        with torch.no_grad():
            flow_dev = flow.to(device)
            fit_log_p = flow_dev.log_prob(fc_pca_norm.to(device)).cpu()
            flow_dev.cpu()

        fit_lp_mean = fit_log_p.mean()
        fit_lp_std = fit_log_p.std().clamp(min=EPS_STD)

        pack["fit_lp_mean"] = fit_lp_mean
        pack["fit_lp_std"] = fit_lp_std

        # Gaussian energy on fit set
        diff = fc - mu
        gauss_energy = (diff @ cov_inv * diff).sum(dim=1)
        pack["fit_ge_mean"] = gauss_energy.mean()
        pack["fit_ge_std"] = gauss_energy.std().clamp(min=EPS_STD)

        if calib_feats is not None and len(calib_feats) > 0:
            pos_mask = (calib_labels == c)
            neg_mask = (calib_labels != c)
            n_pos = int(pos_mask.sum())
            n_neg = int(neg_mask.sum())

            # Flow calibration
            calib_pca = pca.transform(calib_feats)
            calib_pca_norm = (calib_pca - pca_mean) / pca_std

            with torch.no_grad():
                flow_dev = flow.to(device)
                calib_lp = flow_dev.log_prob(calib_pca_norm.to(device)).cpu()
                flow_dev.cpu()

            if n_pos >= 3:
                pos_lp_mean = calib_lp[pos_mask].mean()
                pos_lp_std = calib_lp[pos_mask].std().clamp(min=EPS_STD)
            else:
                pos_lp_mean = fit_lp_mean
                pos_lp_std = fit_lp_std

            if n_neg >= 5:
                neg_lp_mean = calib_lp[neg_mask].mean()
                neg_lp_std = calib_lp[neg_mask].std().clamp(min=EPS_STD)
            else:
                neg_lp_mean = pos_lp_mean - 2.5 * pos_lp_std
                neg_lp_std = pos_lp_std * 1.5 + 1.0

            flow_threshold = 0.5 * (pos_lp_mean + neg_lp_mean)
            flow_scale = (pos_lp_std + neg_lp_std).clamp(min=0.1)

            sep = ((pos_lp_mean - neg_lp_mean) / (pos_lp_std + neg_lp_std + EPS_STD)).item()
            sep = max(0.0, sep)
            support = math.log(n + 1.0)
            pq = min(1.0, n_pos / 20.0) if n_pos > 0 else 0.0
            nq = min(1.0, n_neg / 50.0) if n_neg > 0 else 0.0
            reliability = sep * support * (0.5 + 0.5 * pq) * (0.5 + 0.5 * nq)
            reliability = min(reliability, MAX_RELIABILITY)

            # Gaussian calibration
            calib_ge = (((calib_feats - mu) @ cov_inv) * (calib_feats - mu)).sum(dim=1)
            if n_pos >= 3:
                gpos_mean = calib_ge[pos_mask].mean()
                gpos_std = calib_ge[pos_mask].std().clamp(min=EPS_STD)
            else:
                gpos_mean = pack["fit_ge_mean"]
                gpos_std = pack["fit_ge_std"]

            if n_neg >= 5:
                gneg_mean = calib_ge[neg_mask].mean()
                gneg_std = calib_ge[neg_mask].std().clamp(min=EPS_STD)
            else:
                gneg_mean = gpos_mean + 2.5 * gpos_std
                gneg_std = gpos_std * 1.5 + 1.0

            gauss_threshold = 0.5 * (gpos_mean + gneg_mean)
            gauss_scale = torch.sqrt(0.5 * (gpos_std**2 + gneg_std**2)).clamp(min=1.0)

            pack.update({
                "flow_threshold": flow_threshold,
                "flow_scale": flow_scale,
                "gauss_threshold": gauss_threshold,
                "gauss_scale": gauss_scale,
                "reliability": float(reliability),
                "n_pos_calib": n_pos,
                "n_neg_calib": n_neg,
                "flow_sep": float(sep),
            })
        else:
            pack.update({
                "flow_threshold": fit_lp_mean - 1.0 * fit_lp_std,
                "flow_scale": fit_lp_std,
                "gauss_threshold": pack["fit_ge_mean"] + 1.25 * pack["fit_ge_std"],
                "gauss_scale": pack["fit_ge_std"],
                "reliability": 0.5 * math.log(n + 1.0),
                "n_pos_calib": 0,
                "n_neg_calib": 0,
                "flow_sep": 0.0,
            })

        models[c] = pack

    print(f"  Client {client_id}: {len(models)} class models")
    for c, p in models.items():
        print(f"    c{c}: n={p['n']:5d}, rel={p['reliability']:.3f}, "
              f"flow_nll={p['flow_nll']:.2f}, flow_sep={p['flow_sep']:.2f}, "
              f"pos={p['n_pos_calib']}, neg={p['n_neg_calib']}")

    return models


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate(backbones, class_models, class_counts_fit, test_loader, device):
    n_test = len(test_loader.dataset)
    test_labels = None

    # Score tensors
    flow_scores = torch.full((N_CLIENTS, n_test, N_CLASSES), float("-inf"))
    gauss_scores = torch.full((N_CLIENTS, n_test, N_CLASSES), float("-inf"))
    flow_probs = torch.zeros(N_CLIENTS, n_test, N_CLASSES)
    gauss_probs = torch.zeros(N_CLIENTS, n_test, N_CLASSES)
    reliabilities = torch.zeros(N_CLIENTS, N_CLASSES)

    for k in range(N_CLIENTS):
        if k not in backbones:
            continue

        backbone = backbones[k].to(device).eval()
        feats, labels = [], []
        for x, y in test_loader:
            feats.append(backbone.encode(x.to(device, non_blocking=True)).cpu())
            if test_labels is None:
                labels.append(y)
        feats = torch.cat(feats)
        if test_labels is None:
            test_labels = torch.cat(labels).numpy()
        backbone.cpu()

        for c, pack in class_models[k].items():
            rel = pack["reliability"]
            reliabilities[k, c] = rel

            # --- Flow score ---
            pca_feats = pack["pca"].transform(feats)
            pca_norm = (pca_feats - pack["pca_mean"]) / pack["pca_std"]

            flow = pack["flow"].to(device)
            log_p = flow.log_prob(pca_norm.to(device)).cpu()
            flow.cpu()

            flow_scores[k, :, c] = log_p

            margin = (log_p - pack["flow_threshold"]) / pack["flow_scale"]
            flow_probs[k, :, c] = torch.sigmoid(margin)

            # --- Gaussian score (full covariance) ---
            diff = feats - pack["mu"]
            energy = (diff @ pack["cov_inv"] * diff).sum(dim=1)
            gauss_margin = -(energy - pack["gauss_threshold"]) / pack["gauss_scale"]
            gauss_scores[k, :, c] = -energy
            gauss_probs[k, :, c] = torch.sigmoid(gauss_margin)

    results = {}

    # ==== Flow-based strategies ====

    # F1: flow top expert
    top_flow = torch.full((n_test, N_CLASSES), float("-inf"))
    for c in range(N_CLASSES):
        bk, bn = -1, -1
        for k in range(N_CLIENTS):
            if k in class_models and c in class_models[k]:
                n = class_models[k][c]["n"]
                if n > bn:
                    bn = n
                    bk = k
        if bk >= 0:
            top_flow[:, c] = flow_scores[bk, :, c]
    results["F1_flow_topK"] = float((top_flow.argmax(1).numpy() == test_labels).mean())

    # F2: flow max across clients
    best_flow, _ = flow_scores.max(dim=0)
    results["F2_flow_max"] = float((best_flow.argmax(1).numpy() == test_labels).mean())

    # F3: flow reliability-weighted prob
    fused = torch.zeros(n_test, N_CLASSES)
    fden = torch.zeros(N_CLASSES)
    for k in range(N_CLIENTS):
        if k not in class_models:
            continue
        for c in class_models[k]:
            w = reliabilities[k, c].item()
            if w <= 0:
                w = 0.01
            fused[:, c] += flow_probs[k, :, c] * w
            fden[c] += w
    for c in range(N_CLASSES):
        if fden[c] > 0:
            fused[:, c] /= fden[c]
    results["F3_flow_rel_prob"] = float((fused.argmax(1).numpy() == test_labels).mean())

    # F4: flow logN-weighted
    wt = torch.zeros(n_test, N_CLASSES)
    wd = torch.zeros(N_CLASSES)
    for k in range(N_CLIENTS):
        if k not in class_models:
            continue
        for c in class_models[k]:
            w = math.log(class_models[k][c]["n"] + 1.0)
            valid = torch.isfinite(flow_scores[k, :, c])
            if valid.any():
                wt[valid, c] += flow_scores[k, valid, c] * w
                wd[c] += w
    for c in range(N_CLASSES):
        if wd[c] > 0:
            wt[:, c] /= wd[c]
        else:
            wt[:, c] = float("-inf")
    results["F4_flow_wlogN"] = float((wt.argmax(1).numpy() == test_labels).mean())

    # ==== Gaussian (full-cov) strategies ====

    # G1: Gaussian top expert
    top_g = torch.full((n_test, N_CLASSES), float("-inf"))
    for c in range(N_CLASSES):
        bk, bn = -1, -1
        for k in range(N_CLIENTS):
            if k in class_models and c in class_models[k]:
                n = class_models[k][c]["n"]
                if n > bn:
                    bn = n
                    bk = k
        if bk >= 0:
            top_g[:, c] = gauss_scores[bk, :, c]
    results["G1_gauss_topK"] = float((top_g.argmax(1).numpy() == test_labels).mean())

    # G2: Gaussian reliability-weighted prob
    gfused = torch.zeros(n_test, N_CLASSES)
    gfden = torch.zeros(N_CLASSES)
    for k in range(N_CLIENTS):
        if k not in class_models:
            continue
        for c in class_models[k]:
            w = reliabilities[k, c].item()
            if w <= 0:
                w = 0.01
            gfused[:, c] += gauss_probs[k, :, c] * w
            gfden[c] += w
    for c in range(N_CLASSES):
        if gfden[c] > 0:
            gfused[:, c] /= gfden[c]
    results["G2_gauss_rel_prob"] = float((gfused.argmax(1).numpy() == test_labels).mean())

    # ==== Ensemble: flow + Gaussian ====
    ens = torch.zeros(n_test, N_CLASSES)
    for c in range(N_CLASSES):
        fp = fused[:, c] if fden[c] > 0 else torch.zeros(n_test)
        gp = gfused[:, c] if gfden[c] > 0 else torch.zeros(n_test)
        if fden[c] > 0 and gfden[c] > 0:
            ens[:, c] = 0.5 * fp + 0.5 * gp
        elif fden[c] > 0:
            ens[:, c] = fp
        else:
            ens[:, c] = gp
    results["E1_ensemble"] = float((ens.argmax(1).numpy() == test_labels).mean())

    return results


# ============================================================
# Main
# ============================================================
def run_experiment(alpha, seed, gpu):
    seed_everything(seed)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*72}")
    print(f"  SSL + Normalizing Flow Pipeline")
    print(f"  alpha={alpha}  seed={seed}  bb_epochs={BB_EPOCHS}")
    print(f"  flow: PCA={FLOW_PCA_DIM}  layers={FLOW_N_LAYERS}  "
          f"hidden={FLOW_HIDDEN}  epochs={FLOW_EPOCHS}")
    print(f"{'='*72}")

    train_base = datasets.CIFAR10("./data", train=True, download=True)
    targets = np.array(train_base.targets)
    client_indices, client_class_counts = dirichlet_split(targets, N_CLIENTS, alpha, seed)

    client_train_idx, client_calib_idx = {}, {}
    client_class_counts_fit = defaultdict(lambda: defaultdict(int))

    print("\n  Data distribution:")
    for k in range(N_CLIENTS):
        tr, ca = split_client_train_calib(
            targets, client_indices[k], CALIB_RATIO, seed + 1000 + k)
        client_train_idx[k] = tr
        client_calib_idx[k] = ca
        for idx in tr:
            client_class_counts_fit[k][int(targets[idx])] += 1

        ccc = client_class_counts.get(k, {})
        n_cls = sum(v > 0 for v in ccc.values())
        n_smp = sum(ccc.values())
        top = sorted(ccc.items(), key=lambda x: -x[1])[:5]
        print(f"    Client {k}: {n_cls:2d} cls, {n_smp:5d} smp  "
              f"train={len(tr):5d} calib={len(ca):4d}  "
              f"top: {', '.join(f'c{c}={n}' for c,n in top)}")

    test_ds = datasets.CIFAR10("./data", train=False, transform=get_test_transform())
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=4, pin_memory=True)

    # Phase 1: Train backbones
    print(f"\n{'='*60}")
    print("  Phase 1: VICReg Backbone Training")
    print(f"{'='*60}")

    backbones = {}
    t0 = time.time()
    for k in range(N_CLIENTS):
        if len(client_train_idx[k]) < 2:
            continue
        backbones[k] = train_backbone(k, train_base, client_train_idx[k],
                                       device, BB_EPOCHS, BB_LR, BB_WD)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    t_bb = time.time() - t0
    print(f"\n  Backbone time: {t_bb:.0f}s ({t_bb/60:.1f}min)")

    # Phase 2: Per-class density models (Gaussian + Flow)
    print(f"\n{'='*60}")
    print("  Phase 2: Per-class Density Models (Gaussian + Flow)")
    print(f"{'='*60}")

    class_models = {}
    t1 = time.time()
    for k in range(N_CLIENTS):
        if k not in backbones:
            continue
        print(f"\n  Client {k}:")
        class_models[k] = build_class_models(
            k, backbones[k], train_base,
            client_train_idx[k], client_calib_idx[k],
            {c: n for c, n in client_class_counts_fit[k].items() if n > 0},
            device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    t_flow = time.time() - t1
    print(f"\n  Flow training time: {t_flow:.0f}s ({t_flow/60:.1f}min)")

    # Phase 3: Evaluation
    print(f"\n{'='*60}")
    print("  Phase 3: Evaluation")
    print(f"{'='*60}")

    results = evaluate(backbones, class_models, client_class_counts_fit,
                       test_loader, device)

    print("\n  Results:")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"    {name:24s}: {acc:.2%}")

    best_name = max(results, key=results.get)
    best_acc = results[best_name]

    os.makedirs("results", exist_ok=True)
    out = {
        "alpha": alpha, "seed": seed,
        "bb_epochs": BB_EPOCHS, "feat_dim": FEAT_DIM,
        "flow_pca_dim": FLOW_PCA_DIM, "flow_layers": FLOW_N_LAYERS,
        "flow_hidden": FLOW_HIDDEN, "flow_epochs": FLOW_EPOCHS,
        "results": {k: float(v) for k, v in results.items()},
        "best_name": best_name, "best_acc": float(best_acc),
        "time_backbone": t_bb, "time_flow": t_flow,
    }
    path = f"results/flow_pipeline_a{alpha}_s{seed}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*72}")
    print(f"  BEST: {best_name} = {best_acc:.2%}")
    print(f"  Saved: {path}")
    print(f"{'='*72}\n")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--bb_epochs", type=int, default=300)
    parser.add_argument("--bb_lr", type=float, default=1e-3)
    parser.add_argument("--flow_pca_dim", type=int, default=64)
    parser.add_argument("--flow_layers", type=int, default=6)
    parser.add_argument("--flow_hidden", type=int, default=128)
    parser.add_argument("--flow_epochs", type=int, default=200)
    parser.add_argument("--flow_lr", type=float, default=5e-4)
    args = parser.parse_args()

    global BB_EPOCHS, BB_LR, FLOW_PCA_DIM, FLOW_N_LAYERS, FLOW_HIDDEN, FLOW_EPOCHS, FLOW_LR
    BB_EPOCHS = args.bb_epochs
    BB_LR = args.bb_lr
    FLOW_PCA_DIM = args.flow_pca_dim
    FLOW_N_LAYERS = args.flow_layers
    FLOW_HIDDEN = args.flow_hidden
    FLOW_EPOCHS = args.flow_epochs
    FLOW_LR = args.flow_lr

    run_experiment(args.alpha, args.seed, args.gpu)


if __name__ == "__main__":
    main()