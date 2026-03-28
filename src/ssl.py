"""
Pipeline: Class-Conditional Compact SSL for Intrinsic Knowledge in OFL

Core idea:
  - L_inv:     augmentation invariance (remove noise, keep semantics)
  - L_compact: pull same-class features toward class center
  - L_recon:   reconstruct clean-view features from augmented projections
               (anti-collapse WITHOUT pushing samples apart)
  - L_decor:   class-conditional decorrelation (efficient features)

Why this works for intrinsic knowledge:
  - L_compact ensures density estimation is meaningful (tight clusters)
  - L_recon prevents collapse without uniformity (preserves info, not scatter)
  - L_inv removes augmentation noise (unlike pixel reconstruction)
  - More data per class → better centers, better decoders → monotonically better
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
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

warnings.filterwarnings("ignore")

# ============================================================
# Global Config
# ============================================================
N_CLIENTS = 5
N_CLASSES = 10

BATCH_SIZE = 256
LR = 1e-3
WD = 1e-4
EPOCHS = 300

FEAT_DIM = 256
PROJ_DIM = 256

MIN_SAMPLES = 30
CALIB_RATIO = 0.10

# Compact SSL loss weights
W_INV = 1.0       # invariance
W_COMPACT = 0.5   # compactness (ramp up after warmup)
W_RECON = 2.0     # feature reconstruction (anti-collapse)
W_DECOR = 0.1     # class-conditional decorrelation

COMPACT_WARMUP = 30   # epochs before L_compact kicks in
EMA_DECAY = 0.99      # for class center EMA

# Gaussian / calibration (same as before)
EPS_VAR = 1e-4
EPS_STD = 1e-6
MAX_RELIABILITY = 6.0

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


# ============================================================
# Utils (unchanged)
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
# Transforms
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


# ============================================================
# Dataset: returns (aug1, aug2, clean, label)
# ============================================================
class CompactSSLDataset(Dataset):
    """
    Returns:
        aug1, aug2: two augmented views (for invariance + projection)
        clean:      non-augmented view (reconstruction target)
        label:      class label
    """
    def __init__(self, base_dataset, indices):
        self.data = base_dataset.data
        self.targets = np.array(base_dataset.targets)
        self.indices = list(indices)
        self.ssl_transform = get_ssl_transform()
        self.clean_transform = get_test_transform()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx % len(self.indices)]
        img = Image.fromarray(self.data[real_idx])
        label = int(self.targets[real_idx])

        aug1 = self.ssl_transform(img)
        aug2 = self.ssl_transform(img)
        clean = self.clean_transform(img)

        return aug1, aug2, clean, label


class IndexedClassDataset(Dataset):
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
        y = int(self.targets[real_idx])
        return self.transform(img), y


# ============================================================
# Model
# ============================================================
class GNBasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        gn_groups = min(8, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(gn_groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(gn_groups, out_ch)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(gn_groups, out_ch)
            )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)), inplace=True)
        out = self.gn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class SmallBackbone(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(GNBasicBlock(64, 64, 1), GNBasicBlock(64, 64, 1))
        self.layer2 = nn.Sequential(GNBasicBlock(64, 128, 2), GNBasicBlock(128, 128, 1))
        self.layer3 = nn.Sequential(GNBasicBlock(128, 256, 2), GNBasicBlock(256, 256, 1))
        self.layer4 = nn.Sequential(GNBasicBlock(256, 512, 2), GNBasicBlock(512, 512, 1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, feat_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        # NOTE: NO L2 normalization — density estimation needs Euclidean space
        return x


class Projector(nn.Module):
    def __init__(self, in_dim=256, proj_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)


class PerClassDecoder(nn.Module):
    """Lightweight MLP: z → h_reconstructed"""
    def __init__(self, proj_dim=256, feat_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, feat_dim),
        )

    def forward(self, z):
        return self.net(z)


class CompactSSLModel(nn.Module):
    def __init__(self, local_classes, feat_dim=256, proj_dim=256):
        """
        Args:
            local_classes: list of class indices this client has
        """
        super().__init__()
        self.backbone = SmallBackbone(feat_dim=feat_dim)
        self.projector = Projector(in_dim=feat_dim, proj_dim=proj_dim)

        # per-class decoders
        self.decoders = nn.ModuleDict({
            str(c): PerClassDecoder(proj_dim=proj_dim, feat_dim=feat_dim)
            for c in local_classes
        })

        self.feat_dim = feat_dim
        self.local_classes = sorted(local_classes)

    def encode(self, x):
        return self.backbone(x)

    def project(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return h, z

    def decode(self, z, c):
        return self.decoders[str(c)](z)


# ============================================================
# EMA Class Centers
# ============================================================
class ClassCenters:
    def __init__(self, classes, feat_dim, device, decay=EMA_DECAY):
        self.decay = decay
        self.centers = {c: torch.zeros(feat_dim, device=device) for c in classes}
        self.initialized = {c: False for c in classes}

    @torch.no_grad()
    def update(self, features, labels):
        """Update centers with batch features"""
        for c in self.centers:
            mask = (labels == c)
            if mask.sum() == 0:
                continue
            batch_mean = features[mask].mean(dim=0)
            if not self.initialized[c]:
                self.centers[c] = batch_mean.clone()
                self.initialized[c] = True
            else:
                self.centers[c] = self.decay * self.centers[c] + (1 - self.decay) * batch_mean

    def get(self, c):
        return self.centers[c]


# ============================================================
# Loss Functions
# ============================================================
def invariance_loss(z1, z2):
    """MSE between two views in projection space"""
    return F.mse_loss(z1, z2)


def compactness_loss(features, labels, centers):
    """Pull features toward their class center"""
    loss = 0.0
    count = 0
    for c in centers.centers:
        mask = (labels == c)
        n = mask.sum()
        if n == 0:
            continue
        fc = features[mask]
        mu = centers.get(c).detach()
        loss = loss + ((fc - mu) ** 2).sum(dim=1).mean()
        count += 1
    return loss / max(count, 1)


def reconstruction_loss(z1, z2, h_clean, model, labels):
    """
    Reconstruct clean-view features from augmented projections.
    Per-class decoder ensures class-specific reconstruction.
    This prevents collapse: if all z collapse to one point,
    decoder can't reconstruct diverse h_clean → loss explodes.
    """
    loss = 0.0
    count = 0
    for c_int in model.local_classes:
        mask = (labels == c_int)
        n = mask.sum()
        if n == 0:
            continue
        z1_c = z1[mask]
        z2_c = z2[mask]
        h_target = h_clean[mask].detach()  # stop gradient on target

        h_recon1 = model.decode(z1_c, c_int)
        h_recon2 = model.decode(z2_c, c_int)

        loss = loss + F.mse_loss(h_recon1, h_target) + F.mse_loss(h_recon2, h_target)
        count += 1

    return loss / max(count, 1)


def class_decorrelation_loss(features, labels, classes):
    """
    Decorrelate feature dimensions within each class.
    Unlike VICReg's global covariance, this is class-conditional.
    """
    loss = 0.0
    count = 0
    for c in classes:
        mask = (labels == c)
        n = mask.sum()
        if n < 4:  # need enough samples for covariance
            continue
        fc = features[mask]
        fc_centered = fc - fc.mean(dim=0)
        cov = (fc_centered.T @ fc_centered) / (n - 1)
        loss = loss + off_diagonal(cov).pow(2).mean()
        count += 1
    return loss / max(count, 1)


# ============================================================
# Train Client
# ============================================================
def train_client_compact(client_id, base_dataset, client_indices,
                         client_class_counts, device,
                         epochs=EPOCHS, lr=LR, wd=WD):

    # determine local classes
    local_classes = sorted([
        c for c, n in client_class_counts.items() if n >= MIN_SAMPLES
    ])
    if len(local_classes) == 0:
        local_classes = sorted(client_class_counts.keys())

    dataset = CompactSSLDataset(base_dataset, client_indices)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=len(dataset) >= BATCH_SIZE,
        persistent_workers=True
    )

    model = CompactSSLModel(
        local_classes=local_classes,
        feat_dim=FEAT_DIM,
        proj_dim=PROJ_DIM
    ).to(device)

    centers = ClassCenters(local_classes, FEAT_DIM, device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

    best_state = None
    best_quality = -1e9

    print(f"\n  Client {client_id}: compact SSL on {len(client_indices)} samples, "
          f"classes={local_classes}")

    for ep in range(epochs):
        model.train()
        loss_sum = 0.0
        stat_sum = defaultdict(float)
        n_batch = 0

        # ramp up compactness after warmup
        compact_weight = W_COMPACT * min(1.0, max(0.0, (ep - COMPACT_WARMUP) / 30.0))

        for aug1, aug2, clean, labels in loader:
            aug1 = aug1.to(device, non_blocking=True)
            aug2 = aug2.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                # forward
                h1, z1 = model.project(aug1)
                h2, z2 = model.project(aug2)
                h_clean = model.encode(clean)

                # losses
                l_inv = invariance_loss(z1, z2)
                l_compact = compactness_loss(h1, labels, centers)
                l_recon = reconstruction_loss(z1, z2, h_clean, model, labels)
                l_decor = class_decorrelation_loss(h1, labels, local_classes)

                loss = (W_INV * l_inv
                        + compact_weight * l_compact
                        + W_RECON * l_recon
                        + W_DECOR * l_decor)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            # update centers (after gradient step, using detached features)
            with torch.no_grad():
                centers.update(h1.float().detach(), labels)

            loss_sum += loss.item()
            stat_sum["inv"] += l_inv.item()
            stat_sum["compact"] += l_compact.item()
            stat_sum["recon"] += l_recon.item()
            stat_sum["decor"] += l_decor.item()
            n_batch += 1

        sch.step()
        avg_loss = loss_sum / max(n_batch, 1)

        # quality check
        model.eval()
        with torch.no_grad():
            batch = next(iter(loader))
            x_probe = batch[0][:min(128, len(batch[0]))].to(device)
            y_probe = batch[3][:min(128, len(batch[3]))].to(device)
            feats = model.encode(x_probe)
            feat_std = feats.std(dim=0).mean().item()
            active_ratio = (feats.std(dim=0) > 0.01).float().mean().item()

            # class-specific compactness quality
            intra_var = 0.0
            n_cls = 0
            for c in local_classes:
                mask = (y_probe == c)
                if mask.sum() >= 2:
                    fc = feats[mask]
                    intra_var += fc.var(dim=0).mean().item()
                    n_cls += 1
            intra_var = intra_var / max(n_cls, 1)

            quality_score = active_ratio + 5.0 * feat_std - 0.5 * intra_var

        if quality_score > best_quality:
            best_quality = quality_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (ep + 1) % 50 == 0 or ep == 0:
            def avg(name):
                return stat_sum[name] / max(n_batch, 1)
            print(
                f"    ep {ep+1:3d}/{epochs}  "
                f"loss={avg_loss:.4f}  "
                f"inv={avg('inv'):.4f}  "
                f"compact={avg('compact'):.4f}  "
                f"recon={avg('recon'):.4f}  "
                f"decor={avg('decor'):.4f}  "
                f"feat_std={feat_std:.4f}  active={active_ratio:.1%}  "
                f"intra_var={intra_var:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    model = model.cpu()

    # export final centers to CPU
    final_centers = {c: centers.get(c).detach().cpu() for c in local_classes}

    return model, final_centers, best_quality


# ============================================================
# Feature Extraction
# ============================================================
@torch.no_grad()
def extract_features(backbone_model, loader, device):
    backbone_model = backbone_model.to(device).eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        f = backbone_model.encode(x).cpu()
        feats.append(f)
        labels.append(y)
    backbone_model = backbone_model.cpu()
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


# ============================================================
# Expert Fitting + Calibration (same structure, minor cleanup)
# ============================================================
@torch.no_grad()
def fit_client_intrinsic_models(client_id, backbone_model,
                                ema_centers,
                                base_dataset,
                                fit_indices, calib_indices,
                                client_class_counts_fit,
                                device, min_samples=MIN_SAMPLES):
    # fit features
    fit_ds = IndexedClassDataset(base_dataset, fit_indices, transform=get_test_transform())
    fit_loader = DataLoader(fit_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    fit_feats, fit_labels = extract_features(backbone_model, fit_loader, device)

    # calib features
    calib_feats, calib_labels = None, None
    if len(calib_indices) > 0:
        calib_ds = IndexedClassDataset(base_dataset, calib_indices, transform=get_test_transform())
        calib_loader = DataLoader(calib_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
        calib_feats, calib_labels = extract_features(backbone_model, calib_loader, device)

    models = {}
    for c in sorted(client_class_counts_fit[client_id].keys()):
        n = client_class_counts_fit[client_id][c]
        if n < min_samples:
            continue

        mask_fit = (fit_labels == c)
        fc = fit_feats[mask_fit]
        if len(fc) < min_samples:
            continue

        # use EMA center if available, otherwise batch mean
        if c in ema_centers:
            mu = ema_centers[c]
        else:
            mu = fc.mean(dim=0)

        var = fc.var(dim=0, unbiased=False).clamp(min=EPS_VAR)

        # fit-set energy stats
        energy = ((fc - mu) ** 2 / var).sum(dim=1)
        fit_e_mean = energy.mean()
        fit_e_std = energy.std(unbiased=False).clamp(min=EPS_STD)

        pack = {
            "mu": mu,
            "var": var,
            "fit_e_mean": fit_e_mean,
            "fit_e_std": fit_e_std,
            "n": int(n),
        }

        # calibration
        if calib_feats is not None and len(calib_feats) > 0:
            energy_all = ((calib_feats - mu) ** 2 / var).sum(dim=1)
            pos_mask = (calib_labels == c)
            neg_mask = (calib_labels != c)
            n_pos = int(pos_mask.sum().item())
            n_neg = int(neg_mask.sum().item())

            if n_pos >= 3:
                pos_e = energy_all[pos_mask]
                pos_mean = pos_e.mean()
                pos_std = pos_e.std(unbiased=False).clamp(min=EPS_STD)
            else:
                pos_mean = fit_e_mean
                pos_std = fit_e_std

            if n_neg >= 5:
                neg_e = energy_all[neg_mask]
                neg_mean = neg_e.mean()
                neg_std = neg_e.std(unbiased=False).clamp(min=EPS_STD)
            else:
                neg_mean = pos_mean + 2.5 * pos_std
                neg_std = pos_std * 1.5 + 1.0

            threshold = 0.5 * (pos_mean + neg_mean)
            scale = torch.sqrt(0.5 * (pos_std ** 2 + neg_std ** 2)).clamp(min=1.0)

            sep = ((neg_mean - pos_mean) / (pos_std + neg_std + EPS_STD)).item()
            sep = max(0.0, sep)
            support = math.log(n + 1.0)
            pos_quality = min(1.0, n_pos / 20.0) if n_pos > 0 else 0.0
            neg_quality = min(1.0, n_neg / 50.0) if n_neg > 0 else 0.0

            reliability = sep * support * (0.5 + 0.5 * pos_quality) * (0.5 + 0.5 * neg_quality)
            reliability = min(reliability, MAX_RELIABILITY)

            pack.update({
                "calib_pos_mean": pos_mean,
                "calib_pos_std": pos_std,
                "calib_neg_mean": neg_mean,
                "calib_neg_std": neg_std,
                "threshold": threshold,
                "scale": scale,
                "reliability": float(reliability),
                "n_pos_calib": int(n_pos),
                "n_neg_calib": int(n_neg),
            })
        else:
            pack.update({
                "calib_pos_mean": fit_e_mean,
                "calib_pos_std": fit_e_std,
                "calib_neg_mean": fit_e_mean + 2.5 * fit_e_std,
                "calib_neg_std": fit_e_std * 1.5 + 1.0,
                "threshold": fit_e_mean + 1.25 * fit_e_std,
                "scale": max(float(fit_e_std.item()), 1.0),
                "reliability": 0.5 * math.log(n + 1.0),
                "n_pos_calib": 0,
                "n_neg_calib": 0,
            })

        models[c] = pack

    print(f"  Client {client_id}: {len(models)} intrinsic Gaussian models")
    for c, pack in models.items():
        rel = pack['reliability']
        print(
            f"    c{c}: n={pack['n']:5d}, "
            f"rel={rel:.3f}, "
            f"pos_calib={pack['n_pos_calib']}, neg_calib={pack['n_neg_calib']}"
        )

    return models


# ============================================================
# Evaluation (same multi-strategy evaluation)
# ============================================================
@torch.no_grad()
def evaluate_intrinsic(backbones, intrinsic_models, client_class_counts_fit,
                       test_loader, device):
    test_labels = None
    n_test = len(test_loader.dataset)

    raw_scores = torch.full((N_CLIENTS, n_test, N_CLASSES), float("-inf"))
    calib_scores = torch.full((N_CLIENTS, n_test, N_CLASSES), float("-inf"))
    calib_probs = torch.zeros(N_CLIENTS, n_test, N_CLASSES)
    reliabilities = torch.zeros(N_CLIENTS, N_CLASSES)

    for k in range(N_CLIENTS):
        backbone = backbones[k].to(device).eval()
        feats = []
        labels = []
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            f = backbone.encode(x).cpu()
            feats.append(f)
            if test_labels is None:
                labels.append(y)
        feats = torch.cat(feats, dim=0)
        if test_labels is None:
            test_labels = torch.cat(labels).numpy()

        for c, pack in intrinsic_models[k].items():
            mu = pack["mu"]
            var = pack["var"]
            energy = ((feats - mu) ** 2 / var).sum(dim=1)

            raw_z = (energy - pack["fit_e_mean"]) / pack["fit_e_std"]
            raw_scores[k, :, c] = -raw_z

            threshold = pack["threshold"]
            scale = pack["scale"]
            margin = -(energy - threshold) / scale
            calib_scores[k, :, c] = margin
            calib_probs[k, :, c] = torch.sigmoid(margin)
            reliabilities[k, c] = float(pack["reliability"])

        backbones[k] = backbone.cpu()

    results = {}

    # S1: raw max-score
    best_per_class, _ = raw_scores.max(dim=0)
    preds_s1 = best_per_class.argmax(dim=1).numpy()
    results["S1_max_score"] = float((preds_s1 == test_labels).mean())

    # S2: weighted logN
    weighted = torch.zeros(n_test, N_CLASSES)
    denom = torch.zeros(N_CLASSES)
    for k in range(N_CLIENTS):
        for c in intrinsic_models[k].keys():
            w = math.log(client_class_counts_fit[k][c] + 1.0)
            valid = torch.isfinite(raw_scores[k, :, c])
            if valid.any():
                weighted[valid, c] += raw_scores[k, valid, c] * w
                denom[c] += w
    for c in range(N_CLASSES):
        if denom[c] > 0:
            weighted[:, c] /= denom[c]
        else:
            weighted[:, c] = float("-inf")
    preds_s2 = weighted.argmax(dim=1).numpy()
    results["S2_weighted_logN"] = float((preds_s2 == test_labels).mean())

    # S3: top expert only
    top = torch.full((n_test, N_CLASSES), float("-inf"))
    for c in range(N_CLASSES):
        best_k, best_n = -1, -1
        for k in range(N_CLIENTS):
            if c in intrinsic_models[k]:
                n = intrinsic_models[k][c]["n"]
                if n > best_n:
                    best_n = n
                    best_k = k
        if best_k >= 0:
            top[:, c] = raw_scores[best_k, :, c]
    preds_s3 = top.argmax(dim=1).numpy()
    results["S3_top_expert"] = float((preds_s3 == test_labels).mean())

    # S4: calibrated reliability prob fusion
    fused_prob = torch.zeros(n_test, N_CLASSES)
    fused_den = torch.zeros(N_CLASSES)
    for k in range(N_CLIENTS):
        for c in intrinsic_models[k].keys():
            w = reliabilities[k, c].item()
            if w <= 0:
                continue
            fused_prob[:, c] += calib_probs[k, :, c] * w
            fused_den[c] += w
    for c in range(N_CLASSES):
        if fused_den[c] > 0:
            fused_prob[:, c] /= fused_den[c]
    preds_s4 = fused_prob.argmax(dim=1).numpy()
    results["S4_calib_rel_prob"] = float((preds_s4 == test_labels).mean())

    # S5: calibrated margin fusion
    fused_margin = torch.full((n_test, N_CLASSES), float("-inf"))
    for c in range(N_CLASSES):
        agg_num = torch.zeros(n_test)
        agg_den = 0.0
        has_any = False
        for k in range(N_CLIENTS):
            if c in intrinsic_models[k]:
                w = reliabilities[k, c].item()
                if w <= 0:
                    continue
                agg_num += calib_scores[k, :, c] * w
                agg_den += w
                has_any = True
        if has_any and agg_den > 0:
            fused_margin[:, c] = agg_num / agg_den
    preds_s5 = fused_margin.argmax(dim=1).numpy()
    results["S5_calib_rel_margin"] = float((preds_s5 == test_labels).mean())

    return results


# ============================================================
# Main
# ============================================================
def run_experiment(alpha, seed, gpu):
    seed_everything(seed)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*72}")
    print(f"  Pipeline: Compact SSL (Intrinsic Knowledge)")
    print(f"  alpha={alpha}  seed={seed}  epochs={EPOCHS}")
    print(f"  W_INV={W_INV}  W_COMPACT={W_COMPACT}  W_RECON={W_RECON}  W_DECOR={W_DECOR}")
    print(f"  COMPACT_WARMUP={COMPACT_WARMUP}")
    print(f"{'='*72}")

    train_base = datasets.CIFAR10("./data", train=True, download=True)
    targets = np.array(train_base.targets)

    client_indices_all, client_class_counts_all = dirichlet_split(
        targets, N_CLIENTS, alpha, seed=seed
    )

    client_train_idx = {}
    client_calib_idx = {}
    client_class_counts_fit = defaultdict(lambda: defaultdict(int))

    print("\n  Data distribution:")
    for k in range(N_CLIENTS):
        idxs_all = client_indices_all[k]
        tr_idx, ca_idx = split_client_train_calib(
            targets, idxs_all, calib_ratio=CALIB_RATIO, seed=seed + 1000 + k
        )
        client_train_idx[k] = tr_idx
        client_calib_idx[k] = ca_idx

        for idx in tr_idx:
            c = int(targets[idx])
            client_class_counts_fit[k][c] += 1

        ccc = client_class_counts_all.get(k, {})
        n_cls = sum(v > 0 for v in ccc.values())
        n_smp = sum(ccc.values())
        top = sorted(ccc.items(), key=lambda x: -x[1])[:5]
        top_str = ", ".join(f"c{c}={n}" for c, n in top)
        print(f"    Client {k}: {n_cls:2d} cls, {n_smp:5d} smp  "
              f"train={len(tr_idx):5d} calib={len(ca_idx):4d}  top: {top_str}")

    test_ds = datasets.CIFAR10("./data", train=False, transform=get_test_transform())
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
    )

    # Phase 1: train local backbone with compact SSL
    print(f"\n{'='*60}")
    print("  Phase 1: Local Compact SSL Training")
    print(f"{'='*60}")

    backbones = {}
    ema_centers_all = {}
    bb_scores = {}
    t0 = time.time()

    for k in range(N_CLIENTS):
        idxs = client_train_idx[k]
        if len(idxs) < 2:
            continue

        ccc = {c: n for c, n in client_class_counts_fit[k].items() if n > 0}

        model, centers, best_quality = train_client_compact(
            k, train_base, idxs, ccc, device,
            epochs=EPOCHS, lr=LR, wd=WD
        )
        backbones[k] = model
        ema_centers_all[k] = centers
        bb_scores[k] = best_quality

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    t_bb = time.time() - t0
    print(f"\n  Training time: {t_bb:.0f}s ({t_bb/60:.1f}min)")

    # Phase 2: fit + calibrate Gaussian experts
    print(f"\n{'='*60}")
    print("  Phase 2: Intrinsic Gaussian Fitting + Calibration")
    print(f"{'='*60}")

    intrinsic_models = {}
    for k in range(N_CLIENTS):
        intrinsic_models[k] = fit_client_intrinsic_models(
            client_id=k,
            backbone_model=backbones[k],
            ema_centers=ema_centers_all.get(k, {}),
            base_dataset=train_base,
            fit_indices=client_train_idx[k],
            calib_indices=client_calib_idx[k],
            client_class_counts_fit=client_class_counts_fit,
            device=device,
            min_samples=MIN_SAMPLES
        )

    # Phase 3: evaluation
    print(f"\n{'='*60}")
    print("  Phase 3: Evaluation")
    print(f"{'='*60}")

    results = evaluate_intrinsic(
        backbones=backbones,
        intrinsic_models=intrinsic_models,
        client_class_counts_fit=client_class_counts_fit,
        test_loader=test_loader,
        device=device
    )

    print("\n  Results:")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"    {name:24s}: {acc:.2%}")

    best_name = max(results, key=results.get)
    best_acc = results[best_name]

    # save
    os.makedirs("results", exist_ok=True)
    out = {
        "alpha": alpha,
        "seed": seed,
        "epochs": EPOCHS,
        "feat_dim": FEAT_DIM,
        "proj_dim": PROJ_DIM,
        "min_samples": MIN_SAMPLES,
        "calib_ratio": CALIB_RATIO,
        "W_INV": W_INV,
        "W_COMPACT": W_COMPACT,
        "W_RECON": W_RECON,
        "W_DECOR": W_DECOR,
        "bb_scores": {str(k): float(v) for k, v in bb_scores.items()},
        "results": {k: float(v) for k, v in results.items()},
        "best_name": best_name,
        "best_acc": float(best_acc),
        "time_backbone": t_bb,
    }
    outpath = f"results/compact_ssl_a{alpha}_s{seed}.json"
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*72}")
    print(f"  BEST: {best_name} = {best_acc:.2%}")
    print(f"  Saved: {outpath}")
    print(f"{'='*72}\n")

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)

    parser.add_argument("--feat_dim", type=int, default=256)
    parser.add_argument("--proj_dim", type=int, default=256)

    parser.add_argument("--min_samples", type=int, default=30)
    parser.add_argument("--calib_ratio", type=float, default=0.10)

    parser.add_argument("--w_inv", type=float, default=1.0)
    parser.add_argument("--w_compact", type=float, default=0.5)
    parser.add_argument("--w_recon", type=float, default=2.0)
    parser.add_argument("--w_decor", type=float, default=0.1)
    parser.add_argument("--compact_warmup", type=int, default=30)

    args = parser.parse_args()

    global EPOCHS, LR, WD, FEAT_DIM, PROJ_DIM, MIN_SAMPLES, CALIB_RATIO
    global W_INV, W_COMPACT, W_RECON, W_DECOR, COMPACT_WARMUP

    EPOCHS = args.epochs
    LR = args.lr
    WD = args.wd
    FEAT_DIM = args.feat_dim
    PROJ_DIM = args.proj_dim
    MIN_SAMPLES = args.min_samples
    CALIB_RATIO = args.calib_ratio

    W_INV = args.w_inv
    W_COMPACT = args.w_compact
    W_RECON = args.w_recon
    W_DECOR = args.w_decor
    COMPACT_WARMUP = args.compact_warmup

    run_experiment(args.alpha, args.seed, args.gpu)


if __name__ == "__main__":
    main()