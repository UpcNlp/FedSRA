"""
Pipeline: Distribution-Shifted One-Class SSL for Intrinsic Knowledge in OFL

Core insight:
  Single-class learning objective = distinguish "natural instances of class c"
  from "distribution-shifted versions of class c".

  Shifted augmentations (rotation 90/180/270, patch shuffle, CutPerm) break
  class-specific spatial structure while preserving low-level statistics.
  The encoder MUST learn class-specific semantics to distinguish them.

  Anti-collapse is a CONSEQUENCE of this objective, not an extra regularizer:
  if encoder collapses, it cannot distinguish natural from shifted → loss = log(2).

Properties:
  1. Single-class learnable: negatives are self-generated from the same class
  2. Monotonically improving: more data → better natural-boundary estimation
  3. Semantic-sensitive, noise-insensitive: standard augs = noise, shifts = semantics
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
MIN_SAMPLES = 30
CALIB_RATIO = 0.10

# Loss weights
W_OC = 1.0        # one-class contrastive
W_INV = 0.5       # augmentation invariance
W_VAR = 0.1       # minimal variance (safety net, not primary anti-collapse)
SIGMA_MIN = 0.05   # much smaller than VICReg's 1.0

# Calibration
EPS_VAR = 1e-4
EPS_STD = 1e-6
MAX_RELIABILITY = 6.0

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

EMA_DECAY = 0.99


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
# Distribution-Shifting Augmentations
# ============================================================
class PatchShuffle:
    """Shuffle image patches on a grid. Destroys spatial structure."""
    def __init__(self, grid_size=4):
        self.grid_size = grid_size

    def __call__(self, img_tensor):
        # img_tensor: (C, H, W)
        C, H, W = img_tensor.shape
        gh, gw = H // self.grid_size, W // self.grid_size

        # split into patches
        patches = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                patch = img_tensor[:, i*gh:(i+1)*gh, j*gw:(j+1)*gw]
                patches.append(patch)

        # shuffle
        idx = list(range(len(patches)))
        random.shuffle(idx)

        # reassemble
        result = torch.zeros_like(img_tensor)
        for pos, src in enumerate(idx):
            i, j = pos // self.grid_size, pos % self.grid_size
            result[:, i*gh:(i+1)*gh, j*gw:(j+1)*gw] = patches[src]

        return result


class CutPerm:
    """Cut image into 4 quadrants and randomly permute them."""
    def __call__(self, img_tensor):
        C, H, W = img_tensor.shape
        mh, mw = H // 2, W // 2

        quads = [
            img_tensor[:, :mh, :mw],    # top-left
            img_tensor[:, :mh, mw:],     # top-right
            img_tensor[:, mh:, :mw],     # bottom-left
            img_tensor[:, mh:, mw:],     # bottom-right
        ]

        # random permutation (excluding identity)
        while True:
            perm = list(range(4))
            random.shuffle(perm)
            if perm != [0, 1, 2, 3]:
                break

        result = torch.zeros_like(img_tensor)
        positions = [(0, 0), (0, mw), (mh, 0), (mh, mw)]
        for dst_idx, src_idx in enumerate(perm):
            r, c_pos = positions[dst_idx]
            result[:, r:r+mh, c_pos:c_pos+mw] = quads[src_idx]

        return result


class DistributionShift:
    """
    Apply a random distribution-shifting augmentation.
    These break class-specific spatial structure while preserving
    low-level statistics — forcing the encoder to learn semantics.
    """
    def __init__(self):
        self.patch_shuffle = PatchShuffle(grid_size=4)
        self.cut_perm = CutPerm()

    def __call__(self, img_tensor):
        choice = random.randint(0, 2)

        if choice == 0:
            # Rotation by 90, 180, or 270 degrees
            k = random.choice([1, 2, 3])
            return torch.rot90(img_tensor, k, [1, 2])

        elif choice == 1:
            # Patch shuffle
            return self.patch_shuffle(img_tensor)

        else:
            # CutPerm
            return self.cut_perm(img_tensor)


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
# Dataset
# ============================================================
class ShiftedOneClassDataset(Dataset):
    """
    Returns:
        x_pos1: standard augmented view 1 (positive)
        x_pos2: standard augmented view 2 (positive, for invariance)
        x_neg:  distribution-shifted view (negative)
        label:  class label
    """
    def __init__(self, base_dataset, indices):
        self.data = base_dataset.data
        self.targets = np.array(base_dataset.targets)
        self.indices = list(indices)
        self.ssl_transform = get_ssl_transform()
        self.clean_transform = get_test_transform()
        self.dist_shift = DistributionShift()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx % len(self.indices)]
        img = Image.fromarray(self.data[real_idx])
        label = int(self.targets[real_idx])

        # Two standard augmented views (positives)
        x_pos1 = self.ssl_transform(img)
        x_pos2 = self.ssl_transform(img)

        # Distribution-shifted view (negative):
        # first apply clean transform (normalize), then shift
        x_clean = self.clean_transform(img)
        x_neg = self.dist_shift(x_clean)

        return x_pos1, x_pos2, x_neg, label


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
        # No L2 norm — need Euclidean space for density estimation
        return x


class OneClassModel(nn.Module):
    """
    Encoder + per-class score head.
    Score head: maps features to scalar "naturalness score" for each class.
    """
    def __init__(self, local_classes, feat_dim=256):
        super().__init__()
        self.backbone = SmallBackbone(feat_dim=feat_dim)
        self.feat_dim = feat_dim
        self.local_classes = sorted(local_classes)

        # Per-class binary classifier: "is this a natural instance of class c?"
        self.heads = nn.ModuleDict({
            str(c): nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
            )
            for c in local_classes
        })

    def encode(self, x):
        return self.backbone(x)

    def score(self, h, c):
        """Scalar naturalness score for class c"""
        return self.heads[str(c)](h).squeeze(-1)


# ============================================================
# EMA Class Centers (for downstream Gaussian fitting)
# ============================================================
class ClassCenters:
    def __init__(self, classes, feat_dim, device, decay=EMA_DECAY):
        self.decay = decay
        self.centers = {c: torch.zeros(feat_dim, device=device) for c in classes}
        self.initialized = {c: False for c in classes}

    @torch.no_grad()
    def update(self, features, labels):
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
def one_class_loss(model, h_pos, h_neg, labels, local_classes):
    """
    Core loss: for each class c, the score head should output
    high values for natural samples, low values for shifted samples.

    If encoder collapses → h_pos ≈ h_neg → score_pos ≈ score_neg
    → loss ≈ log(2) → gradient pushes encoder to distinguish them
    → anti-collapse is automatic.
    """
    loss = 0.0
    count = 0
    pos_acc = 0.0
    neg_acc = 0.0

    for c in local_classes:
        mask = (labels == c)
        n = mask.sum()
        if n == 0:
            continue

        s_pos = model.score(h_pos[mask], c)
        s_neg = model.score(h_neg[mask], c)

        loss_pos = F.binary_cross_entropy_with_logits(s_pos, torch.ones_like(s_pos))
        loss_neg = F.binary_cross_entropy_with_logits(s_neg, torch.zeros_like(s_neg))

        loss = loss + loss_pos + loss_neg
        count += 1

        with torch.no_grad():
            pos_acc += (s_pos > 0).float().mean().item()
            neg_acc += (s_neg < 0).float().mean().item()

    loss = loss / max(count, 1)
    pos_acc = pos_acc / max(count, 1)
    neg_acc = neg_acc / max(count, 1)

    return loss, pos_acc, neg_acc


def invariance_loss(h1, h2):
    """MSE between two standard-augmented views in feature space"""
    return F.mse_loss(h1, h2)


def minimal_variance_loss(features, sigma_min=SIGMA_MIN):
    """
    Safety-net variance regularizer. Much weaker than VICReg (σ_min=0.05 vs 1.0).
    Prevents pathological collapse, but does NOT enforce uniformity.
    The primary anti-collapse comes from the one-class loss.
    """
    std = torch.sqrt(features.var(dim=0) + 1e-4)
    return torch.mean(F.relu(sigma_min - std))


# ============================================================
# Train Client
# ============================================================
def train_client_oneclass(client_id, base_dataset, client_indices,
                          client_class_counts, device,
                          epochs=EPOCHS, lr=LR, wd=WD):

    local_classes = sorted([
        c for c, n in client_class_counts.items() if n >= MIN_SAMPLES
    ])
    if len(local_classes) == 0:
        local_classes = sorted(client_class_counts.keys())

    dataset = ShiftedOneClassDataset(base_dataset, client_indices)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=len(dataset) >= BATCH_SIZE,
        persistent_workers=True
    )

    model = OneClassModel(
        local_classes=local_classes,
        feat_dim=FEAT_DIM,
    ).to(device)

    centers = ClassCenters(local_classes, FEAT_DIM, device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

    best_state = None
    best_quality = -1e9

    print(f"\n  Client {client_id}: one-class SSL on {len(client_indices)} samples, "
          f"classes={local_classes}")

    for ep in range(epochs):
        model.train()
        loss_sum = 0.0
        stat_sum = defaultdict(float)
        n_batch = 0

        for x_pos1, x_pos2, x_neg, labels in loader:
            x_pos1 = x_pos1.to(device, non_blocking=True)
            x_pos2 = x_pos2.to(device, non_blocking=True)
            x_neg = x_neg.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                h_pos1 = model.encode(x_pos1)
                h_pos2 = model.encode(x_pos2)
                h_neg = model.encode(x_neg)

                # Core: one-class contrastive loss
                l_oc, p_acc, n_acc = one_class_loss(
                    model, h_pos1, h_neg, labels, local_classes
                )

                # Augmentation invariance (in feature space)
                l_inv = invariance_loss(h_pos1, h_pos2)

                # Minimal variance (safety net)
                l_var = minimal_variance_loss(h_pos1, sigma_min=SIGMA_MIN)

                loss = W_OC * l_oc + W_INV * l_inv + W_VAR * l_var

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            # update centers
            with torch.no_grad():
                centers.update(h_pos1.float().detach(), labels)

            loss_sum += loss.item()
            stat_sum["oc"] += l_oc.item()
            stat_sum["inv"] += l_inv.item()
            stat_sum["var"] += l_var.item()
            stat_sum["p_acc"] += p_acc
            stat_sum["n_acc"] += n_acc
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

            intra_var = 0.0
            n_cls = 0
            for c in local_classes:
                mask = (y_probe == c)
                if mask.sum() >= 2:
                    intra_var += feats[mask].var(dim=0).mean().item()
                    n_cls += 1
            intra_var = intra_var / max(n_cls, 1)

            quality_score = active_ratio + 5.0 * feat_std

        if quality_score > best_quality:
            best_quality = quality_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (ep + 1) % 50 == 0 or ep == 0:
            def avg(name):
                return stat_sum[name] / max(n_batch, 1)
            print(
                f"    ep {ep+1:3d}/{epochs}  "
                f"loss={avg_loss:.4f}  "
                f"oc={avg('oc'):.4f}  "
                f"inv={avg('inv'):.4f}  "
                f"var={avg('var'):.4f}  "
                f"p_acc={avg('p_acc'):.1%}  n_acc={avg('n_acc'):.1%}  "
                f"feat_std={feat_std:.4f}  active={active_ratio:.1%}  "
                f"intra_var={intra_var:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    model = model.cpu()
    final_centers = {c: centers.get(c).detach().cpu() for c in local_classes}

    return model, final_centers, best_quality


# ============================================================
# Feature Extraction
# ============================================================
@torch.no_grad()
def extract_features(model, loader, device):
    model = model.to(device).eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        f = model.encode(x).cpu()
        feats.append(f)
        labels.append(y)
    model = model.cpu()
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


# ============================================================
# Expert Fitting + Calibration
# ============================================================
@torch.no_grad()
def fit_client_intrinsic_models(client_id, model,
                                ema_centers,
                                base_dataset,
                                fit_indices, calib_indices,
                                client_class_counts_fit,
                                device, min_samples=MIN_SAMPLES):
    fit_ds = IndexedClassDataset(base_dataset, fit_indices, transform=get_test_transform())
    fit_loader = DataLoader(fit_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    fit_feats, fit_labels = extract_features(model, fit_loader, device)

    calib_feats, calib_labels = None, None
    if len(calib_indices) > 0:
        calib_ds = IndexedClassDataset(base_dataset, calib_indices, transform=get_test_transform())
        calib_loader = DataLoader(calib_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
        calib_feats, calib_labels = extract_features(model, calib_loader, device)

    # also compute score-head based scores for calibration
    model_dev = model.to(device)

    models = {}
    for c in sorted(client_class_counts_fit[client_id].keys()):
        n = client_class_counts_fit[client_id][c]
        if n < min_samples:
            continue

        mask_fit = (fit_labels == c)
        fc = fit_feats[mask_fit]
        if len(fc) < min_samples:
            continue

        mu = ema_centers[c] if c in ema_centers else fc.mean(dim=0)
        var = fc.var(dim=0, unbiased=False).clamp(min=EPS_VAR)

        energy = ((fc - mu) ** 2 / var).sum(dim=1)
        fit_e_mean = energy.mean()
        fit_e_std = energy.std(unbiased=False).clamp(min=EPS_STD)

        # score-head statistics on fit set
        if str(c) in model_dev.heads:
            scores_fit = model_dev.score(fc.to(device), c).cpu()
            score_fit_mean = scores_fit.mean()
            score_fit_std = scores_fit.std().clamp(min=EPS_STD)
        else:
            score_fit_mean = torch.tensor(0.0)
            score_fit_std = torch.tensor(1.0)

        pack = {
            "mu": mu,
            "var": var,
            "fit_e_mean": fit_e_mean,
            "fit_e_std": fit_e_std,
            "score_fit_mean": score_fit_mean,
            "score_fit_std": score_fit_std,
            "n": int(n),
            "has_head": str(c) in model_dev.heads,
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

            # score-head calibration
            if pack["has_head"]:
                scores_calib = model_dev.score(calib_feats.to(device), c).cpu()
                if n_pos >= 3:
                    score_pos_mean = scores_calib[pos_mask].mean()
                else:
                    score_pos_mean = score_fit_mean
                if n_neg >= 3:
                    score_neg_mean = scores_calib[neg_mask].mean()
                else:
                    score_neg_mean = score_fit_mean - 2.0 * score_fit_std

                score_threshold = 0.5 * (score_pos_mean + score_neg_mean)
                score_scale = max(float((score_pos_mean - score_neg_mean).abs().item()), 0.1)
                pack["score_threshold"] = score_threshold
                pack["score_scale"] = score_scale

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

    model_dev = model_dev.cpu()

    print(f"  Client {client_id}: {len(models)} intrinsic models")
    for c, pack in models.items():
        print(
            f"    c{c}: n={pack['n']:5d}, "
            f"rel={pack['reliability']:.3f}, "
            f"pos_calib={pack['n_pos_calib']}, neg_calib={pack['n_neg_calib']}, "
            f"has_head={pack['has_head']}"
        )

    return models


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate_intrinsic(models, intrinsic_models, client_class_counts_fit,
                       test_loader, device):
    test_labels = None
    n_test = len(test_loader.dataset)

    # Gaussian-based scores
    raw_scores = torch.full((N_CLIENTS, n_test, N_CLASSES), float("-inf"))
    calib_scores = torch.full((N_CLIENTS, n_test, N_CLASSES), float("-inf"))
    calib_probs = torch.zeros(N_CLIENTS, n_test, N_CLASSES)
    reliabilities = torch.zeros(N_CLIENTS, N_CLASSES)

    # Score-head based scores
    head_scores = torch.full((N_CLIENTS, n_test, N_CLASSES), float("-inf"))
    head_probs = torch.zeros(N_CLIENTS, n_test, N_CLASSES)

    for k in range(N_CLIENTS):
        model = models[k].to(device).eval()

        feats = []
        labels = []
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            f = model.encode(x).cpu()
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

            # Gaussian z-score
            raw_z = (energy - pack["fit_e_mean"]) / pack["fit_e_std"]
            raw_scores[k, :, c] = -raw_z

            # calibrated margin
            threshold = pack["threshold"]
            scale = pack["scale"]
            margin = -(energy - threshold) / scale
            calib_scores[k, :, c] = margin
            calib_probs[k, :, c] = torch.sigmoid(margin)
            reliabilities[k, c] = float(pack["reliability"])

            # score head
            if pack["has_head"]:
                s = model.score(feats.to(device), c).cpu()
                head_scores[k, :, c] = s

                if "score_threshold" in pack:
                    s_calib = (s - pack["score_threshold"]) / pack["score_scale"]
                    head_probs[k, :, c] = torch.sigmoid(s_calib)
                else:
                    head_probs[k, :, c] = torch.sigmoid(s)

        models[k] = model.cpu()

    results = {}

    # ---- Gaussian-based strategies ----

    # S1: raw max
    best_per_class, _ = raw_scores.max(dim=0)
    preds = best_per_class.argmax(dim=1).numpy()
    results["S1_gauss_max"] = float((preds == test_labels).mean())

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
    preds = weighted.argmax(dim=1).numpy()
    results["S2_gauss_wlogN"] = float((preds == test_labels).mean())

    # S3: top expert
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
    preds = top.argmax(dim=1).numpy()
    results["S3_gauss_topK"] = float((preds == test_labels).mean())

    # S4: calibrated reliability prob
    fused = torch.zeros(n_test, N_CLASSES)
    fden = torch.zeros(N_CLASSES)
    for k in range(N_CLIENTS):
        for c in intrinsic_models[k].keys():
            w = reliabilities[k, c].item()
            if w <= 0:
                continue
            fused[:, c] += calib_probs[k, :, c] * w
            fden[c] += w
    for c in range(N_CLASSES):
        if fden[c] > 0:
            fused[:, c] /= fden[c]
    preds = fused.argmax(dim=1).numpy()
    results["S4_gauss_calib"] = float((preds == test_labels).mean())

    # ---- Score-head based strategies ----

    # S5: score head max
    best_head, _ = head_scores.max(dim=0)
    preds = best_head.argmax(dim=1).numpy()
    results["S5_head_max"] = float((preds == test_labels).mean())

    # S6: score head weighted by reliability
    fused_head = torch.zeros(n_test, N_CLASSES)
    fden_head = torch.zeros(N_CLASSES)
    for k in range(N_CLIENTS):
        for c in intrinsic_models[k].keys():
            if not intrinsic_models[k][c]["has_head"]:
                continue
            w = reliabilities[k, c].item()
            if w <= 0:
                w = 0.1  # small fallback for score heads
            valid = torch.isfinite(head_scores[k, :, c])
            if valid.any():
                fused_head[valid, c] += head_probs[k, valid, c] * w
                fden_head[c] += w
    for c in range(N_CLASSES):
        if fden_head[c] > 0:
            fused_head[:, c] /= fden_head[c]
    preds = fused_head.argmax(dim=1).numpy()
    results["S6_head_calib"] = float((preds == test_labels).mean())

    # S7: score head top expert only
    top_head = torch.full((n_test, N_CLASSES), float("-inf"))
    for c in range(N_CLASSES):
        best_k, best_n = -1, -1
        for k in range(N_CLIENTS):
            if c in intrinsic_models[k] and intrinsic_models[k][c]["has_head"]:
                n = intrinsic_models[k][c]["n"]
                if n > best_n:
                    best_n = n
                    best_k = k
        if best_k >= 0:
            top_head[:, c] = head_scores[best_k, :, c]
    preds = top_head.argmax(dim=1).numpy()
    results["S7_head_topK"] = float((preds == test_labels).mean())

    # S8: ensemble: head + Gaussian (arithmetic mean of probs)
    ensemble = torch.zeros(n_test, N_CLASSES)
    for c in range(N_CLASSES):
        gauss_p = fused[:, c] if fden[c] > 0 else torch.zeros(n_test)
        head_p = fused_head[:, c] if fden_head[c] > 0 else torch.zeros(n_test)

        if fden[c] > 0 and fden_head[c] > 0:
            ensemble[:, c] = 0.5 * gauss_p + 0.5 * head_p
        elif fden[c] > 0:
            ensemble[:, c] = gauss_p
        elif fden_head[c] > 0:
            ensemble[:, c] = head_p
    preds = ensemble.argmax(dim=1).numpy()
    results["S8_ensemble"] = float((preds == test_labels).mean())

    return results


# ============================================================
# Main
# ============================================================
def run_experiment(alpha, seed, gpu):
    seed_everything(seed)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*72}")
    print(f"  Pipeline: Distribution-Shifted One-Class SSL")
    print(f"  alpha={alpha}  seed={seed}  epochs={EPOCHS}")
    print(f"  W_OC={W_OC}  W_INV={W_INV}  W_VAR={W_VAR}  SIGMA_MIN={SIGMA_MIN}")
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

    # Phase 1: train
    print(f"\n{'='*60}")
    print("  Phase 1: One-Class SSL Training")
    print(f"{'='*60}")

    client_models = {}
    ema_centers_all = {}
    bb_scores = {}
    t0 = time.time()

    for k in range(N_CLIENTS):
        idxs = client_train_idx[k]
        if len(idxs) < 2:
            continue

        ccc = {c: n for c, n in client_class_counts_fit[k].items() if n > 0}

        model, centers, best_quality = train_client_oneclass(
            k, train_base, idxs, ccc, device,
            epochs=EPOCHS, lr=LR, wd=WD
        )
        client_models[k] = model
        ema_centers_all[k] = centers
        bb_scores[k] = best_quality

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    t_train = time.time() - t0
    print(f"\n  Training time: {t_train:.0f}s ({t_train/60:.1f}min)")

    # Phase 2: fit + calibrate
    print(f"\n{'='*60}")
    print("  Phase 2: Intrinsic Model Fitting + Calibration")
    print(f"{'='*60}")

    intrinsic_models = {}
    for k in range(N_CLIENTS):
        intrinsic_models[k] = fit_client_intrinsic_models(
            client_id=k,
            model=client_models[k],
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
        models=client_models,
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

    os.makedirs("results", exist_ok=True)
    out = {
        "alpha": alpha,
        "seed": seed,
        "epochs": EPOCHS,
        "feat_dim": FEAT_DIM,
        "W_OC": W_OC,
        "W_INV": W_INV,
        "W_VAR": W_VAR,
        "SIGMA_MIN": SIGMA_MIN,
        "bb_scores": {str(k): float(v) for k, v in bb_scores.items()},
        "results": {k: float(v) for k, v in results.items()},
        "best_name": best_name,
        "best_acc": float(best_acc),
        "time_train": t_train,
    }
    outpath = f"results/oneclass_ssl_a{alpha}_s{seed}.json"
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
    parser.add_argument("--min_samples", type=int, default=30)
    parser.add_argument("--calib_ratio", type=float, default=0.10)

    parser.add_argument("--w_oc", type=float, default=1.0)
    parser.add_argument("--w_inv", type=float, default=0.5)
    parser.add_argument("--w_var", type=float, default=0.1)
    parser.add_argument("--sigma_min", type=float, default=0.05)

    args = parser.parse_args()

    global EPOCHS, LR, WD, FEAT_DIM, MIN_SAMPLES, CALIB_RATIO
    global W_OC, W_INV, W_VAR, SIGMA_MIN

    EPOCHS = args.epochs
    LR = args.lr
    WD = args.wd
    FEAT_DIM = args.feat_dim
    MIN_SAMPLES = args.min_samples
    CALIB_RATIO = args.calib_ratio

    W_OC = args.w_oc
    W_INV = args.w_inv
    W_VAR = args.w_var
    SIGMA_MIN = args.sigma_min

    run_experiment(args.alpha, args.seed, args.gpu)


if __name__ == "__main__":
    main()