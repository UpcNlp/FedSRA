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
EPOCHS = 200

FEAT_DIM = 256
MIN_SAMPLES = 30
CALIB_RATIO = 0.10

# LMPRE config
N_PROTOS = 4
TAU = 0.10
LAMBDA_INST = 1.0
LAMBDA_PROTO = 1.0
LAMBDA_REC = 0.5
LAMBDA_SPREAD = 0.1
SPREAD_MARGIN = 0.5

EPS = 1e-8
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
# Datasets
# ============================================================
class ClientClassSSLPairDataset(Dataset):
    """
    Return two augmentations of the same sample and its class label.
    Used for client-local class-conditional manifold learning.
    """
    def __init__(self, base_dataset, indices):
        self.data = base_dataset.data
        self.targets = np.array(base_dataset.targets)
        self.indices = list(indices)
        self.transform = get_ssl_transform()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ridx = self.indices[idx]
        img = Image.fromarray(self.data[ridx])
        y = int(self.targets[ridx])
        x1 = self.transform(img)
        x2 = self.transform(img)
        return x1, x2, y


class IndexedClassDataset(Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.data = base_dataset.data
        self.targets = np.array(base_dataset.targets)
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ridx = self.indices[idx]
        img = Image.fromarray(self.data[ridx])
        y = int(self.targets[ridx])
        return self.transform(img), y


# ============================================================
# Model
# ============================================================
class GNBasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        gn_groups1 = 8 if out_ch >= 8 else 1
        gn_groups2 = 8 if out_ch >= 8 else 1

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(gn_groups1, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(gn_groups2, out_ch)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(gn_groups1, out_ch)
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
        self.layer1 = nn.Sequential(
            GNBasicBlock(64, 64, 1),
            GNBasicBlock(64, 64, 1),
        )
        self.layer2 = nn.Sequential(
            GNBasicBlock(64, 128, 2),
            GNBasicBlock(128, 128, 1),
        )
        self.layer3 = nn.Sequential(
            GNBasicBlock(128, 256, 2),
            GNBasicBlock(256, 256, 1),
        )
        self.layer4 = nn.Sequential(
            GNBasicBlock(256, 512, 2),
            GNBasicBlock(512, 512, 1),
        )
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
        x = F.normalize(x, dim=1)
        return x


class ClassExpert(nn.Module):
    def __init__(self, feat_dim=256, n_proto=4):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(n_proto, feat_dim))
        self.recon = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, z, tau=0.1):
        p = F.normalize(self.prototypes, dim=1)
        sim = z @ p.t()
        w = F.softmax(sim / tau, dim=1)
        z_bar = w @ p
        z_tilde = self.recon(z_bar)
        z_tilde = F.normalize(z_tilde, dim=1)
        return z_bar, z_tilde, w, p, sim


class ClientLMPREModel(nn.Module):
    def __init__(self, feat_dim=256, n_classes=10, n_proto=4):
        super().__init__()
        self.backbone = SmallBackbone(feat_dim=feat_dim)
        self.experts = nn.ModuleList([
            ClassExpert(feat_dim=feat_dim, n_proto=n_proto) for _ in range(n_classes)
        ])

    def encode(self, x):
        return self.backbone(x)

    def expert_forward(self, z, cls_id, tau=0.1):
        return self.experts[cls_id](z, tau=tau)


# ============================================================
# Loss
# ============================================================
def spread_loss(prototypes, margin=0.5):
    p = F.normalize(prototypes, dim=1)
    sim = p @ p.t()
    eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    vals = sim[~eye]
    if vals.numel() == 0:
        return torch.tensor(0.0, device=sim.device)
    return F.relu(vals - margin).mean()


# ============================================================
# Train Client LMPRE
# ============================================================
def train_client_lmpre(client_id, base_dataset, client_indices, device,
                       epochs=EPOCHS, lr=LR, wd=WD,
                       feat_dim=FEAT_DIM, n_proto=N_PROTOS):
    dataset = ClientClassSSLPairDataset(base_dataset, client_indices)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=len(dataset) >= BATCH_SIZE,
        persistent_workers=True,
    )

    model = ClientLMPREModel(feat_dim=feat_dim, n_classes=N_CLASSES, n_proto=n_proto).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

    best_state = None
    best_quality = -1e9

    print(f"\n  Client {client_id}: LMPRE training on {len(client_indices)} samples")

    for ep in range(epochs):
        model.train()
        loss_sum = 0.0
        n_batch = 0
        stat_sum = defaultdict(float)

        for x1, x2, y in loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                z1 = model.encode(x1)
                z2 = model.encode(x2)

                loss_inst = F.mse_loss(z1, z2)

                loss_proto = torch.tensor(0.0, device=device)
                loss_rec = torch.tensor(0.0, device=device)
                loss_spread = torch.tensor(0.0, device=device)
                n_groups = 0

                unique_classes = y.unique(sorted=True)
                for cls_id in unique_classes.tolist():
                    mask = (y == cls_id)
                    if int(mask.sum().item()) == 0:
                        continue

                    z1c = z1[mask]
                    z2c = z2[mask]

                    zbar1, ztilde1, _, p, _ = model.expert_forward(z1c, cls_id, tau=TAU)
                    zbar2, ztilde2, _, _, _ = model.expert_forward(z2c, cls_id, tau=TAU)

                    loss_proto = loss_proto + F.mse_loss(z1c, zbar1) + F.mse_loss(z2c, zbar2)
                    loss_rec = loss_rec + F.mse_loss(z1c, ztilde1) + F.mse_loss(z2c, ztilde2)
                    loss_spread = loss_spread + spread_loss(p, margin=SPREAD_MARGIN)
                    n_groups += 1

                if n_groups > 0:
                    loss_proto = loss_proto / n_groups
                    loss_rec = loss_rec / n_groups
                    loss_spread = loss_spread / n_groups

                loss = (
                    LAMBDA_INST * loss_inst
                    + LAMBDA_PROTO * loss_proto
                    + LAMBDA_REC * loss_rec
                    + LAMBDA_SPREAD * loss_spread
                )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            loss_sum += float(loss.item())
            stat_sum["inst"] += float(loss_inst.item())
            stat_sum["proto"] += float(loss_proto.item())
            stat_sum["rec"] += float(loss_rec.item())
            stat_sum["spread"] += float(loss_spread.item())
            n_batch += 1

        sch.step()

        avg_loss = loss_sum / max(n_batch, 1)

        model.eval()
        with torch.no_grad():
            batch = next(iter(loader))
            x_probe = batch[0][:min(128, len(batch[0]))].to(device)
            feats = model.encode(x_probe)
            feat_std = feats.std(dim=0).mean().item()
            active_ratio = (feats.std(dim=0) > 0.01).float().mean().item()
            quality_score = active_ratio + 10.0 * feat_std - 0.01 * avg_loss

        if quality_score > best_quality:
            best_quality = quality_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (ep + 1) % 50 == 0 or ep == 0:
            def avg(name):
                return stat_sum[name] / max(n_batch, 1)
            print(
                f"    ep {ep+1:3d}/{epochs}  "
                f"loss={avg_loss:.4f}  "
                f"inst={avg('inst'):.4f}  "
                f"proto={avg('proto'):.4f}  "
                f"rec={avg('rec'):.4f}  "
                f"spread={avg('spread'):.4f}  "
                f"feat_std={feat_std:.4f}  active={active_ratio:.1%}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    model = model.cpu()
    return model, best_quality


# ============================================================
# Feature Extraction
# ============================================================
@torch.no_grad()
def extract_features(model, loader, device):
    model = model.to(device).eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        z = model.encode(x).cpu()
        feats.append(z)
        labels.append(y)
    model = model.cpu()
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


# ============================================================
# Fit LMPRE Intrinsic Models + Calibration
# ============================================================
@torch.no_grad()
def fit_client_lmpre_models(client_id, model, base_dataset,
                            fit_indices, calib_indices,
                            client_class_counts_fit, device,
                            min_samples=MIN_SAMPLES):
    fit_ds = IndexedClassDataset(base_dataset, fit_indices, transform=get_test_transform())
    fit_loader = DataLoader(fit_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    fit_feats, fit_labels = extract_features(model, fit_loader, device)

    calib_feats = None
    calib_labels = None
    if len(calib_indices) > 0:
        calib_ds = IndexedClassDataset(base_dataset, calib_indices, transform=get_test_transform())
        calib_loader = DataLoader(calib_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
        calib_feats, calib_labels = extract_features(model, calib_loader, device)

    model = model.to(device).eval()
    intrinsic_models = {}

    for c in sorted(client_class_counts_fit[client_id].keys()):
        n = client_class_counts_fit[client_id][c]
        if n < min_samples:
            continue

        mask_fit = (fit_labels == c)
        fc = fit_feats[mask_fit]
        if len(fc) < min_samples:
            continue

        expert = model.experts[c]
        p = F.normalize(expert.prototypes.detach().cpu(), dim=1)

        # fit-set score statistics
        z = fc.to(device)
        zbar, ztilde, _, _, sim = expert(z, tau=TAU)
        s_cos = sim.max(dim=1).values
        s_proj = -((z - zbar) ** 2).sum(dim=1)
        s_rec = -((z - ztilde) ** 2).sum(dim=1)
        score = (1.0 * s_cos + 1.0 * s_proj + 0.5 * s_rec).detach().cpu()

        fit_mean = score.mean()
        fit_std = score.std(unbiased=False).clamp(min=1e-6)

        pack = {
            "prototypes": p,
            "fit_mean": fit_mean,
            "fit_std": fit_std,
            "n": int(n),
        }

        if calib_feats is not None and len(calib_feats) > 0:
            zc = calib_feats.to(device)
            zbar_c, ztilde_c, _, _, sim_c = expert(zc, tau=TAU)
            sc_cos = sim_c.max(dim=1).values
            sc_proj = -((zc - zbar_c) ** 2).sum(dim=1)
            sc_rec = -((zc - ztilde_c) ** 2).sum(dim=1)
            sc = (1.0 * sc_cos + 1.0 * sc_proj + 0.5 * sc_rec).detach().cpu()

            pos_mask = (calib_labels == c)
            neg_mask = (calib_labels != c)

            n_pos = int(pos_mask.sum().item())
            n_neg = int(neg_mask.sum().item())

            if n_pos >= 3:
                pos_s = sc[pos_mask]
                pos_mean = pos_s.mean()
                pos_std = pos_s.std(unbiased=False).clamp(min=1e-6)
            else:
                pos_mean = fit_mean
                pos_std = fit_std

            if n_neg >= 5:
                neg_s = sc[neg_mask]
                neg_mean = neg_s.mean()
                neg_std = neg_s.std(unbiased=False).clamp(min=1e-6)
            else:
                neg_mean = pos_mean - 2.5 * pos_std
                neg_std = pos_std * 1.5 + 1e-3

            threshold = 0.5 * (pos_mean + neg_mean)
            scale = torch.sqrt(0.5 * (pos_std ** 2 + neg_std ** 2)).clamp(min=0.2)

            sep = ((pos_mean - neg_mean) / (pos_std + neg_std + 1e-6)).item()
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
                "calib_pos_mean": fit_mean,
                "calib_pos_std": fit_std,
                "calib_neg_mean": fit_mean - 2.5 * fit_std,
                "calib_neg_std": fit_std * 1.5 + 1e-3,
                "threshold": fit_mean - 1.25 * fit_std,
                "scale": max(float(fit_std.item()), 0.2),
                "reliability": 0.5 * math.log(n + 1.0),
                "n_pos_calib": 0,
                "n_neg_calib": 0,
            })

        intrinsic_models[c] = pack

    model = model.cpu()
    print(f"  Client {client_id}: {len(intrinsic_models)} LMPRE class models")
    for c, pack in intrinsic_models.items():
        print(
            f"    c{c}: n={pack['n']:5d}, rel={pack['reliability']:.3f}, "
            f"pos_calib={pack['n_pos_calib']}, neg_calib={pack['n_neg_calib']}"
        )

    return intrinsic_models


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate_lmpre(models, intrinsic_models, client_class_counts_fit, test_loader, device):
    test_labels = None
    n_test = len(test_loader.dataset)

    raw_scores = torch.full((N_CLIENTS, n_test, N_CLASSES), float("-inf"))
    calib_scores = torch.full((N_CLIENTS, n_test, N_CLASSES), float("-inf"))
    calib_probs = torch.zeros(N_CLIENTS, n_test, N_CLASSES)
    reliabilities = torch.zeros(N_CLIENTS, N_CLASSES)

    for k in range(N_CLIENTS):
        model = models[k].to(device).eval()

        feats = []
        labels = []
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            z = model.encode(x)
            feats.append(z.cpu())
            if test_labels is None:
                labels.append(y)
        feats = torch.cat(feats, dim=0)

        if test_labels is None:
            test_labels = torch.cat(labels).numpy()

        for c, pack in intrinsic_models[k].items():
            expert = model.experts[c]
            z = feats.to(device)
            zbar, ztilde, _, _, sim = expert(z, tau=TAU)

            s_cos = sim.max(dim=1).values
            s_proj = -((z - zbar) ** 2).sum(dim=1)
            s_rec = -((z - ztilde) ** 2).sum(dim=1)
            score = 1.0 * s_cos + 1.0 * s_proj + 0.5 * s_rec
            score = score.cpu()

            raw_z = (score - pack["fit_mean"]) / pack["fit_std"]
            raw_scores[k, :, c] = raw_z

            threshold = pack["threshold"]
            scale = pack["scale"]
            margin = (score - threshold) / scale
            calib_scores[k, :, c] = margin
            calib_probs[k, :, c] = torch.sigmoid(margin)

            reliabilities[k, c] = float(pack["reliability"])

        models[k] = model.cpu()

    results = {}

    # S1: raw max-score
    best_per_class, _ = raw_scores.max(dim=0)
    preds_s1 = best_per_class.argmax(dim=1).numpy()
    results["S1_max_score"] = float((preds_s1 == test_labels).mean())

    # S2: weighted logN on raw z-score
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
        best_k = -1
        best_n = -1
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

    # S4: calibrated reliability fusion on probs
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
        else:
            fused_prob[:, c] = 0.0

    preds_s4 = fused_prob.argmax(dim=1).numpy()
    results["S4_calib_rel_prob"] = float((preds_s4 == test_labels).mean())

    # S5: calibrated reliability fusion on margin
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
    print(f"  Pipeline LMPRE")
    print(f"  alpha={alpha}  seed={seed}  epochs={EPOCHS}  calib_ratio={CALIB_RATIO}")
    print(f"  feat_dim={FEAT_DIM}  n_proto={N_PROTOS}")
    print(f"{'='*72}")

    train_base = datasets.CIFAR10("./data", train=True, download=True)
    targets = np.array(train_base.targets)

    client_indices_all, client_class_counts_all = dirichlet_split(targets, N_CLIENTS, alpha, seed=seed)

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
        print(
            f"    Client {k}: {n_cls:2d} cls, {n_smp:5d} smp  "
            f"train={len(tr_idx):5d} calib={len(ca_idx):4d}  top: {top_str}"
        )

    test_ds = datasets.CIFAR10("./data", train=False, transform=get_test_transform())
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    print(f"\n{'='*60}")
    print("  Phase 1: Local LMPRE Training")
    print(f"{'='*60}")

    models = {}
    bb_scores = {}
    t0 = time.time()

    for k in range(N_CLIENTS):
        idxs = client_train_idx[k]
        if len(idxs) < 2:
            continue
        model, best_quality = train_client_lmpre(
            k, train_base, idxs, device,
            epochs=EPOCHS, lr=LR, wd=WD,
            feat_dim=FEAT_DIM, n_proto=N_PROTOS,
        )
        models[k] = model
        bb_scores[k] = best_quality
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    t_bb = time.time() - t0
    print(f"\n  Backbone time: {t_bb:.0f}s ({t_bb/60:.1f}min)")

    print(f"\n{'='*60}")
    print("  Phase 2: Fit + Calibration")
    print(f"{'='*60}")

    intrinsic_models = {}
    for k in range(N_CLIENTS):
        intrinsic_models[k] = fit_client_lmpre_models(
            client_id=k,
            model=models[k],
            base_dataset=train_base,
            fit_indices=client_train_idx[k],
            calib_indices=client_calib_idx[k],
            client_class_counts_fit=client_class_counts_fit,
            device=device,
            min_samples=MIN_SAMPLES,
        )

    print(f"\n{'='*60}")
    print("  Phase 3: Evaluation")
    print(f"{'='*60}")

    results = evaluate_lmpre(
        models=models,
        intrinsic_models=intrinsic_models,
        client_class_counts_fit=client_class_counts_fit,
        test_loader=test_loader,
        device=device,
    )

    print("\n  LMPRE intrinsic results:")
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
        "n_proto": N_PROTOS,
        "min_samples": MIN_SAMPLES,
        "calib_ratio": CALIB_RATIO,
        "bb_scores": {str(k): float(v) for k, v in bb_scores.items()},
        "results": {k: float(v) for k, v in results.items()},
        "best_name": best_name,
        "best_acc": float(best_acc),
        "time_backbone": t_bb,
    }
    outpath = f"results/pipeline_lmpre_a{alpha}_s{seed}.json"
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*72}")
    print(f"  ★ BEST: {best_name} = {best_acc:.2%}")
    print(f"  Saved: {outpath}")
    print(f"{'='*72}\n")

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)

    parser.add_argument("--feat_dim", type=int, default=256)
    parser.add_argument("--min_samples", type=int, default=30)
    parser.add_argument("--calib_ratio", type=float, default=0.10)

    parser.add_argument("--n_proto", type=int, default=4)
    parser.add_argument("--tau", type=float, default=0.10)
    parser.add_argument("--lambda_inst", type=float, default=1.0)
    parser.add_argument("--lambda_proto", type=float, default=1.0)
    parser.add_argument("--lambda_rec", type=float, default=0.5)
    parser.add_argument("--lambda_spread", type=float, default=0.1)
    parser.add_argument("--spread_margin", type=float, default=0.5)

    args = parser.parse_args()

    global EPOCHS, LR, WD, FEAT_DIM, MIN_SAMPLES, CALIB_RATIO
    global N_PROTOS, TAU, LAMBDA_INST, LAMBDA_PROTO, LAMBDA_REC, LAMBDA_SPREAD, SPREAD_MARGIN

    EPOCHS = args.epochs
    LR = args.lr
    WD = args.wd
    FEAT_DIM = args.feat_dim
    MIN_SAMPLES = args.min_samples
    CALIB_RATIO = args.calib_ratio

    N_PROTOS = args.n_proto
    TAU = args.tau
    LAMBDA_INST = args.lambda_inst
    LAMBDA_PROTO = args.lambda_proto
    LAMBDA_REC = args.lambda_rec
    LAMBDA_SPREAD = args.lambda_spread
    SPREAD_MARGIN = args.spread_margin

    run_experiment(args.alpha, args.seed, args.gpu)


if __name__ == "__main__":
    main()
