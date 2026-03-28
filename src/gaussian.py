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
# Config
# ============================================================
N_CLIENTS   = 5
N_CLASSES   = 10

BATCH_SIZE  = 256
LR          = 1e-3
WD          = 1e-4
EPOCHS      = 300

FEAT_DIM    = 256
PROJ_DIM    = 256
MIN_SAMPLES = 30

# VICReg-style weights
LAMBDA_INV  = 25.0
MU_VAR      = 25.0
NU_COV      = 1.0

# dual-positive weights
W_INST_Z    = 1.0   # instance pair on projector output
W_CLASS_Z   = 1.0   # class pair on projector output
W_INST_F    = 0.5   # instance pair on backbone feature
W_CLASS_F   = 0.5   # class pair on backbone feature

# Gaussian intrinsic
EPS_VAR     = 1e-4

# checkpoint selection
CKPT_SCORE_W_ACTIVE = 1.0
CKPT_SCORE_W_STD    = 50.0   # feat_std is small, scale it up in score

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)


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


# ============================================================
# Data
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


class ClientDualPositiveDataset(Dataset):
    """
    每个样本返回：
      1) instance-level positive:
         同一张图像的两个增强视图
      2) class-level positive:
         同类不同样本的两个增强视图

    不引入类间分离，不按客户端结构切换训练方式。
    """
    def __init__(self, base_dataset, indices, min_class_samples_for_pair=2):
        self.data = base_dataset.data
        self.targets = np.array(base_dataset.targets)
        self.transform = get_ssl_transform()

        self.indices = list(indices)
        self.length = max(len(self.indices), 1)

        by_class = defaultdict(list)
        for idx in self.indices:
            c = int(self.targets[idx])
            by_class[c].append(idx)

        self.by_class = dict(by_class)
        self.classes_all = sorted(self.by_class.keys())
        self.classes_pairable = sorted([c for c, idxs in self.by_class.items()
                                        if len(idxs) >= min_class_samples_for_pair])

        counts_all = np.array([len(self.by_class[c]) for c in self.classes_all], dtype=np.float64)
        self.class_probs_all = counts_all / counts_all.sum() if len(counts_all) > 0 else None

        counts_pair = np.array([len(self.by_class[c]) for c in self.classes_pairable], dtype=np.float64)
        self.class_probs_pair = counts_pair / counts_pair.sum() if len(counts_pair) > 0 else None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # ---------- instance positive ----------
        idx_inst = random.choice(self.indices)
        img_inst = Image.fromarray(self.data[idx_inst])
        xi1 = self.transform(img_inst)
        xi2 = self.transform(img_inst)
        yi = int(self.targets[idx_inst])

        # ---------- class positive ----------
        if len(self.classes_pairable) > 0:
            c = int(np.random.choice(self.classes_pairable, p=self.class_probs_pair))
            idxs = self.by_class[c]
            j1, j2 = np.random.choice(idxs, size=2, replace=False)
        else:
            # 极端兜底：如果没有可配对的类，就退化成同一样本双视图
            c = int(np.random.choice(self.classes_all, p=self.class_probs_all))
            idxs = self.by_class[c]
            j1 = j2 = random.choice(idxs)

        img_c1 = Image.fromarray(self.data[j1])
        img_c2 = Image.fromarray(self.data[j2])
        xc1 = self.transform(img_c1)
        xc2 = self.transform(img_c2)

        return xi1, xi2, xc1, xc2, yi, c


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
        g1 = 8 if out_ch >= 8 else 1
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(g1, out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(g1, out_ch)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(g1, out_ch)
            )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)), inplace=True)
        out = self.gn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class SmallBackbone(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM):
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


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=FEAT_DIM, proj_dim=PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)


class ClientBackboneModel(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, proj_dim=PROJ_DIM):
        super().__init__()
        self.backbone = SmallBackbone(feat_dim=feat_dim)
        self.projector = ProjectionHead(in_dim=feat_dim, proj_dim=proj_dim)

    def encode(self, x):
        return self.backbone(x)

    def project(self, x):
        f = self.backbone(x)
        z = self.projector(f)
        return f, z


# ============================================================
# Loss
# ============================================================
def vicreg_like_loss(z1, z2, gamma=1.0,
                     lambda_inv=LAMBDA_INV,
                     mu_var=MU_VAR,
                     nu_cov=NU_COV):
    # invariance
    inv_loss = F.mse_loss(z1, z2)

    # variance
    std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    var_loss = torch.mean(F.relu(gamma - std_z1)) + torch.mean(F.relu(gamma - std_z2))

    # covariance
    z1c = z1 - z1.mean(dim=0)
    z2c = z2 - z2.mean(dim=0)
    cov_z1 = (z1c.T @ z1c) / (z1.shape[0] - 1)
    cov_z2 = (z2c.T @ z2c) / (z2.shape[0] - 1)
    cov_loss = off_diagonal(cov_z1).pow(2).mean() + off_diagonal(cov_z2).pow(2).mean()

    loss = lambda_inv * inv_loss + mu_var * var_loss + nu_cov * cov_loss
    parts = {
        "inv": float(inv_loss.item()),
        "var": float(var_loss.item()),
        "cov": float(cov_loss.item()),
    }
    return loss, parts


def dual_positive_loss(fi1, fi2, zi1, zi2,
                       fc1, fc2, zc1, zc2):
    # instance pair loss on projector output
    loss_inst_z, parts_inst_z = vicreg_like_loss(zi1, zi2)
    # class pair loss on projector output
    loss_class_z, parts_class_z = vicreg_like_loss(zc1, zc2)

    # backbone feature also regularized
    loss_inst_f, parts_inst_f = vicreg_like_loss(fi1, fi2)
    loss_class_f, parts_class_f = vicreg_like_loss(fc1, fc2)

    total = (
        W_INST_Z  * loss_inst_z +
        W_CLASS_Z * loss_class_z +
        W_INST_F  * loss_inst_f +
        W_CLASS_F * loss_class_f
    )

    logs = {
        "loss_inst_z": float(loss_inst_z.item()),
        "loss_class_z": float(loss_class_z.item()),
        "loss_inst_f": float(loss_inst_f.item()),
        "loss_class_f": float(loss_class_f.item()),

        "inst_z_inv": parts_inst_z["inv"],
        "inst_z_var": parts_inst_z["var"],
        "inst_z_cov": parts_inst_z["cov"],

        "class_z_inv": parts_class_z["inv"],
        "class_z_var": parts_class_z["var"],
        "class_z_cov": parts_class_z["cov"],

        "inst_f_inv": parts_inst_f["inv"],
        "inst_f_var": parts_inst_f["var"],
        "inst_f_cov": parts_inst_f["cov"],

        "class_f_inv": parts_class_f["inv"],
        "class_f_var": parts_class_f["var"],
        "class_f_cov": parts_class_f["cov"],
    }
    return total, logs


# ============================================================
# Training
# ============================================================
@torch.no_grad()
def probe_feature_quality(model, loader, device, n_batches=1):
    model.eval()
    feat_list = []

    it = iter(loader)
    for _ in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            break

        xi1, _, _, _, _, _ = batch
        xi1 = xi1.to(device, non_blocking=True)
        f = model.encode(xi1)
        feat_list.append(f.detach().cpu())

    if len(feat_list) == 0:
        return 0.0, 0.0

    feats = torch.cat(feat_list, dim=0)
    feat_std = feats.std(dim=0).mean().item()
    active_ratio = (feats.std(dim=0) > 0.01).float().mean().item()
    return feat_std, active_ratio


def train_client_backbone(client_id, base_dataset, client_indices, device,
                          epochs=EPOCHS, lr=LR, wd=WD):
    dataset = ClientDualPositiveDataset(base_dataset, client_indices, min_class_samples_for_pair=2)
    drop_last = len(dataset) >= BATCH_SIZE

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=True
    )

    model = ClientBackboneModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

    best_state = None
    best_score = -1e18
    best_epoch = 0

    print(f"\n  Client {client_id}: backbone training on {len(client_indices)} samples")

    for ep in range(epochs):
        model.train()

        loss_sum = 0.0
        count = 0

        acc_logs = defaultdict(float)

        for xi1, xi2, xc1, xc2, _, _ in loader:
            xi1 = xi1.to(device, non_blocking=True)
            xi2 = xi2.to(device, non_blocking=True)
            xc1 = xc1.to(device, non_blocking=True)
            xc2 = xc2.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                fi1, zi1 = model.project(xi1)
                fi2, zi2 = model.project(xi2)
                fc1, zc1 = model.project(xc1)
                fc2, zc2 = model.project(xc2)

                loss, logs = dual_positive_loss(fi1, fi2, zi1, zi2, fc1, fc2, zc1, zc2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            loss_sum += float(loss.item())
            count += 1
            for k, v in logs.items():
                acc_logs[k] += float(v)

        sch.step()

        avg_loss = loss_sum / max(count, 1)
        avg_logs = {k: v / max(count, 1) for k, v in acc_logs.items()}

        feat_std, active_ratio = probe_feature_quality(model, loader, device, n_batches=1)

        # checkpoint selection: not by training loss, by feature quality
        ckpt_score = CKPT_SCORE_W_ACTIVE * active_ratio + CKPT_SCORE_W_STD * feat_std
        if ckpt_score > best_score:
            best_score = ckpt_score
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (ep + 1) % 50 == 0 or ep == 0:
            print(
                f"    ep {ep+1:3d}/{epochs}  "
                f"loss={avg_loss:.4f}  "
                f"inst_z={avg_logs['loss_inst_z']:.4f}  "
                f"class_z={avg_logs['loss_class_z']:.4f}  "
                f"inst_f={avg_logs['loss_inst_f']:.4f}  "
                f"class_f={avg_logs['loss_class_f']:.4f}  "
                f"feat_std={feat_std:.4f}  active={active_ratio:.1%}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    model = model.cpu()

    meta = {
        "best_epoch": best_epoch,
        "best_score": best_score,
    }
    return model, meta


# ============================================================
# Fit local Gaussian intrinsic models
# ============================================================
@torch.no_grad()
def extract_features(backbone_model, loader, device):
    backbone_model = backbone_model.to(device).eval()
    feats = []
    labels = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        f = backbone_model.encode(x).cpu()
        feats.append(f)
        labels.append(y)

    backbone_model = backbone_model.cpu()
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


@torch.no_grad()
def fit_client_intrinsic_models(client_id, backbone_model, base_dataset, client_indices,
                                client_class_counts, device, min_samples=MIN_SAMPLES):
    ds = IndexedClassDataset(base_dataset, client_indices, transform=get_test_transform())
    loader = DataLoader(
        ds,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    feats, labels = extract_features(backbone_model, loader, device)

    models = {}
    for c in sorted(client_class_counts[client_id].keys()):
        n = client_class_counts[client_id][c]
        if n < min_samples:
            continue

        mask = (labels == c)
        fc = feats[mask]
        if len(fc) < min_samples:
            continue

        mu = fc.mean(dim=0)
        var = fc.var(dim=0, unbiased=False).clamp(min=EPS_VAR)

        energy = ((fc - mu) ** 2 / var).sum(dim=1)
        e_mean = energy.mean()
        e_std = energy.std(unbiased=False).clamp(min=1e-6)

        models[c] = {
            "mu": mu,
            "var": var,
            "e_mean": e_mean,
            "e_std": e_std,
            "n": n,
        }

    print(f"  Client {client_id}: {len(models)} intrinsic Gaussian models")
    for c, pack in models.items():
        print(f"    c{c}: n={pack['n']:5d}")

    return models


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate_intrinsic(backbones, intrinsic_models, client_class_counts, test_loader, device):
    all_labels = None
    n_test = len(test_loader.dataset)

    all_scores = torch.full((N_CLIENTS, n_test, N_CLASSES), float("-inf"))

    for k in range(N_CLIENTS):
        backbone = backbones[k].to(device).eval()

        feats = []
        labels = []
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            f = backbone.encode(x).cpu()
            feats.append(f)
            if k == 0:
                labels.append(y)

        feats = torch.cat(feats, dim=0)
        if k == 0:
            all_labels = torch.cat(labels).numpy()

        for c, pack in intrinsic_models[k].items():
            mu = pack["mu"]
            var = pack["var"]
            e_mean = pack["e_mean"]
            e_std = pack["e_std"]

            energy = ((feats - mu) ** 2 / var).sum(dim=1)
            z = (energy - e_mean) / e_std
            score = -z
            all_scores[k, :, c] = score

        backbones[k] = backbone.cpu()

    results = {}

    # S1: max over clients
    best_per_class, _ = all_scores.max(dim=0)
    preds = best_per_class.argmax(dim=1).numpy()
    results["S1_max_score"] = float((preds == all_labels).mean())

    # S2: weighted by log(n+1)
    weighted = torch.zeros(n_test, N_CLASSES)
    denom = torch.zeros(N_CLASSES)

    for k in range(N_CLIENTS):
        for c in intrinsic_models[k].keys():
            w = math.log(client_class_counts[k][c] + 1.0)
            valid = torch.isfinite(all_scores[k, :, c])
            if valid.any():
                weighted[valid, c] += all_scores[k, valid, c] * w
                denom[c] += w

    for c in range(N_CLASSES):
        if denom[c] > 0:
            weighted[:, c] /= denom[c]

    preds_w = weighted.argmax(dim=1).numpy()
    results["S2_weighted_logN"] = float((preds_w == all_labels).mean())

    # S3: top expert per class
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
            top[:, c] = all_scores[best_k, :, c]

    preds_top = top.argmax(dim=1).numpy()
    results["S3_top_expert"] = float((preds_top == all_labels).mean())

    return results


# ============================================================
# Main
# ============================================================
def run_experiment(alpha, seed, gpu):
    seed_everything(seed)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*72}")
    print(f"  Pipeline Intrinsic-DualPos-Gaussian")
    print(f"  alpha={alpha}  seed={seed}  epochs={EPOCHS}")
    print(f"{'='*72}")

    # data
    train_base = datasets.CIFAR10("./data", train=True, download=True)
    targets = np.array(train_base.targets)

    client_indices, client_class_counts = dirichlet_split(targets, N_CLIENTS, alpha, seed=seed)

    print("\n  Data distribution:")
    for k in range(N_CLIENTS):
        ccc = client_class_counts.get(k, {})
        n_cls = sum(v > 0 for v in ccc.values())
        n_smp = sum(ccc.values())
        top = sorted(ccc.items(), key=lambda x: -x[1])[:5]
        top_str = ", ".join(f"c{c}={n}" for c, n in top)
        print(f"    Client {k}: {n_cls:2d} cls, {n_smp:5d} smp  top: {top_str}")

    test_ds = datasets.CIFAR10("./data", train=False, transform=get_test_transform())
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # --------------------------------------------------------
    # Phase 1: local client backbone training
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Phase 1: Local Backbone Training")
    print(f"{'='*60}")

    backbones = {}
    bb_meta = {}
    t0 = time.time()

    for k in range(N_CLIENTS):
        idxs = client_indices[k]
        if len(idxs) < 2:
            continue
        model, meta = train_client_backbone(
            k, train_base, idxs, device,
            epochs=EPOCHS, lr=LR, wd=WD
        )
        backbones[k] = model
        bb_meta[k] = meta
        torch.cuda.empty_cache()

    t_bb = time.time() - t0
    print(f"\n  Backbone time: {t_bb:.0f}s ({t_bb/60:.1f}min)")

    # --------------------------------------------------------
    # Phase 2: fit class-wise intrinsic Gaussian models
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Phase 2: Local Intrinsic Gaussian Fitting")
    print(f"{'='*60}")

    intrinsic_models = {}
    for k in range(N_CLIENTS):
        intrinsic_models[k] = fit_client_intrinsic_models(
            k, backbones[k], train_base, client_indices[k],
            client_class_counts, device, min_samples=MIN_SAMPLES
        )

    # --------------------------------------------------------
    # Phase 3: evaluation
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Phase 3: Evaluation")
    print(f"{'='*60}")

    results = evaluate_intrinsic(
        backbones, intrinsic_models, client_class_counts,
        test_loader, device
    )

    print("\n  Intrinsic results:")
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
        "weights": {
            "W_INST_Z": W_INST_Z,
            "W_CLASS_Z": W_CLASS_Z,
            "W_INST_F": W_INST_F,
            "W_CLASS_F": W_CLASS_F,
            "LAMBDA_INV": LAMBDA_INV,
            "MU_VAR": MU_VAR,
            "NU_COV": NU_COV,
        },
        "bb_meta": {str(k): v for k, v in bb_meta.items()},
        "results": {k: float(v) for k, v in results.items()},
        "best_name": best_name,
        "best_acc": float(best_acc),
        "time_backbone": t_bb,
    }
    outpath = f"results/pipeline_intrinsic_dualpos_gaussian_a{alpha}_s{seed}.json"
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

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)

    parser.add_argument("--feat_dim", type=int, default=256)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--min_samples", type=int, default=30)

    args = parser.parse_args()

    global EPOCHS, LR, WD, FEAT_DIM, PROJ_DIM, MIN_SAMPLES
    EPOCHS = args.epochs
    LR = args.lr
    WD = args.wd
    FEAT_DIM = args.feat_dim
    PROJ_DIM = args.proj_dim
    MIN_SAMPLES = args.min_samples

    run_experiment(args.alpha, args.seed, args.gpu)


if __name__ == "__main__":
    main()