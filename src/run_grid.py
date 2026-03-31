"""
run_grid.py
===========
对比实验：Relational (CE+logit ensemble) vs Intrinsic (SSL+Gaussian)
变量：alpha x K 的二维网格

用法：
    python run_grid.py --alpha 0.05 --n_clients 5 --gpu 0

结果保存到 results/grid_a{alpha}_k{n_clients}_s{seed}.json
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
# Config
# ============================================================
N_CLASSES   = 10
BATCH_SIZE  = 256
EPOCHS_CE   = 100   # CE pipeline epochs（比 SSL 少，因为 CE 收敛快）
EPOCHS_SSL  = 200   # SSL pipeline epochs
LR          = 1e-3
WD          = 1e-4
FEAT_DIM    = 256
PROJ_DIM    = 256
MIN_SAMPLES = 20
CALIB_RATIO = 0.10

# VICReg
LAMBDA_INV  = 25.0
MU_VAR      = 25.0
NU_COV      = 1.0
# Dual-positive
LAMBDA_INST = 1.0
LAMBDA_CLS  = 1.0
LAMBDA_FEAT = 0.5

EPS_VAR     = 1e-4
EPS_STD     = 1e-6
MAX_REL     = 6.0

CIFAR_MEAN  = (0.4914, 0.4822, 0.4465)
CIFAR_STD   = (0.2470, 0.2435, 0.2616)


# ============================================================
# Reproducibility
# ============================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Data partition
# ============================================================
def dirichlet_split(targets, n_clients, alpha, seed=42):
    rng = np.random.RandomState(seed)
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[int(label)].append(idx)

    client_indices      = defaultdict(list)
    client_class_counts = defaultdict(lambda: defaultdict(int))

    for c in range(N_CLASSES):
        idxs = np.array(class_indices[c])
        rng.shuffle(idxs)
        props  = rng.dirichlet([alpha] * n_clients)
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


def split_train_calib(base_targets, indices, calib_ratio=0.1, seed=42):
    rng     = np.random.RandomState(seed)
    by_cls  = defaultdict(list)
    for idx in indices:
        by_cls[int(base_targets[idx])].append(idx)

    train_idx, calib_idx = [], []
    for c, idxs in by_cls.items():
        idxs = idxs.copy(); rng.shuffle(idxs)
        if len(idxs) <= 2:
            train_idx.extend(idxs); continue
        n_c = max(1, int(round(len(idxs) * calib_ratio)))
        n_c = min(n_c, len(idxs) - 1)
        calib_idx.extend(idxs[:n_c])
        train_idx.extend(idxs[n_c:])

    rng.shuffle(train_idx); rng.shuffle(calib_idx)
    return train_idx, calib_idx


# ============================================================
# Transforms
# ============================================================
def ssl_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


# ============================================================
# Datasets
# ============================================================
class IndexedDataset(Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.data    = base_dataset.data
        self.targets = np.array(base_dataset.targets)
        self.indices = list(indices)
        self.tf      = transform

    def __len__(self):  return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img = Image.fromarray(self.data[idx])
        return self.tf(img), int(self.targets[idx])


class DualPositiveDataset(Dataset):
    """SSL 训练集：instance positive + class positive"""
    def __init__(self, base_dataset, indices):
        self.data    = base_dataset.data
        self.targets = np.array(base_dataset.targets)
        self.indices = list(indices)
        self.tf      = ssl_transform()
        self.length  = max(len(self.indices), 1)

        by_cls = defaultdict(list)
        for idx in self.indices:
            by_cls[int(self.targets[idx])].append(idx)
        self.by_cls   = {c: v for c, v in by_cls.items() if len(v) >= 1}
        self.classes  = sorted(self.by_cls.keys())
        counts        = np.array([len(self.by_cls[c]) for c in self.classes], dtype=np.float64)
        self.cls_prob = counts / counts.sum() if len(counts) > 0 else None

    def __len__(self):  return self.length

    def __getitem__(self, i):
        idx_a   = self.indices[i % len(self.indices)]
        img_a   = Image.fromarray(self.data[idx_a])
        xi1, xi2 = self.tf(img_a), self.tf(img_a)

        c       = int(np.random.choice(self.classes, p=self.cls_prob))
        idxs    = self.by_cls[c]
        i1, i2  = (np.random.choice(idxs, 2, replace=False) if len(idxs) >= 2
                   else (idxs[0], idxs[0]))
        xc1 = self.tf(Image.fromarray(self.data[i1]))
        xc2 = self.tf(Image.fromarray(self.data[i2]))
        return xi1, xi2, xc1, xc2


# ============================================================
# Models
# ============================================================
class GNBlock(nn.Module):
    def __init__(self, ic, oc, stride=1):
        super().__init__()
        g = 8 if oc >= 8 else 1
        self.c1 = nn.Conv2d(ic, oc, 3, stride=stride, padding=1, bias=False)
        self.n1 = nn.GroupNorm(g, oc)
        self.c2 = nn.Conv2d(oc, oc, 3, padding=1, bias=False)
        self.n2 = nn.GroupNorm(g, oc)
        self.skip = (nn.Sequential(nn.Conv2d(ic, oc, 1, stride=stride, bias=False),
                                   nn.GroupNorm(g, oc))
                     if stride != 1 or ic != oc else nn.Identity())

    def forward(self, x):
        out = F.relu(self.n1(self.c1(x)), True)
        out = self.n2(self.c2(out))
        return F.relu(out + self.skip(x), True)


class Backbone(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        self.stem   = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1, bias=False),
                                    nn.GroupNorm(8, 64), nn.ReLU(True))
        self.layer1 = nn.Sequential(GNBlock(64,  64),  GNBlock(64,  64))
        self.layer2 = nn.Sequential(GNBlock(64,  128, 2), GNBlock(128, 128))
        self.layer3 = nn.Sequential(GNBlock(128, 256, 2), GNBlock(256, 256))
        self.layer4 = nn.Sequential(GNBlock(256, 512, 2), GNBlock(512, 512))
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(512, feat_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return F.normalize(self.fc(x), dim=1)


class ProjHead(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.BatchNorm1d(d),
                                 nn.ReLU(True), nn.Linear(d, d))
    def forward(self, x): return self.net(x)


class SSLModel(nn.Module):
    def __init__(self, feat_dim=256, proj_dim=256):
        super().__init__()
        self.backbone  = Backbone(feat_dim)
        self.projector = ProjHead(feat_dim)

    def encode(self, x):   return self.backbone(x)
    def project(self, x):
        f = self.backbone(x)
        return f, self.projector(f)


class CEModel(nn.Module):
    """CE pipeline: backbone + linear classifier"""
    def __init__(self, n_classes=10, feat_dim=256):
        super().__init__()
        self.backbone = Backbone(feat_dim)
        self.head     = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        return self.head(self.backbone(x))

    def encode(self, x):   return self.backbone(x)


# ============================================================
# Losses
# ============================================================
def off_diagonal(x):
    n = x.shape[0]
    return x.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()


def vicreg(a, b):
    inv = F.mse_loss(a, b)
    sa  = torch.sqrt(a.var(0) + 1e-4); sb = torch.sqrt(b.var(0) + 1e-4)
    var = F.relu(1 - sa).mean() + F.relu(1 - sb).mean()
    ac  = a - a.mean(0); bc = b - b.mean(0)
    ca  = ac.T @ ac / max(a.shape[0]-1, 1)
    cb  = bc.T @ bc / max(b.shape[0]-1, 1)
    cov = off_diagonal(ca).pow(2).mean() + off_diagonal(cb).pow(2).mean()
    return LAMBDA_INV * inv + MU_VAR * var + NU_COV * cov


def dual_positive_loss(fi1, fi2, fc1, fc2, zi1, zi2, zc1, zc2):
    return (LAMBDA_INST * vicreg(zi1, zi2)
          + LAMBDA_CLS  * vicreg(zc1, zc2)
          + LAMBDA_FEAT * (LAMBDA_INST * vicreg(fi1, fi2)
                         + LAMBDA_CLS  * vicreg(fc1, fc2)))


# ============================================================
# Training helpers
# ============================================================
def use_bf16():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def train_ce_client(client_id, base_dataset, indices, device, epochs=EPOCHS_CE):
    """关系性知识 pipeline：CE 训练"""
    ds     = IndexedDataset(base_dataset, indices, ssl_transform())   # augment during training
    loader = DataLoader(ds, batch_size=min(BATCH_SIZE, len(ds)),
                        shuffle=True, num_workers=4, pin_memory=True,
                        drop_last=len(ds) >= BATCH_SIZE,
                        persistent_workers=len(ds) >= BATCH_SIZE)

    model = CEModel(N_CLASSES, FEAT_DIM).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    print(f"  [CE] Client {client_id}: {len(indices)} samples, {epochs} epochs")
    for ep in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16()):
                loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sch.step()

        if (ep + 1) % 50 == 0:
            print(f"    ep {ep+1}/{epochs}  loss={loss.item():.4f}")

    return model.cpu().eval()


def train_ssl_client(client_id, base_dataset, indices, device, epochs=EPOCHS_SSL):
    """本体性知识 pipeline：SSL dual-positive 训练"""
    ds     = DualPositiveDataset(base_dataset, indices)
    loader = DataLoader(ds, batch_size=min(BATCH_SIZE, len(ds)),
                        shuffle=True, num_workers=4, pin_memory=True,
                        drop_last=len(ds) >= BATCH_SIZE,
                        persistent_workers=len(ds) >= BATCH_SIZE)

    model = SSLModel(FEAT_DIM, PROJ_DIM).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_state, best_q = None, -1e9
    print(f"  [SSL] Client {client_id}: {len(indices)} samples, {epochs} epochs")

    for ep in range(epochs):
        model.train()
        loss_sum, n = 0.0, 0
        for xi1, xi2, xc1, xc2 in loader:
            xi1,xi2,xc1,xc2 = [t.to(device, non_blocking=True) for t in [xi1,xi2,xc1,xc2]]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16()):
                fi1,zi1 = model.project(xi1); fi2,zi2 = model.project(xi2)
                fc1,zc1 = model.project(xc1); fc2,zc2 = model.project(xc2)
                loss = dual_positive_loss(fi1,fi2,fc1,fc2,zi1,zi2,zc1,zc2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_sum += loss.item(); n += 1
        sch.step()

        # quality probe
        model.eval()
        with torch.no_grad():
            batch = next(iter(loader))
            xp    = batch[0][:128].to(device)
            f     = model.encode(xp)
            fstd  = f.std(0).mean().item()
            act   = (f.std(0) > 0.01).float().mean().item()
            q     = act + 10 * fstd - 0.01 * (loss_sum / max(n,1))
        if q > best_q:
            best_q     = q
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (ep + 1) % 50 == 0:
            print(f"    ep {ep+1}/{epochs}  loss={loss_sum/max(n,1):.4f}  fstd={fstd:.4f}  active={act:.1%}")

    if best_state: model.load_state_dict(best_state)
    return model.cpu().eval()


# ============================================================
# Inference
# ============================================================
@torch.no_grad()
def extract_feats(model, loader, device, is_ce=False):
    model = model.to(device).eval()
    feats, labels, logits_list = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        f = model.encode(x).cpu()
        feats.append(f); labels.append(y)
        if is_ce:
            logits_list.append(model(x).cpu())
    model = model.cpu()
    feats  = torch.cat(feats)
    labels = torch.cat(labels)
    logits = torch.cat(logits_list) if is_ce else None
    return feats, labels, logits


@torch.no_grad()
def relational_inference(ce_models, test_loader, device, n_clients):
    """CE logit ensemble"""
    all_logits  = []
    test_labels = None

    for k in range(n_clients):
        feats, labels, logits = extract_feats(ce_models[k], test_loader, device, is_ce=True)
        all_logits.append(logits)
        if test_labels is None:
            test_labels = labels.numpy()

    # simple average ensemble
    ensemble = torch.stack(all_logits, 0).mean(0)
    preds    = ensemble.argmax(1).numpy()
    acc      = float((preds == test_labels).mean())
    return acc, test_labels


@torch.no_grad()
def intrinsic_inference(ssl_models, train_datasets, test_loader, device,
                        n_clients, client_train_idx, client_calib_idx,
                        client_class_counts):
    """SSL + per-class Gaussian expert"""
    n_test      = len(test_loader.dataset)
    raw_scores  = torch.full((n_clients, n_test, N_CLASSES), float("-inf"))
    test_labels = None

    expert_models = {}  # k -> {c -> pack}

    for k in range(n_clients):
        model   = ssl_models[k]
        fit_ds  = IndexedDataset(train_datasets, client_train_idx[k], test_transform())
        fit_ldr = DataLoader(fit_ds, 256, shuffle=False, num_workers=4)
        ff, fl, _ = extract_feats(model, fit_ldr, device)

        cal_ds  = IndexedDataset(train_datasets, client_calib_idx[k], test_transform())
        cal_ldr = DataLoader(cal_ds, 256, shuffle=False, num_workers=4)
        cf, cl, _ = extract_feats(model, cal_ldr, device)

        experts = {}
        for c, n in client_class_counts[k].items():
            if n < MIN_SAMPLES: continue
            fc   = ff[fl == c]
            if len(fc) < MIN_SAMPLES: continue
            mu   = fc.mean(0)
            var  = fc.var(0, unbiased=False).clamp(min=EPS_VAR)
            e_fit = ((fc - mu)**2 / var).sum(1)
            experts[c] = {"mu": mu, "var": var,
                          "e_mean": e_fit.mean(), "e_std": e_fit.std(unbiased=False).clamp(EPS_STD),
                          "n": n}
        expert_models[k] = experts

        # test set scores
        model = model.to(device).eval()
        tfeats, tlbls = [], []
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            tfeats.append(model.encode(x).cpu())
            if test_labels is None: tlbls.append(y)
        model = model.cpu()
        tfeats = torch.cat(tfeats)
        if test_labels is None: test_labels = torch.cat(tlbls).numpy()

        for c, pack in experts.items():
            energy   = ((tfeats - pack["mu"])**2 / pack["var"]).sum(1)
            raw_z    = (energy - pack["e_mean"]) / pack["e_std"]
            raw_scores[k, :, c] = -raw_z

    # weighted logN aggregation
    weighted = torch.zeros(n_test, N_CLASSES)
    denom    = torch.zeros(N_CLASSES)
    for k in range(n_clients):
        for c, pack in expert_models[k].items():
            w    = math.log(pack["n"] + 1.0)
            valid = torch.isfinite(raw_scores[k, :, c])
            if valid.any():
                weighted[valid, c] += raw_scores[k, valid, c] * w
                denom[c]           += w
    for c in range(N_CLASSES):
        if denom[c] > 0: weighted[:, c] /= denom[c]
        else: weighted[:, c] = float("-inf")

    preds = weighted.argmax(1).numpy()
    acc   = float((preds == test_labels).mean())
    return acc


# ============================================================
# Main experiment
# ============================================================
def run(alpha, n_clients, seed, gpu):
    seed_everything(seed)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"  alpha={alpha}  K={n_clients}  seed={seed}")
    print(f"{'='*70}")

    train_base = datasets.CIFAR10("./data", train=True, download=True)
    targets    = np.array(train_base.targets)
    test_ds    = datasets.CIFAR10("./data", train=False, transform=test_transform())
    test_ldr   = DataLoader(test_ds, 256, shuffle=False, num_workers=4, pin_memory=True)

    # partition
    client_indices, client_counts = dirichlet_split(targets, n_clients, alpha, seed)

    client_train, client_calib, client_counts_fit = {}, {}, defaultdict(lambda: defaultdict(int))
    for k in range(n_clients):
        idxs = client_indices.get(k, [])   # client may have received no data
        if len(idxs) == 0:
            client_train[k] = []
            client_calib[k] = []
            continue
        tr, ca = split_train_calib(targets, idxs, CALIB_RATIO, seed+1000+k)
        client_train[k] = tr; client_calib[k] = ca
        for idx in tr: client_counts_fit[k][int(targets[idx])] += 1

    print("\n  Data distribution:")
    for k in range(n_clients):
        cc  = client_counts.get(k, {})
        nc  = sum(v > 0 for v in cc.values())
        ns  = sum(cc.values())
        top = sorted(cc.items(), key=lambda x: -x[1])[:3]
        print(f"    Client {k}: {nc} cls, {ns} samples  top: {', '.join(f'c{c}={n}' for c,n in top)}")

    t0 = time.time()

    # ── Relational pipeline (CE) ──────────────────────────────
    print(f"\n── Relational Pipeline (CE) ──")
    ce_models = {}
    for k in range(n_clients):
        if len(client_train[k]) < 2: continue
        ce_models[k] = train_ce_client(k, train_base, client_train[k], device, EPOCHS_CE)
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    acc_rel, test_labels = relational_inference(ce_models, test_ldr, device, n_clients)
    print(f"\n  Relational accuracy: {acc_rel:.4f} ({acc_rel:.2%})")

    t_ce = time.time() - t0

    # ── Intrinsic pipeline (SSL) ──────────────────────────────
    print(f"\n── Intrinsic Pipeline (SSL+Gaussian) ──")
    ssl_models = {}
    for k in range(n_clients):
        if len(client_train[k]) < 2: continue
        ssl_models[k] = train_ssl_client(k, train_base, client_train[k], device, EPOCHS_SSL)
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    acc_int = intrinsic_inference(
        ssl_models, train_base, test_ldr, device,
        n_clients, client_train, client_calib, client_counts_fit
    )
    print(f"\n  Intrinsic accuracy: {acc_int:.4f} ({acc_int:.2%})")

    t_total = time.time() - t0

    # ── Save ─────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    out = {
        "alpha":       alpha,
        "n_clients":   n_clients,
        "seed":        seed,
        "acc_relational": acc_rel,
        "acc_intrinsic":  acc_int,
        "gap":            acc_rel - acc_int,
        "time_total":     t_total,
        "epochs_ce":   EPOCHS_CE,
        "epochs_ssl":  EPOCHS_SSL,
    }
    path = f"results/grid_a{alpha}_k{n_clients}_s{seed}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  Relational: {acc_rel:.2%}   Intrinsic: {acc_int:.2%}   Gap: {acc_rel-acc_int:+.2%}")
    print(f"  Saved: {path}  ({t_total/60:.1f} min)")
    print(f"{'='*70}\n")
    return out


def main():
    global EPOCHS_CE, EPOCHS_SSL

    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha",      type=float, default=0.05)
    parser.add_argument("--n_clients",  type=int,   default=5)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--gpu",        type=int,   default=0)
    parser.add_argument("--epochs_ce",  type=int,   default=EPOCHS_CE)
    parser.add_argument("--epochs_ssl", type=int,   default=EPOCHS_SSL)
    args = parser.parse_args()

    EPOCHS_CE  = args.epochs_ce
    EPOCHS_SSL = args.epochs_ssl

    run(args.alpha, args.n_clients, args.seed, args.gpu)


if __name__ == "__main__":
    main()