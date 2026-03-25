"""
Pipeline I — 纯 Intrinsic Knowledge for One-Shot FL
═══════════════════════════════════════════════════════
目标: 证明在极端 non-IID (低α) 下, intrinsic knowledge 优于 relational knowledge

核心思想:
  - Backbone: VICReg (纯SSL, 零标签, 零类间信息)
  - Expert: per-class conditional reconstruction (学"猫为什么是猫")
  - 推理: reconstruction error (标量, 天然跨client可比)
  - 不需要 Union aggregation

对比:
  Pipeline R (relational): ETF backbone → prototype matching
  Pipeline I (intrinsic):  VICReg backbone → reconstruction error
  → 两条曲线在 α* ≈ 0.18 交叉

运行:
  单个α:  python pipeline_intrinsic.py --alpha 0.05 --seed 42 --gpu 0
  全矩阵: python pipeline_intrinsic.py --seed 42 --gpu 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
import numpy as np
import json, os, time, warnings, argparse
from collections import defaultdict
from PIL import Image

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════
N_CLIENTS   = 5
N_CLASSES   = 10
FEAT_DIM    = 256
EXPAND_DIM  = 512
BATCH_SIZE  = 256
LR_BB       = 1e-3
LR_EXP      = 1e-3
EPOCHS_BB   = 600
EPOCHS_EXP  = 600
EXPERT_HD   = 128
EXPERT_LD   = 32
MARGIN      = 0.05

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)


# ═══════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════
def dirichlet_split(targets, n_clients, alpha, seed=42):
    rng = np.random.RandomState(seed)
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)
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
                client_indices[k].extend(idxs[start:end].tolist())
                client_class_counts[k][c] = int(counts[k])
            start = end
    return dict(client_indices), dict(client_class_counts)


def get_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])


# ═══════════════════════════════════════════════════════════
# VICReg 双增强 Dataset
# ═══════════════════════════════════════════════════════════
class DualAugDataset(Dataset):
    """包装 dataset, 返回同一张图片的两个不同增强版本
    标签完全不用, 但保留以便兼容 DataLoader"""
    def __init__(self, base_dataset, indices):
        self.data = base_dataset.data
        self.targets = base_dataset.targets
        self.indices = indices
        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(3, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img = Image.fromarray(self.data[real_idx])
        v1 = self.aug(img)
        v2 = self.aug(img)
        label = self.targets[real_idx]
        return v1, v2, label


# ═══════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════
class Backbone(nn.Module):
    def __init__(self, fd=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(256 * 2 * 2, fd)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return F.normalize(self.fc(x), dim=1)


class Expander(nn.Module):
    """VICReg projector: backbone features → expanded space"""
    def __init__(self, fd=256, ed=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fd, ed), nn.BatchNorm1d(ed), nn.ReLU(True),
            nn.Linear(ed, ed), nn.BatchNorm1d(ed), nn.ReLU(True),
            nn.Linear(ed, ed),
        )
    def forward(self, x):
        return self.net(x)


class ConditionalExpert(nn.Module):
    """Per-class reconstruction expert
    输入: feature + class_embedding → 重构 feature
    intrinsic 的核心: 只有"认识"这个类的 expert 能低误差重构"""
    def __init__(self, fd=256, ed=256, hd=128, ld=32):
        super().__init__()
        self.enc1 = nn.Linear(fd + ed, hd)
        self.ebn = nn.LayerNorm(hd)
        self.enc2 = nn.Linear(hd, ld)
        self.dec1 = nn.Linear(ld + ed, hd)
        self.dbn = nn.LayerNorm(hd)
        self.dec2 = nn.Linear(hd, fd)

    def encode(self, f, c):
        return self.enc2(F.relu(self.ebn(self.enc1(torch.cat([f, c], 1)))))

    def decode(self, z, c):
        return self.dec2(F.relu(self.dbn(self.dec1(torch.cat([z, c], 1)))))

    def forward(self, f, c):
        z = self.encode(f, c)
        return self.decode(z, c), z


# ═══════════════════════════════════════════════════════════
# VICReg Loss (完整版, 含防坍缩)
# ═══════════════════════════════════════════════════════════
def vicreg_loss(z1, z2, sim_w=25.0, std_w=25.0, cov_w=1.0):
    """
    Invariance: z1 和 z2 应该相似 (同一图的两个增强)
    Variance:   每个维度的标准差 >= 1 (★ 防坍缩!)
    Covariance: 不同维度去相关 (防冗余)
    """
    N, D = z1.shape

    # Invariance
    sim = F.mse_loss(z1, z2)

    # Variance — 这是防坍缩的关键
    std1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    std_loss = F.relu(1.0 - std1).mean() + F.relu(1.0 - std2).mean()

    # Covariance
    z1c = z1 - z1.mean(dim=0)
    z2c = z2 - z2.mean(dim=0)
    cov1 = (z1c.T @ z1c) / max(N - 1, 1)
    cov2 = (z2c.T @ z2c) / max(N - 1, 1)

    def off_diag(m):
        return m.pow(2).sum() - m.diagonal().pow(2).sum()

    cov_loss = (off_diag(cov1) + off_diag(cov2)) / D

    return sim_w * sim + std_w * std_loss + cov_w * cov_loss


# ═══════════════════════════════════════════════════════════
# Phase 1: VICReg Backbone Training (纯SSL, 无标签)
# ═══════════════════════════════════════════════════════════
def train_backbone_vicreg(indices, base_dataset, device,
                          epochs=EPOCHS_BB, lr=LR_BB):
    """
    纯 SSL backbone 训练 — 完全不看标签
    VICReg 学习: 翻转/裁剪/变色后的猫还是"同一个东西"
    这就是 intrinsic knowledge 的基础: 视觉不变性
    """
    dataset = DualAugDataset(base_dataset, indices)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=len(indices) > BATCH_SIZE,
                        num_workers=8, pin_memory=True, persistent_workers=True)

    bb = Backbone(FEAT_DIM).to(device)
    expander = Expander(FEAT_DIM, EXPAND_DIM).to(device)
    params = list(bb.parameters()) + list(expander.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    USE_BF16 = (torch.cuda.is_available()
                and torch.cuda.get_device_capability()[0] >= 8)
    amp_ctx = (torch.amp.autocast('cuda', dtype=torch.bfloat16) if USE_BF16
               else torch.amp.autocast('cuda', enabled=False))

    bb.train(); expander.train()

    for ep in range(epochs):
        total_loss = 0; n_batch = 0
        for v1, v2, _ in loader:   # _ = label, 完全不用!
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)

            with amp_ctx:
                f1, f2 = bb(v1), bb(v2)
                z1, z2 = expander(f1), expander(f2)
                loss = vicreg_loss(z1, z2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += loss.item(); n_batch += 1

        sch.step()

        if (ep + 1) % 100 == 0 or ep == 0:
            avg = total_loss / max(n_batch, 1)
            with torch.no_grad():
                bb.eval()
                sv1, sv2, _ = next(iter(loader))
                sf = bb(sv1[:min(64, len(sv1))].to(device))
                feat_std = sf.std(dim=0).mean().item()
                feat_collapse = (sf.std(dim=0) < 0.01).float().mean().item()
                bb.train()
            print(f"      BB ep {ep+1:3d}/{epochs}  loss={avg:.4f}  "
                  f"feat_std={feat_std:.4f}  collapse_ratio={feat_collapse:.2%}")
            if feat_std < 0.01:
                print(f"      ⚠️  WARNING: possible collapse!")

    return bb


# ═══════════════════════════════════════════════════════════
# Phase 2: Compute Prototypes (类内均值 → Expert 的 class embedding)
# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def compute_prototypes(bb, indices, targets, device):
    """
    类内特征均值 — 纯 intrinsic 统计量
    不含类间关系: prototype[cat] 不依赖 dog 的存在
    """
    bb.eval()
    test_tf = get_test_transform()
    dataset = datasets.CIFAR10('./data', train=True, transform=test_tf)

    class_indices = defaultdict(list)
    for idx in indices:
        class_indices[targets[idx]].append(idx)

    prototypes = {}
    for c, cidxs in class_indices.items():
        if len(cidxs) == 0:
            continue
        loader = DataLoader(Subset(dataset, cidxs), batch_size=256,
                            shuffle=False, num_workers=4, pin_memory=True)
        feats = []
        for x, _ in loader:
            feats.append(bb(x.to(device)).cpu())
        feats = torch.cat(feats, 0)
        prototypes[c] = F.normalize(feats.mean(0), dim=0)

    return prototypes


# ═══════════════════════════════════════════════════════════
# Phase 3: Expert Training
# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def preextract_features(bb, indices, device):
    """预提取特征 (用测试变换, 不加随机增强)"""
    bb.eval()
    test_tf = get_test_transform()
    dataset = datasets.CIFAR10('./data', train=True, transform=test_tf)
    loader = DataLoader(Subset(dataset, indices), batch_size=256,
                        shuffle=False, num_workers=4, pin_memory=True)
    feats = []
    for x, _ in loader:
        feats.append(bb(x.to(device)))
    return torch.cat(feats, 0)   # (N, FEAT_DIM) on device


def train_expert(expert, cached_feats, class_proto, all_protos,
                 other_classes, device,
                 epochs=EPOCHS_EXP, lr=LR_EXP, margin=MARGIN):
    """
    训练 per-class expert

    正: 本类特征 + 本类 prototype → 低重构误差
    负: 本类特征 + 其他类 prototype → 高重构误差 (margin)
        伪造特征 + 其他类 prototype → 高重构误差
        伪造特征 + 本类 prototype → 高重构误差

    这些都是 intrinsic 的: expert 学 "我只认识猫, 不认识非猫"
    不学 "猫和狗的决界在哪"
    """
    expert = expert.to(device)
    N = cached_feats.size(0)
    proto_dim = FEAT_DIM

    # 构建 prototype tensor
    proto_tensor = torch.zeros(N_CLASSES, proto_dim, device=device)
    for c, p in all_protos.items():
        proto_tensor[c] = p.to(device)
    class_proto_dev = class_proto.to(device)
    other_cls_tensor = torch.tensor(other_classes, device=device, dtype=torch.long)

    opt = torch.optim.Adam(expert.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bs = min(64, N)
    n_neg = 64

    for ep in range(epochs):
        expert.train()
        perm = torch.randperm(N, device=device)

        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            fp = cached_feats[idx]
            B = fp.size(0)

            # L1: 正样本重构
            co = class_proto_dev.unsqueeze(0).expand(B, -1)
            fr1, _ = expert(fp, co)
            l1 = F.mse_loss(fr1, fp)

            # L2: 本类特征 + 错误 prototype → 重构差
            nc = other_cls_tensor[torch.randint(0, len(other_classes), (B,), device=device)]
            neg_proto = proto_tensor[nc]
            fr2, _ = expert(fp, neg_proto)
            l2 = F.relu(margin - ((fp - fr2) ** 2).mean(1)).mean()

            # L3: 伪造特征 + 其他 prototype → 重构差
            nc2 = other_cls_tensor[torch.randint(0, len(other_classes), (n_neg,), device=device)]
            scale = 0.05 + 0.25 * torch.rand(n_neg, 1, device=device)
            fake = F.normalize(
                proto_tensor[nc2] + torch.randn(n_neg, proto_dim, device=device) * scale,
                dim=1
            )
            fr3, _ = expert(fake, proto_tensor[nc2])
            l3 = F.relu(margin - ((fake - fr3) ** 2).mean(1)).mean()

            # L4: 伪造特征 + 本类 prototype → 重构差
            fr4, _ = expert(fake, class_proto_dev.unsqueeze(0).expand(n_neg, -1))
            l4 = F.relu(margin - ((fake - fr4) ** 2).mean(1)).mean()

            loss = l1 + (l2 + l3 + l4)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        sch.step()

    expert.eval()
    return expert


# ═══════════════════════════════════════════════════════════
# Phase 4: Intrinsic Inference (重构误差, 标量聚合)
# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_intrinsic(bbs, client_experts, client_protos, ccc,
                       test_loader, device):
    """
    核心: 重构误差是标量, 天然跨 client 可比, 不需要 Union
    """
    N_test = 10000
    K = len(bbs)

    # 收集误差: (K, N, C)
    all_errors = torch.full((K, N_test, N_CLASSES), float('inf'))
    all_labels = []

    for k in range(K):
        bbs[k].to(device).eval()
        # 提取特征
        feats_list = []
        for x, y in test_loader:
            feats_list.append(bbs[k](x.to(device)))
            if k == 0:
                all_labels.append(y)
        feats_k = torch.cat(feats_list, 0)

        # 计算每个 expert 的重构误差
        for c, expert in client_experts[k].items():
            expert.to(device).eval()
            proto_c = client_protos[k][c].to(device)
            errs = []
            for i in range(0, N_test, 256):
                f_batch = feats_k[i:i + 256]
                B = f_batch.size(0)
                proto_exp = proto_c.unsqueeze(0).expand(B, -1)
                fr, _ = expert(f_batch, proto_exp)
                err = ((f_batch - fr) ** 2).mean(1)
                errs.append(err.cpu())
            all_errors[k, :, c] = torch.cat(errs)
            expert.cpu()

        bbs[k].cpu()
        torch.cuda.empty_cache()

    labels = torch.cat(all_labels).numpy()

    # ═══ 多种聚合策略 ═══

    results = {}
    errs_clean = all_errors.clone()
    errs_clean[errs_clean == float('inf')] = 1e6

    # S1: 全局 min_k
    best_per_class, _ = errs_clean.min(dim=0)  # (N, C)
    preds = best_per_class.argmin(dim=1).numpy()
    results['S1_min_k'] = (preds == labels).mean()

    # S2: Quality min — 只用 n >= threshold 的 expert
    for min_n in [50, 100, 200, 500]:
        filtered = torch.full_like(all_errors, 1e6)
        for k in range(K):
            for c in client_experts[k]:
                if ccc[k].get(c, 0) >= min_n:
                    filtered[k, :, c] = errs_clean[k, :, c]
        best_f, _ = filtered.min(dim=0)
        preds_f = best_f.argmin(1).numpy()
        # fallback: 如果某样本所有类都无 expert
        fallback = (best_f.min(1)[0] >= 1e5).numpy()
        preds_f[fallback] = preds[fallback]
        results[f'S2_quality_n{min_n}'] = (preds_f == labels).mean()

    # S3: Top-quality — 每类只用训练量最大的 client
    best_err_top = torch.full((N_test, N_CLASSES), 1e6)
    for c in range(N_CLASSES):
        best_k, best_n = -1, 0
        for k in range(K):
            n = ccc[k].get(c, 0)
            if n > best_n and c in client_experts[k]:
                best_n = n; best_k = k
        if best_k >= 0:
            best_err_top[:, c] = errs_clean[best_k, :, c]
    preds_top = best_err_top.argmin(1).numpy()
    results['S3_top_quality'] = (preds_top == labels).mean()

    # S4: Per-client z-score normalized ensemble
    for min_n_thr in [0, 50, 100]:
        ensemble = torch.zeros(N_test, N_CLASSES)
        for k in range(K):
            valid_c = [c for c in client_experts[k]
                       if ccc[k].get(c, 0) >= min_n_thr]
            if not valid_c:
                continue

            cl = torch.zeros(N_test, N_CLASSES)
            cl_mask = torch.zeros(N_test, N_CLASSES, dtype=torch.bool)
            for c in valid_c:
                cl[:, c] = -errs_clean[k, :, c]   # 误差取负 → 越高越好
                cl_mask[:, c] = True

            # z-score per client
            n_valid = cl_mask.sum(1, keepdim=True).clamp(min=1)
            cm = (cl * cl_mask.float()).sum(1, keepdim=True) / n_valid
            diff = (cl - cm) * cl_mask.float()
            cs = ((diff ** 2).sum(1, keepdim=True) / n_valid).sqrt() + 1e-8
            cl_n = diff / cs
            cl_n[~cl_mask] = 0

            # 按训练量加权
            for c in valid_c:
                w = np.log(ccc[k].get(c, 0) + 1)
                ensemble[:, c] += cl_n[:, c] * w

        preds_e = ensemble.argmax(1).numpy()
        results[f'S4_ensemble_n{min_n_thr}'] = (preds_e == labels).mean()

    # S5: Weighted min — 误差除以 log(n+1)
    for min_n_thr in [50, 100]:
        scaled = torch.full((K, N_test, N_CLASSES), 1e6)
        for k in range(K):
            for c in client_experts[k]:
                n = ccc[k].get(c, 0)
                if n < min_n_thr:
                    continue
                w = np.log(n + 1)
                scaled[k, :, c] = errs_clean[k, :, c] / w
        best_s, _ = scaled.min(dim=0)
        preds_s = best_s.argmin(1).numpy()
        fallback_s = (best_s.min(1)[0] >= 1e5).numpy()
        preds_s[fallback_s] = preds[fallback_s]
        results[f'S5_wmin_n{min_n_thr}'] = (preds_s == labels).mean()

    return results, labels, all_errors


# ═══════════════════════════════════════════════════════════
# Phase 5: Relational Baseline (prototype matching, 同一 backbone)
# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_relational(bbs, client_protos, ccc, test_loader, device):
    """
    同一个 VICReg backbone, 但用 relational 推理 (prototype matching)
    → 预期在低 α 时比 intrinsic 差
    """
    N_test = 10000; K = len(bbs)

    scores = torch.zeros(N_test, N_CLASSES)
    weights = torch.zeros(N_CLASSES)
    all_labels = []

    for k in range(K):
        bbs[k].to(device).eval()
        feats = []
        for x, y in test_loader:
            feats.append(bbs[k](x.to(device)).cpu())
            if k == 0:
                all_labels.append(y)
        feats = F.normalize(torch.cat(feats, 0), dim=1)
        bbs[k].cpu()

        for c, proto in client_protos[k].items():
            proto_n = F.normalize(proto.unsqueeze(0), dim=1)
            sim = torch.mm(feats, proto_n.T).squeeze(1)
            w = float(ccc[k].get(c, 0))
            if w > 0:
                scores[:, c] += sim * w
                weights[c] += w

        torch.cuda.empty_cache()

    for c in range(N_CLASSES):
        if weights[c] > 0:
            scores[:, c] /= weights[c]
        else:
            scores[:, c] = -float('inf')

    labels = torch.cat(all_labels).numpy()
    preds = scores.argmax(1).numpy()
    return (preds == labels).mean()


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
def run_experiment(alpha, seed, gpu):
    device = torch.device(f'cuda:{gpu}')
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f'\n{"=" * 70}')
    print(f'  Pipeline I: Intrinsic Knowledge  |  α={alpha}  seed={seed}')
    print(f'{"=" * 70}')

    # ── 数据分割 ──
    base_dataset = datasets.CIFAR10('./data', train=True, download=True)
    targets = np.array(base_dataset.targets)
    ci, cc = dirichlet_split(base_dataset.targets, N_CLIENTS, alpha, seed=seed)

    print(f'\n  数据分布:')
    for k in range(N_CLIENTS):
        c_cc = cc.get(k, {})
        top = sorted(c_cc.items(), key=lambda x: -x[1])[:3]
        n_cls = sum(1 for v in c_cc.values() if v > 0)
        n_smp = sum(c_cc.values())
        print(f'    Client {k}: {n_cls:2d} cls, {n_smp:5d} smp  '
              f'top: {", ".join(f"c{c}={n}" for c, n in top)}')

    test_ds = datasets.CIFAR10('./data', train=False, transform=get_test_transform())
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=4, pin_memory=True)

    # ── Phase 1: VICReg Backbone (纯SSL) ──
    print(f'\n{"=" * 60}')
    print(f'  Phase 1: VICReg Backbone (纯SSL, 零标签)')
    print(f'{"=" * 60}')

    bbs = []
    t0 = time.time()
    for k in range(N_CLIENTS):
        n_smp = sum(cc[k].values())
        n_cls = sum(1 for v in cc[k].values() if v > 0)
        print(f'\n  Client {k} ({n_smp} samples, {n_cls} classes):')

        sk = seed + hash(('vicreg', alpha, k)) % 100000
        torch.manual_seed(sk)
        np.random.seed(sk % (2 ** 31))

        bb = train_backbone_vicreg(ci[k], base_dataset, device,
                                   epochs=EPOCHS_BB, lr=LR_BB)
        bbs.append(bb.cpu())
        torch.cuda.empty_cache()

    t_bb = time.time() - t0
    print(f'\n  ⏱ Backbone: {t_bb:.0f}s ({t_bb/60:.1f}min)')

    # ── Phase 2: Prototypes ──
    print(f'\n{"=" * 60}')
    print(f'  Phase 2: Prototypes (类内均值)')
    print(f'{"=" * 60}')

    client_protos = []
    for k in range(N_CLIENTS):
        bbs[k].to(device)
        protos = compute_prototypes(bbs[k], ci[k], targets, device)
        client_protos.append(protos)
        bbs[k].cpu()
        torch.cuda.empty_cache()
        print(f'  Client {k}: {len(protos)} protos '
              f'({sorted(protos.keys())})')

    # ── Phase 3: Expert Training ──
    print(f'\n{"=" * 60}')
    print(f'  Phase 3: Expert Training')
    print(f'{"=" * 60}')

    client_experts = []
    t1 = time.time()
    for k in range(N_CLIENTS):
        bbs[k].to(device)
        classes_k = sorted(cc[k].keys())
        protos_k = client_protos[k]
        print(f'\n  Client {k}: {len(classes_k)} experts')

        # 按类提取特征
        class_feats = {}
        for c in classes_k:
            c_indices = [idx for idx in ci[k] if targets[idx] == c]
            if len(c_indices) < 2:
                continue
            class_feats[c] = preextract_features(bbs[k], c_indices, device)

        experts_k = {}
        for c in classes_k:
            if c not in class_feats:
                continue
            # 负样本类: 当前 client 拥有 prototype 的其他类
            other_c = [j for j in protos_k if j != c]
            if not other_c:
                # 只有一个类: 用全局随机方向作负样本
                # 生成随机 prototype 替代
                n_fake = min(3, N_CLASSES - 1)
                fake_protos = F.normalize(torch.randn(n_fake, FEAT_DIM), dim=1)
                for i, fp in enumerate(fake_protos):
                    fake_c = (c + 1 + i) % N_CLASSES
                    protos_k[fake_c] = fp
                other_c = [fake_c for fake_c in protos_k if fake_c != c]

            t_exp = time.time()
            expert = ConditionalExpert(FEAT_DIM, FEAT_DIM, EXPERT_HD, EXPERT_LD)
            expert = train_expert(
                expert, class_feats[c], protos_k[c], protos_k,
                other_c, device, epochs=EPOCHS_EXP, lr=LR_EXP
            )

            # 训练质量检查
            with torch.no_grad():
                ne = min(256, class_feats[c].size(0))
                proto_exp = protos_k[c].to(device).unsqueeze(0).expand(ne, -1)
                fr, _ = expert.to(device)(class_feats[c][:ne], proto_exp)
                mse = ((class_feats[c][:ne] - fr) ** 2).mean().item()

            experts_k[c] = expert.cpu()
            n_samp = cc[k].get(c, 0)
            print(f'    c{c}: n={n_samp:5d} MSE={mse:.6f} ({time.time()-t_exp:.1f}s)')

        client_experts.append(experts_k)
        bbs[k].cpu()
        torch.cuda.empty_cache()

    t_exp_total = time.time() - t1
    print(f'\n  ⏱ Experts: {t_exp_total:.0f}s ({t_exp_total/60:.1f}min)')

    # ── Phase 4: Evaluation ──
    print(f'\n{"=" * 60}')
    print(f'  Phase 4: Evaluation')
    print(f'{"=" * 60}')

    # Intrinsic
    print(f'\n  ── Intrinsic (reconstruction error) ──')
    intrinsic_results, labels, all_errors = evaluate_intrinsic(
        bbs, client_experts, client_protos, cc, test_loader, device
    )
    for name, acc in sorted(intrinsic_results.items(), key=lambda x: -x[1]):
        print(f'    {name:30s}: {acc:.2%}')

    # Relational (same backbone)
    print(f'\n  ── Relational (prototype matching, same backbone) ──')
    rel_acc = evaluate_relational(bbs, client_protos, cc, test_loader, device)
    print(f'    prototype_matching          : {rel_acc:.2%}')

    # ── Summary ──
    best_i = max(intrinsic_results.values())
    best_i_name = max(intrinsic_results, key=intrinsic_results.get)

    print(f'\n{"=" * 70}')
    print(f'  ★ RESULTS  α={alpha}  seed={seed}')
    print(f'{"=" * 70}')
    print(f'  Relational (prototype):  {rel_acc:.2%}')
    print(f'  Intrinsic  (best):       {best_i:.2%}  ({best_i_name})')
    print(f'  Gap (I - R):             {(best_i - rel_acc) * 100:+.1f} pp')
    print(f'  Time: BB={t_bb:.0f}s  Exp={t_exp_total:.0f}s  '
          f'Total={time.time() - t0:.0f}s')

    # ── Save ──
    os.makedirs('results', exist_ok=True)
    out = {
        'alpha': alpha, 'seed': seed,
        'relational_acc': float(rel_acc),
        'intrinsic_results': {k: float(v) for k, v in intrinsic_results.items()},
        'best_intrinsic': float(best_i),
        'best_intrinsic_name': best_i_name,
        'gap_pp': float((best_i - rel_acc) * 100),
        'time_bb': t_bb, 'time_exp': t_exp_total,
    }
    outpath = f'results/pipeline_I_a{alpha}_s{seed}.json'
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'  Saved: {outpath}\n')
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--alpha', type=float, default=None,
                   help='单个α值, 不指定则跑 [0.05, 0.1, 0.3, 0.5, 1.0]')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--epochs_bb', type=int, default=600)
    p.add_argument('--epochs_exp', type=int, default=600)
    args = p.parse_args()

    global EPOCHS_BB, EPOCHS_EXP
    EPOCHS_BB = args.epochs_bb
    EPOCHS_EXP = args.epochs_exp

    if args.alpha is not None:
        run_experiment(args.alpha, args.seed, args.gpu)
    else:
        all_results = []
        for alpha in [0.05, 0.1, 0.3, 0.5, 1.0]:
            res = run_experiment(alpha, args.seed, args.gpu)
            all_results.append(res)

        print(f'\n{"=" * 70}')
        print(f'  CROSSOVER SUMMARY')
        print(f'{"=" * 70}')
        print(f'  {"α":>6s} | {"Relational":>12s} | {"Intrinsic":>12s} | '
              f'{"Gap":>10s} | Winner')
        print(f'  {"-" * 6} | {"-" * 12} | {"-" * 12} | '
              f'{"-" * 10} | ------')
        for r in all_results:
            rel = r['relational_acc']
            intr = r['best_intrinsic']
            gap = r['gap_pp']
            w = 'Intrinsic ★' if gap > 0 else 'Relational'
            print(f'  {r["alpha"]:6.2f} | {rel:>11.2%} | {intr:>11.2%} | '
                  f'{gap:>+9.1f}pp | {w}')

        outpath = f'results/pipeline_I_summary_s{args.seed}.json'
        with open(outpath, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'\n  Summary: {outpath}')


if __name__ == '__main__':
    main()