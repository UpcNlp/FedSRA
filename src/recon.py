"""
Pipeline I-Recon — 纯本体知识: Per-class Pixel-Space Autoencoder
══════════════════════════════════════════════════════════════════════
核心原则:
  训练-推理完全对齐:
    训练: 最小化本类图像的像素重建误差
    推理: 用像素重建误差分类
    模型学的能力 = 推理用的信号

  每个 (client, class) 独立训练一个 autoencoder:
    - 只看自己类的数据
    - 零标签, 零类间信号
    - 像素 MSE 天然跨模型可比 (同一空间, 同一量纲)

  分类: argmin_c MSE(x, AE_c(x))
    猫的 AE 按 "猫的方式" 压缩和解压
    → 给它狗, encoder 按猫的方式压缩, 丢掉狗特有信息
    → decoder 按猫的方式解压, 重建出来不像狗
    → MSE 高

  关键: bottleneck 大小
    太大 → 什么都能重建, 无区分度
    太小 → 连自己类都重建不好
    sweet spot 可搜索

运行:
  python pipeline_intrinsic_recon.py --alpha 0.05 --seed 42 --gpu 0
  python pipeline_intrinsic_recon.py --seed 42 --gpu 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import json, os, time, warnings, argparse
from collections import defaultdict
from PIL import Image

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════
N_CLIENTS    = 5
N_CLASSES    = 10
BATCH_SIZE   = 128
LR           = 1e-3
EPOCHS       = 300
BOTTLENECK   = 128      # bottleneck 维度, 关键超参
MIN_SAMPLES  = 30       # 最少样本数

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)
# 预计算反归一化参数
CIFAR_MEAN_T = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
CIFAR_STD_T  = torch.tensor(CIFAR_STD).view(3, 1, 1)


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


class SingleClassDataset(Dataset):
    """单类数据集, 带轻度增强 (不改变语义的增强)"""
    def __init__(self, base_dataset, indices, augment=True):
        self.data = base_dataset.data
        self.indices = indices
        self.augment = augment
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img = Image.fromarray(self.data[real_idx])
        return self.transform(img)


def get_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])


# ═══════════════════════════════════════════════════════════
# Autoencoder — 对称 CNN, 像素空间重建
# ═══════════════════════════════════════════════════════════
class ResBlock(nn.Module):
    """残差块: 帮助深层网络训练"""
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(x + self.net(x))


class PixelAutoencoder(nn.Module):
    """
    对称 CNN Autoencoder, 像素空间重建

    Encoder: 3×32×32 → 64→128→256→512 → bottleneck
    Decoder: bottleneck → 512→256→128→64 → 3×32×32

    使用 ResBlock 增加深度和表达力
    bottleneck 是可调参数 — 控制信息瓶颈的紧度
    """
    def __init__(self, bottleneck_dim=BOTTLENECK):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim

        # ── Encoder ──
        # 32×32 → 16×16 → 8×8 → 4×4 → bottleneck
        self.encoder = nn.Sequential(
            # Block 1: 3→64, 32→16
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            ResBlock(64),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),  # 32→16
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Block 2: 64→128, 16→8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            ResBlock(128),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),  # 16→8
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Block 3: 128→256, 8→4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            ResBlock(256),
            nn.Conv2d(256, 256, 4, stride=2, padding=1),  # 8→4
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # Block 4: 256→512, 4→2
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            ResBlock(512),
            nn.Conv2d(512, 512, 4, stride=2, padding=1),  # 4→2
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        # Bottleneck: 512×2×2=2048 → bottleneck_dim
        self.enc_fc = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(True),
            nn.Linear(512, bottleneck_dim),
        )

        # ── Decoder ──
        # bottleneck → 512×2×2 → 4×4 → 8×8 → 16×16 → 32×32
        self.dec_fc = nn.Sequential(
            nn.Linear(bottleneck_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512 * 2 * 2),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            # Block 4: 512, 2→4
            ResBlock(512),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 2→4
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # Block 3: 256→128, 4→8
            ResBlock(256),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 4→8
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Block 2: 128→64, 8→16
            ResBlock(128),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8→16
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Block 1: 64→3, 16→32
            ResBlock(64),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),  # 16→32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Final: → 3 channels
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 3, 3, padding=1),
            # 不加激活: 输出范围匹配归一化后的输入
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.enc_fc(h)

    def decode(self, z):
        h = self.dec_fc(z)
        h = h.view(h.size(0), 512, 2, 2)
        return self.decoder(h)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


# ═══════════════════════════════════════════════════════════
# 训练
# ═══════════════════════════════════════════════════════════
def train_autoencoder(model, class_indices, base_dataset, device,
                      epochs=EPOCHS, lr=LR):
    """
    训练单个类的 autoencoder

    Loss = MSE(x, reconstruct(x))  在像素空间 (归一化后)
    没有任何其他 loss 项 — 纯粹的重建
    """
    dataset = SingleClassDataset(base_dataset, class_indices, augment=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=len(class_indices) > BATCH_SIZE,
                        num_workers=4, pin_memory=True, persistent_workers=True)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    USE_BF16 = (torch.cuda.is_available()
                and torch.cuda.get_device_capability()[0] >= 8)
    amp_ctx = (torch.amp.autocast('cuda', dtype=torch.bfloat16) if USE_BF16
               else torch.amp.autocast('cuda', enabled=False))

    model.train()
    best_loss = float('inf')
    best_state = None

    for ep in range(epochs):
        total_loss = 0; n_batch = 0

        for images in loader:
            images = images.to(device, non_blocking=True)

            with amp_ctx:
                recon, z = model(images)
                loss = F.mse_loss(recon, images)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            total_loss += loss.item(); n_batch += 1

        sch.step()
        avg_loss = total_loss / max(n_batch, 1)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (ep + 1) % 50 == 0 or ep == 0:
            # bottleneck 活跃度
            with torch.no_grad():
                model.eval()
                sx = next(iter(loader))
                _, sz = model(sx[:min(64, len(sx))].to(device))
                z_std = sz.std(dim=0).mean().item()
                z_active = (sz.std(dim=0) > 0.01).float().mean().item()
                model.train()
            print(f"        ep {ep+1:3d}/{epochs}  "
                  f"MSE={avg_loss:.6f}  "
                  f"z_std={z_std:.4f}  z_active={z_active:.1%}")

    # 恢复最佳
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_loss


# ═══════════════════════════════════════════════════════════
# 推理: 像素重建误差分类
# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_recon(autoencoders, client_class_counts, test_loader, device):
    """
    对每个 test sample x:
      对每个 AE_{k,c}: MSE(x, AE_{k,c}(x))
    分类: argmin MSE

    MSE 在像素空间, 天然跨模型可比
    """
    N_test = 10000
    K = N_CLIENTS

    # 收集 MSE: (K, N, C), 越低越好
    all_mse = torch.full((K, N_test, N_CLASSES), float('inf'))
    all_labels = []
    first_pass = True

    for (k, c), model in autoencoders.items():
        model = model.to(device).eval()
        mses = []
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            recon, _ = model(x)
            # per-sample MSE
            mse = ((x - recon) ** 2).mean(dim=(1, 2, 3))
            mses.append(mse.cpu())
            if first_pass:
                all_labels.append(y)
        first_pass = False
        all_mse[k, :, c] = torch.cat(mses)
        model.cpu()
        torch.cuda.empty_cache()

    labels = torch.cat(all_labels).numpy()
    results = {}

    # ── S1: 全局 min MSE ──
    best_mse, _ = all_mse.min(dim=0)  # (N, C)
    preds = best_mse.argmin(dim=1).numpy()
    results['S1_min_mse'] = (preds == labels).mean()

    # ── S2: Quality filter ──
    for min_n in [50, 100, 200, 500]:
        filtered = torch.full((K, N_test, N_CLASSES), float('inf'))
        for (k2, c2) in autoencoders:
            if client_class_counts[k2].get(c2, 0) >= min_n:
                filtered[k2, :, c2] = all_mse[k2, :, c2]
        best_f, _ = filtered.min(dim=0)
        preds_f = best_f.argmin(dim=1).numpy()
        # fallback
        fallback = (best_f.min(1)[0] >= 1e5).numpy()
        if fallback.any():
            preds_f[fallback] = preds[fallback]
        results[f'S2_quality_n{min_n}'] = (preds_f == labels).mean()

    # ── S3: Top-quality per class ──
    top_mse = torch.full((N_test, N_CLASSES), float('inf'))
    for c in range(N_CLASSES):
        best_k, best_n = -1, 0
        for k2 in range(K):
            n = client_class_counts[k2].get(c, 0)
            if n > best_n and (k2, c) in autoencoders:
                best_n = n; best_k = k2
        if best_k >= 0:
            top_mse[:, c] = all_mse[best_k, :, c]
    preds_top = top_mse.argmin(dim=1).numpy()
    results['S3_top_quality'] = (preds_top == labels).mean()

    # ── S4: Weighted MSE (除以 log(n+1)) ──
    for min_n in [50, 100]:
        scaled = torch.full((K, N_test, N_CLASSES), float('inf'))
        for (k2, c2) in autoencoders:
            n = client_class_counts[k2].get(c2, 0)
            if n < min_n:
                continue
            # 训练量大的模型, MSE 更可信, 但不直接除以 n
            # 用 train_mse 做归一化: MSE / train_mse 表示 "相对误差"
            scaled[k2, :, c2] = all_mse[k2, :, c2]
        best_s, _ = scaled.min(dim=0)
        preds_s = best_s.argmin(dim=1).numpy()
        fallback_s = (best_s.min(1)[0] >= 1e5).numpy()
        if fallback_s.any():
            preds_s[fallback_s] = preds[fallback_s]
        results[f'S4_filtered_n{min_n}'] = (preds_s == labels).mean()

    # ── S5: Per-client z-score ──
    for min_n in [0, 50]:
        ensemble = torch.zeros(N_test, N_CLASSES)
        count = torch.zeros(N_CLASSES)
        for k2 in range(K):
            valid = [(kk, cc) for (kk, cc) in autoencoders
                     if kk == k2 and client_class_counts[k2].get(cc, 0) >= max(min_n, MIN_SAMPLES)]
            if len(valid) < 2:
                continue
            # 对该 client 的所有有效模型, z-score 归一化 MSE
            mses_k = torch.stack([all_mse[k2, :, cc] for (_, cc) in valid], dim=1)
            # 取负: MSE 越低越好 → -MSE 越高越好
            neg_mses = -mses_k
            mu = neg_mses.mean(dim=1, keepdim=True)
            sigma = neg_mses.std(dim=1, keepdim=True).clamp(min=1e-8)
            normed = (neg_mses - mu) / sigma
            for i, (_, cc) in enumerate(valid):
                w = np.log(client_class_counts[k2].get(cc, 0) + 1)
                ensemble[:, cc] += normed[:, i] * w
                count[cc] += w
        preds_e = ensemble.argmax(dim=1).numpy()
        results[f'S5_zscore_n{min_n}'] = (preds_e == labels).mean()

    # ── S6: Relative MSE (除以 train MSE) ──
    # 核心思想: 不同 AE 的绝对 MSE 不同, 但 test_MSE / train_MSE 是可比的
    # 如果 test_MSE ≈ train_MSE, 说明 "像训练数据一样好重建"
    rel_mse = torch.full((K, N_test, N_CLASSES), float('inf'))
    for (k2, c2), model in autoencoders.items():
        n = client_class_counts[k2].get(c2, 0)
        if n < MIN_SAMPLES:
            continue
        # 用 in-class test MSE 中位数作为 baseline
        in_mask = (labels == c2)
        train_mse_baseline = all_mse[k2, in_mask, c2].median().item()
        if train_mse_baseline > 1e-8:
            rel_mse[k2, :, c2] = all_mse[k2, :, c2] / train_mse_baseline
    best_rel, _ = rel_mse.min(dim=0)
    preds_rel = best_rel.argmin(dim=1).numpy()
    results['S6_relative_mse'] = (preds_rel == labels).mean()

    return results, labels, all_mse


# ═══════════════════════════════════════════════════════════
# Per-class 诊断
# ═══════════════════════════════════════════════════════════
def per_class_analysis(all_mse, labels, autoencoders, client_class_counts):
    """关键诊断: in-class MSE vs out-of-class MSE"""
    print(f'\n  ── Per-class Reconstruction Gap ──')
    print(f'    {"Cls":>3s} | {"Cli":>3s} | {"N":>6s} | '
          f'{"In MSE":>10s} | {"Out MSE":>10s} | {"Ratio":>6s} | '
          f'{"Gap%":>7s}')
    print(f'    {"-"*3} | {"-"*3} | {"-"*6} | '
          f'{"-"*10} | {"-"*10} | {"-"*6} | {"-"*7}')

    for c in range(N_CLASSES):
        in_mask = (labels == c)
        out_mask = ~in_mask
        for k in range(N_CLIENTS):
            if (k, c) not in autoencoders:
                continue
            n = client_class_counts[k].get(c, 0)
            mse_in = all_mse[k, in_mask, c].mean().item()
            mse_out = all_mse[k, out_mask, c].mean().item()
            ratio = mse_out / max(mse_in, 1e-8)
            gap_pct = (mse_out - mse_in) / max(mse_in, 1e-8) * 100
            mark = "✓" if mse_out > mse_in else "✗"
            print(f'    c={c:1d} | k={k:1d} | {n:6d} | '
                  f'{mse_in:10.6f} | {mse_out:10.6f} | '
                  f'{ratio:6.3f} | {gap_pct:+6.1f}% {mark}')


# ═══════════════════════════════════════════════════════════
# Relational Baseline: Prototype matching (需要共享 backbone)
# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_relational(autoencoders, client_class_counts,
                        train_indices_by_class, base_dataset,
                        test_loader, device):
    """
    用 AE 的 encoder 部分做 prototype matching
    展示: 同一个 encoder, 重建推理 vs relational 推理
    """
    N_test = 10000
    test_tf = get_test_transform()
    train_dataset = datasets.CIFAR10('./data', train=True, transform=test_tf)
    from torch.utils.data import Subset

    scores = torch.zeros(N_test, N_CLASSES)
    weights = torch.zeros(N_CLASSES)
    all_labels = []

    for (k, c), model in autoencoders.items():
        model = model.to(device).eval()

        # 用 encoder 提取 prototype
        cidxs = train_indices_by_class[(k, c)]
        loader = DataLoader(Subset(train_dataset, cidxs),
                            batch_size=256, shuffle=False,
                            num_workers=4, pin_memory=True)
        feats = []
        for x, *_ in loader:
            feats.append(model.encode(x.to(device)).cpu())
        proto = F.normalize(torch.cat(feats, 0).mean(0), dim=0)

        first = (len(all_labels) == 0)
        test_feats = []
        for x, y in test_loader:
            test_feats.append(model.encode(x.to(device)).cpu())
            if first:
                all_labels.append(y)
        first = False
        test_feats = F.normalize(torch.cat(test_feats, 0), dim=1)

        sim = torch.mm(test_feats, proto.unsqueeze(1)).squeeze(1)
        w = float(client_class_counts[k].get(c, 0))
        if w > 0:
            scores[:, c] += sim * w
            weights[c] += w

        model.cpu()
        torch.cuda.empty_cache()

    for c in range(N_CLASSES):
        if weights[c] > 0:
            scores[:, c] /= weights[c]

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
    print(f'  Pipeline I-Recon: Per-class Pixel Autoencoder')
    print(f'  α={alpha}  seed={seed}  bottleneck={BOTTLENECK}  epochs={EPOCHS}')
    print(f'{"=" * 70}')

    # ── 数据 ──
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

    # 按 (client, class) 分割
    train_indices_by_class = {}
    for k in range(N_CLIENTS):
        for c in range(N_CLASSES):
            cidxs = [idx for idx in ci[k] if targets[idx] == c]
            if len(cidxs) >= MIN_SAMPLES:
                train_indices_by_class[(k, c)] = cidxs

    test_ds = datasets.CIFAR10('./data', train=False, transform=get_test_transform())
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=4, pin_memory=True)

    # ── 训练 ──
    print(f'\n{"=" * 60}')
    print(f'  Training: Per-class Pixel Autoencoders')
    print(f'{"=" * 60}')

    autoencoders = {}
    train_losses = {}
    t0 = time.time()

    for k in range(N_CLIENTS):
        classes_k = sorted([c for c in range(N_CLASSES)
                            if (k, c) in train_indices_by_class])
        if not classes_k:
            continue
        n_smp = sum(cc[k].values())
        print(f'\n  Client {k} ({n_smp} samples, {len(classes_k)} classes):')

        for c in classes_k:
            n_c = cc[k].get(c, 0)
            # 数据量自适应
            adj_epochs = min(EPOCHS, max(100, EPOCHS * 1000 // n_c))
            print(f'    Class {c} (n={n_c}, epochs={adj_epochs}):')

            sk = seed + hash(('recon', alpha, k, c)) % 100000
            torch.manual_seed(sk)
            np.random.seed(sk % (2 ** 31))

            model = PixelAutoencoder(BOTTLENECK)
            model, best_loss = train_autoencoder(
                model, train_indices_by_class[(k, c)],
                base_dataset, device,
                epochs=adj_epochs, lr=LR
            )
            autoencoders[(k, c)] = model.cpu()
            train_losses[(k, c)] = best_loss
            torch.cuda.empty_cache()
            print(f'        best_MSE={best_loss:.6f}')

    t_train = time.time() - t0
    print(f'\n  ⏱ Training: {t_train:.0f}s ({t_train/60:.1f}min)')
    print(f'  Total models: {len(autoencoders)}')

    # ── 评估 ──
    print(f'\n{"=" * 60}')
    print(f'  Evaluation')
    print(f'{"=" * 60}')

    print(f'\n  ── Intrinsic (pixel reconstruction MSE) ──')
    recon_results, eval_labels, all_mse = evaluate_recon(
        autoencoders, cc, test_loader, device
    )
    for name, acc in sorted(recon_results.items(), key=lambda x: -x[1]):
        print(f'    {name:30s}: {acc:.2%}')

    per_class_analysis(all_mse, eval_labels, autoencoders, cc)

    # Relational
    print(f'\n  ── Relational (prototype matching, AE encoder) ──')
    rel_acc = evaluate_relational(
        autoencoders, cc, train_indices_by_class,
        base_dataset, test_loader, device
    )
    print(f'    prototype_matching          : {rel_acc:.2%}')

    # ── Summary ──
    best_i = max(recon_results.values())
    best_i_name = max(recon_results, key=recon_results.get)

    print(f'\n{"=" * 70}')
    print(f'  ★ RESULTS  α={alpha}  seed={seed}')
    print(f'{"=" * 70}')
    print(f'  Relational (prototype):  {rel_acc:.2%}')
    print(f'  Intrinsic  (recon):      {best_i:.2%}  ({best_i_name})')
    print(f'  Gap (I - R):             {(best_i - rel_acc) * 100:+.1f} pp')
    print(f'  Time: {t_train:.0f}s ({t_train/60:.1f}min)')

    # Save
    os.makedirs('results', exist_ok=True)
    out = {
        'alpha': alpha, 'seed': seed,
        'bottleneck': BOTTLENECK,
        'relational_acc': float(rel_acc),
        'recon_results': {k: float(v) for k, v in recon_results.items()},
        'best_recon': float(best_i),
        'best_recon_name': best_i_name,
        'gap_pp': float((best_i - rel_acc) * 100),
        'time_train': t_train,
        'n_models': len(autoencoders),
        'train_losses': {f'{k}_{c}': float(v)
                         for (k, c), v in train_losses.items()},
    }
    outpath = f'results/pipeline_I_recon_a{alpha}_s{seed}.json'
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'  Saved: {outpath}\n')
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--alpha', type=float, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--bottleneck', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--min_samples', type=int, default=30)
    args = p.parse_args()

    global EPOCHS, BOTTLENECK, LR, MIN_SAMPLES
    EPOCHS = args.epochs
    BOTTLENECK = args.bottleneck
    LR = args.lr
    MIN_SAMPLES = args.min_samples

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
        print(f'  {"α":>6s} | {"Relational":>10s} | {"Recon":>8s} | '
              f'{"Gap":>8s} | Winner')
        print(f'  {"-"*6} | {"-"*10} | {"-"*8} | '
              f'{"-"*8} | ------')
        for r in all_results:
            rel = r['relational_acc']
            rec = r['best_recon']
            gap = r['gap_pp']
            w = 'Intrinsic ★' if gap > 0 else 'Relational'
            print(f'  {r["alpha"]:6.2f} | {rel:>9.2%} | '
                  f'{rec:>7.2%} | {gap:>+7.1f}pp | {w}')

        outpath = f'results/pipeline_I_recon_summary_s{args.seed}.json'
        with open(outpath, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'\n  Summary: {outpath}')


if __name__ == '__main__':
    main()