"""
MoE v16 — h-space Expert: 绕过FC信息瓶颈

核心洞察:
  backbone 的 FC 层 (1024→256) + normalize 把 OOD 样本不可逆地坍缩到已见类方向
  但 conv 输出 h (256,2,2)=1024维 中, OOD 样本和已见类仍有差异
  → 让 expert 在 h 空间操作, 而非 f 空间

新增:
  1. FiLM-Conditioned Conv Expert: 在 (256,2,2) 空间操作, 保留空间结构
  2. 可视化: h-space vs f-space 对 seen/unseen 类的分布对比
  3. 对比实验: f-expert vs h-expert vs f+h 联合

运行: python h_space_expert.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.decomposition import PCA
import time
import os

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')

USE_BF16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}, BF16: {'ON' if USE_BF16 else 'OFF'}")
print("=" * 80)

DL_KWARGS = dict(num_workers=8, pin_memory=True, persistent_workers=True)
CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']


# ═══════════════════════════════════════════════════════════
# 1. 数据 & ETF (不变)
# ═══════════════════════════════════════════════════════════

def dirichlet_split(dataset, n_clients, alpha, n_classes=10):
    targets = np.array(dataset.targets)
    ci = defaultdict(list)
    for idx, l in enumerate(targets): ci[l].append(idx)
    client_idx = defaultdict(list)
    client_cc = defaultdict(lambda: defaultdict(int))
    for c in range(n_classes):
        idxs = np.array(ci[c]); np.random.shuffle(idxs)
        props = np.random.dirichlet([alpha]*n_clients)
        props = (props*len(idxs)).astype(int); props[-1] = len(idxs)-props[:-1].sum()
        s = 0
        for k in range(n_clients):
            e = s+props[k]
            if e > s:
                client_idx[k].extend(idxs[s:e].tolist())
                client_cc[k][c] = props[k]
            s = e
    return dict(client_idx), dict(client_cc)

def prepare_data(n_clients=5, alpha=0.05, n_classes=10):
    tt = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
        transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2), transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
        transforms.RandomErasing(p=0.25, scale=(0.02,0.2)),
    ])
    te = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=tt)
    test_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=te)
    cidx, ccc = dirichlet_split(train_ds, n_clients, alpha, n_classes)
    print(f"\n数据分布 (α={alpha}):")
    for k in range(n_clients):
        counts = [ccc[k].get(c,0) for c in range(n_classes)]
        top = sorted(range(n_classes), key=lambda c: counts[c], reverse=True)[:3]
        print(f"  Client {k}: {sum(1 for c in counts if c>0)} cls, {sum(counts)} samp, "
              f"top: {', '.join(f'c{c}={counts[c]}' for c in top)}")
    targets = np.array(train_ds.targets)
    cal = {}
    for k in range(n_clients):
        cal[k] = DataLoader(Subset(train_ds, cidx[k]), batch_size=128, shuffle=True, drop_last=True, **DL_KWARGS)
    ccl = {}
    for k in range(n_clients):
        ccl[k] = {}
        cm = defaultdict(list)
        for idx in cidx[k]: cm[targets[idx]].append(idx)
        for c, idxs in cm.items():
            dl_kw = dict(num_workers=4, pin_memory=True, persistent_workers=len(idxs) >= 64)
            ccl[k][c] = DataLoader(Subset(train_ds, idxs), batch_size=64, shuffle=True, drop_last=False, **dl_kw)
    tl = DataLoader(test_ds, batch_size=256, shuffle=False, **DL_KWARGS)
    return cal, ccl, tl, ccc

def generate_etf(nc, fd, seed=42):
    rng = torch.Generator(); rng.manual_seed(seed)
    M = np.sqrt(nc/(nc-1)) * (torch.eye(nc) - torch.ones(nc,nc)/nc)
    if fd > nc:
        Q, _ = torch.linalg.qr(torch.randn(fd, nc, generator=rng))
        M = M @ Q.T
    print(f"  ETF: norm={torch.norm(M, dim=1).mean():.4f}")
    return M


# ═══════════════════════════════════════════════════════════
# 2. ★ 更新的 Backbone: 同时返回 h 和 f
# ═══════════════════════════════════════════════════════════

class Backbone(nn.Module):
    def __init__(self, fd=256, channels=None):
        super().__init__()
        if channels is None: channels = [64, 128, 256, 256]
        c1, c2, c3, c4 = channels; self.channels = channels
        self.features = nn.Sequential(
            nn.Conv2d(3,c1,3,padding=1), nn.BatchNorm2d(c1), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(c1,c2,3,padding=1), nn.BatchNorm2d(c2), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(c2,c3,3,padding=1), nn.BatchNorm2d(c3), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(c3,c4,3,padding=1), nn.BatchNorm2d(c4), nn.ReLU(True), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(c4*2*2, fd)

    def forward(self, x):
        x = self.features(x); x = x.view(x.size(0), -1)
        return F.normalize(self.fc(x), dim=1)

    def forward_with_h(self, x):
        """同时返回 f (256, 归一化) 和 h (conv输出, 展平, 1024维)"""
        h_spatial = self.features(x)                       # (B, C4, 2, 2)
        h_flat = h_spatial.view(h_spatial.size(0), -1)     # (B, 1024)
        f = F.normalize(self.fc(h_flat), dim=1)            # (B, 256)
        return f, h_flat, h_spatial


# ═══════════════════════════════════════════════════════════
# 3. ★ 原始 f-space Expert (保留对比用)
# ═══════════════════════════════════════════════════════════

class FSpaceExpert(nn.Module):
    """原始: 在 f (256维, 单位球) 上操作"""
    def __init__(self, fd=256, ed=256, hd=128, ld=32):
        super().__init__()
        self.enc1 = nn.Linear(fd+ed, hd); self.ebn = nn.LayerNorm(hd)
        self.enc2 = nn.Linear(hd, ld)
        self.dec1 = nn.Linear(ld+ed, hd); self.dbn = nn.LayerNorm(hd)
        self.dec2 = nn.Linear(hd, fd)

    def encode(self, f, c):
        return self.enc2(F.relu(self.ebn(self.enc1(torch.cat([f, c], 1)))))

    def decode(self, z, c):
        return self.dec2(F.relu(self.dbn(self.dec1(torch.cat([z, c], 1)))))

    def forward(self, f, c):
        z = self.encode(f, c)
        return self.decode(z, c), z


# ═══════════════════════════════════════════════════════════
# 4. ★★★ 新: h-space Expert (FiLM-Conditioned)
# ═══════════════════════════════════════════════════════════

class HSpaceExpert(nn.Module):
    """
    在 h 空间 (conv输出, 256×2×2) 上操作
    
    架构设计:
      - FiLM conditioning: ETF[c] → 生成 per-channel γ,β 调制 h
      - 编码: Conv(256→128, 1×1) → flatten → Linear(512→latent)
      - 解码: Linear(latent→512) → reshape → Conv(128→256, 1×1)
      - 残差连接: h_rec = decoder_output + residual_gate * h_input
    
    为什么用 FiLM 而非 concat:
      - h 是 (C,2,2) 空间结构, concat ETF 会破坏空间维度
      - FiLM 是 channel-wise 调制, 保留空间结构, 参数高效
    """
    def __init__(self, c4=256, spatial=4, ed=256, ld=64):
        super().__init__()
        # spatial = 2*2 = 4 (展平后每通道的空间元素数)
        self.c4 = c4
        self.spatial = spatial
        h_flat = c4 * spatial  # 1024

        # FiLM: ETF condition → per-channel modulation
        self.film = nn.Sequential(
            nn.Linear(ed, 256), nn.ReLU(True),
            nn.Linear(256, c4 * 2)  # γ 和 β, 各 c4 维
        )

        # Encoder: h_flat → bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(h_flat, 512), nn.LayerNorm(512), nn.ReLU(True),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(True),
            nn.Linear(256, ld)
        )

        # Decoder: bottleneck + condition → h_flat
        self.decoder = nn.Sequential(
            nn.Linear(ld + ed, 256), nn.LayerNorm(256), nn.ReLU(True),
            nn.Linear(256, 512), nn.LayerNorm(512), nn.ReLU(True),
            nn.Linear(512, h_flat)
        )

        # 残差门控: 学习多少保留原始 vs 重构
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, h_flat, cond):
        """
        h_flat: (B, 1024) - conv层展平输出
        cond:   (B, 256)  - ETF[c] 条件向量
        returns: h_rec (B, 1024), z (B, ld)
        """
        B = h_flat.size(0)

        # FiLM modulation
        film_params = self.film(cond)  # (B, c4*2)
        gamma = film_params[:, :self.c4]   # (B, 256)
        beta = film_params[:, self.c4:]    # (B, 256)

        # reshape h to (B, C4, spatial) for channel-wise modulation
        h_reshaped = h_flat.view(B, self.c4, self.spatial)  # (B, 256, 4)
        h_mod = gamma.unsqueeze(2) * h_reshaped + beta.unsqueeze(2)  # (B, 256, 4)
        h_mod_flat = h_mod.view(B, -1)  # (B, 1024)

        # Encode
        z = self.encoder(h_mod_flat)  # (B, ld)

        # Decode (condition-aware)
        h_rec = self.decoder(torch.cat([z, cond], 1))  # (B, 1024)

        # Residual gate
        gate = torch.sigmoid(self.gate)
        h_rec = h_rec + gate * h_flat

        return h_rec, z


# ═══════════════════════════════════════════════════════════
# 5. ★★★ 新: f+h 联合 Expert
# ═══════════════════════════════════════════════════════════

class JointExpert(nn.Module):
    """
    同时在 f 和 h 空间操作, 联合误差
    误差 = α * ||f - f_rec||² + (1-α) * ||h - h_rec||²
    """
    def __init__(self, fd=256, c4=256, spatial=4, ed=256, ld=64):
        super().__init__()
        h_flat_dim = c4 * spatial  # 1024

        # 共享 FiLM
        self.film = nn.Sequential(
            nn.Linear(ed, 256), nn.ReLU(True),
            nn.Linear(256, c4 * 2)
        )
        self.c4 = c4; self.spatial = spatial

        # h-branch encoder
        self.h_enc = nn.Sequential(
            nn.Linear(h_flat_dim, 512), nn.LayerNorm(512), nn.ReLU(True),
            nn.Linear(512, ld)
        )
        # f-branch encoder
        self.f_enc = nn.Sequential(
            nn.Linear(fd + ed, 128), nn.LayerNorm(128), nn.ReLU(True),
            nn.Linear(128, ld)
        )
        # 联合 decoder
        self.h_dec = nn.Sequential(
            nn.Linear(ld * 2 + ed, 512), nn.LayerNorm(512), nn.ReLU(True),
            nn.Linear(512, h_flat_dim)
        )
        self.f_dec = nn.Sequential(
            nn.Linear(ld * 2 + ed, 128), nn.LayerNorm(128), nn.ReLU(True),
            nn.Linear(128, fd)
        )

    def forward(self, f, h_flat, cond):
        B = f.size(0)
        # FiLM on h
        fp = self.film(cond)
        gamma = fp[:, :self.c4]; beta = fp[:, self.c4:]
        h_r = h_flat.view(B, self.c4, self.spatial)
        h_mod = (gamma.unsqueeze(2) * h_r + beta.unsqueeze(2)).view(B, -1)

        z_h = self.h_enc(h_mod)
        z_f = self.f_enc(torch.cat([f, cond], 1))
        z = torch.cat([z_h, z_f], 1)  # (B, 2*ld)

        h_rec = self.h_dec(torch.cat([z, cond], 1))
        f_rec = self.f_dec(torch.cat([z, cond], 1))
        return f_rec, h_rec, z


# ═══════════════════════════════════════════════════════════
# 6. 训练函数
# ═══════════════════════════════════════════════════════════

def etf_cl(features, labels, etf, temp=0.1):
    features = F.normalize(features, dim=1); bs = features.size(0)
    lproto = F.cross_entropy(torch.mm(features, etf.T)/temp, labels)
    lsamp = torch.tensor(0.0, device=features.device)
    if bs > 1:
        sm = torch.eye(bs, device=features.device, dtype=torch.bool); ns = ~sm
        sim = torch.mm(features, features.T) / temp
        mp = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() * ns.float()
        pc = mp.sum(1); v = pc > 0
        if v.sum() > 0:
            ss = sim - sim.max(1, keepdim=True)[0].detach()
            es = torch.exp(ss) * ns.float()
            lp_ = ss - torch.log(es.sum(1) + 1e-8).unsqueeze(1)
            lsamp = -(mp * lp_).sum(1)[v] / (pc[v] + 1e-8)
            lsamp = lsamp.mean()
    return lproto + 0.5 * lsamp

def etf_al(features, labels, etf):
    features = F.normalize(features, dim=1)
    return (1 - (features * etf[labels]).sum(1)).mean()

def train_bb(bb, loader, classes, etf, epochs=600, lr=1e-3):
    bb = bb.to(device); ed = etf.to(device); ncl = len(classes); bb.train()
    opt = torch.optim.Adam(bb.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    amp = (torch.amp.autocast('cuda', dtype=torch.bfloat16) if USE_BF16
           else torch.amp.autocast('cuda', enabled=False))
    for ep in range(epochs):
        el = 0; nb = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            with amp:
                f = bb(x)
                if ncl >= 2: loss = etf_cl(f, y, ed) + 0.5 * etf_al(f, y, ed)
                else: loss = etf_al(f, y, ed)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            el += loss.item(); nb += 1
        sch.step()
        if (ep+1) % 100 == 0 or ep == 0:
            print(f"      BB {ep+1}/{epochs} loss={el/max(nb,1):.4f}")
    return bb


def preextract_h(bb, dl):
    """预提取 f 和 h_flat"""
    bb.eval(); all_f = []; all_h = []
    with torch.no_grad():
        amp = (torch.amp.autocast('cuda', dtype=torch.bfloat16) if USE_BF16
               else torch.amp.autocast('cuda', enabled=False))
        with amp:
            for x, _ in dl:
                f, h_flat, _ = bb.forward_with_h(x.to(device, non_blocking=True))
                all_f.append(f.float())
                all_h.append(h_flat.float())
    return torch.cat(all_f, 0), torch.cat(all_h, 0)


# ── f-space expert 训练 (原始) ──
def train_f_expert(exp, cached_f, eo, ed, others, fdim=256, epochs=600, lr=1e-3, margin=0.05):
    exp = exp.to(device); N = cached_f.size(0); no = others.size(0)
    opt = torch.optim.Adam(exp.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bs = min(64, N); nn_ = 64
    for ep in range(epochs):
        exp.train(); perm = torch.randperm(N, device=device)
        for i in range(0, N, bs):
            idx = perm[i:i+bs]; fp = cached_f[idx]; B = fp.size(0)
            co = eo.unsqueeze(0).expand(B, -1)
            fr1, _ = exp(fp, co); l1 = F.mse_loss(fr1, fp)
            nc = others[torch.randint(0, no, (nn_,), device=device)]
            sc = 0.05 + 0.25 * torch.rand(nn_, 1, device=device)
            ff = F.normalize(ed[nc] + torch.randn(nn_, fdim, device=device) * sc, dim=1)
            fc = others[torch.randint(0, no, (B,), device=device)]
            fr2, _ = exp(fp, ed[fc]); l2 = F.relu(margin - ((fp-fr2)**2).mean(1)).mean()
            fr3, _ = exp(ff, ed[nc]); l3 = F.relu(margin - ((ff-fr3)**2).mean(1)).mean()
            fr4, _ = exp(ff, eo.unsqueeze(0).expand(nn_,-1))
            l4 = F.relu(margin - ((ff-fr4)**2).mean(1)).mean()
            loss = l1 + (l2 + l3 + l4)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sch.step()
    exp.eval(); return exp


# ── ★ h-space expert 训练 ──
def train_h_expert(exp, cached_h, eo, ed, others, hdim=1024, epochs=600, lr=1e-3, margin=0.05):
    """
    与 f-expert 对称的训练, 但在 h 空间
    负样本: 在 h 空间构造 (不再是 ETF 附近的噪声, 而是随机高维向量)
    """
    exp = exp.to(device); N = cached_h.size(0); no = others.size(0)
    opt = torch.optim.Adam(exp.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bs = min(64, N); nn_ = 64

    # 计算训练数据的统计量, 用于构造合理的负样本
    h_mean = cached_h.mean(0)
    h_std = cached_h.std(0).clamp(min=1e-6)

    for ep in range(epochs):
        exp.train(); perm = torch.randperm(N, device=device)
        for i in range(0, N, bs):
            idx = perm[i:i+bs]; hp = cached_h[idx]; B = hp.size(0)
            co = eo.unsqueeze(0).expand(B, -1)

            # L1: 重构自己类的 h
            hr1, _ = exp(hp, co); l1 = F.mse_loss(hr1, hp)

            # L2: 给正确 h + 错误条件 → 应该重构失败
            fc = others[torch.randint(0, no, (B,), device=device)]
            hr2, _ = exp(hp, ed[fc])
            l2 = F.relu(margin - ((hp - hr2)**2).mean(1)).mean()

            # L3: 随机 h (模拟其他类的 conv 输出) + 错误条件
            fake_h = h_mean.unsqueeze(0) + h_std.unsqueeze(0) * torch.randn(nn_, hdim, device=device)
            nc = others[torch.randint(0, no, (nn_,), device=device)]
            hr3, _ = exp(fake_h, ed[nc])
            l3 = F.relu(margin - ((fake_h - hr3)**2).mean(1)).mean()

            # L4: 随机 h + 正确条件 → 也应重构失败
            hr4, _ = exp(fake_h, eo.unsqueeze(0).expand(nn_, -1))
            l4 = F.relu(margin - ((fake_h - hr4)**2).mean(1)).mean()

            loss = l1 + (l2 + l3 + l4)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sch.step()
    exp.eval(); return exp


# ── ★ Joint expert 训练 ──
def train_joint_expert(exp, cached_f, cached_h, eo, ed, others,
                       fdim=256, hdim=1024, epochs=600, lr=1e-3, margin=0.05, alpha_fh=0.3):
    exp = exp.to(device); N = cached_f.size(0); no = others.size(0)
    opt = torch.optim.Adam(exp.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bs = min(64, N); nn_ = 64
    h_mean = cached_h.mean(0); h_std = cached_h.std(0).clamp(min=1e-6)

    for ep in range(epochs):
        exp.train(); perm = torch.randperm(N, device=device)
        for i in range(0, N, bs):
            idx = perm[i:i+bs]; fp = cached_f[idx]; hp = cached_h[idx]; B = fp.size(0)
            co = eo.unsqueeze(0).expand(B, -1)

            # 正样本重构
            f_rec, h_rec, _ = exp(fp, hp, co)
            l1 = alpha_fh * F.mse_loss(f_rec, fp) + (1-alpha_fh) * F.mse_loss(h_rec, hp)

            # 错误条件
            fc = others[torch.randint(0, no, (B,), device=device)]
            f_rec2, h_rec2, _ = exp(fp, hp, ed[fc])
            err_f = ((fp - f_rec2)**2).mean(1)
            err_h = ((hp - h_rec2)**2).mean(1)
            l2 = F.relu(margin - (alpha_fh * err_f + (1-alpha_fh) * err_h)).mean()

            # 假 h + 正确条件
            fake_h = h_mean.unsqueeze(0) + h_std.unsqueeze(0) * torch.randn(nn_, hdim, device=device)
            fake_f = F.normalize(ed[others[torch.randint(0,no,(nn_,),device=device)]]
                                 + torch.randn(nn_, fdim, device=device)*0.15, dim=1)
            f_rec3, h_rec3, _ = exp(fake_f, fake_h, co.expand(nn_,-1) if nn_==B else eo.unsqueeze(0).expand(nn_,-1))
            err_f3 = ((fake_f - f_rec3)**2).mean(1)
            err_h3 = ((fake_h - h_rec3)**2).mean(1)
            l3 = F.relu(margin - (alpha_fh * err_f3 + (1-alpha_fh) * err_h3)).mean()

            loss = l1 + l2 + l3
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sch.step()
    exp.eval(); return exp


def train_all_experts(bb, cls_loaders, classes, etf, nc=10, fdim=256, epochs=600, lr=1e-3):
    """训练三种 expert: f-space, h-space, joint"""
    bb.eval(); ed = etf.to(device)
    om = {c: torch.tensor([k for k in range(nc) if k != c], device=device) for c in range(nc)}

    f_exps = {}; h_exps = {}; j_exps = {}

    for cls in classes:
        print(f"      Class {cls}...", flush=True); t0 = time.time()

        # 预提取 f 和 h
        cached_f, cached_h = preextract_h(bb, cls_loaders[cls])
        hdim = cached_h.size(1)

        # f-space expert (原始)
        f_exp = FSpaceExpert(fdim, fdim, 128, 32).to(device)
        f_exp = train_f_expert(f_exp, cached_f, ed[cls], ed, om[cls], fdim, epochs, lr)
        f_exps[cls] = f_exp

        # h-space expert (新)
        h_exp = HSpaceExpert(c4=256, spatial=4, ed=fdim, ld=64).to(device)
        h_exp = train_h_expert(h_exp, cached_h, ed[cls], ed, om[cls], hdim, epochs, lr)
        h_exps[cls] = h_exp

        # Joint expert
        j_exp = JointExpert(fdim, 256, 4, fdim, 64).to(device)
        j_exp = train_joint_expert(j_exp, cached_f, cached_h, ed[cls], ed, om[cls],
                                   fdim, hdim, epochs, lr)
        j_exps[cls] = j_exp

        # 报告
        with torch.no_grad():
            ne = min(256, cached_f.size(0))
            fr, _ = f_exp(cached_f[:ne], ed[cls].unsqueeze(0).expand(ne,-1))
            f_mse = ((cached_f[:ne] - fr)**2).mean().item()
            hr, _ = h_exp(cached_h[:ne], ed[cls].unsqueeze(0).expand(ne,-1))
            h_mse = ((cached_h[:ne] - hr)**2).mean().item()
            print(f"        done ({time.time()-t0:.1f}s)  f_MSE={f_mse:.6f}  h_MSE={h_mse:.6f}")

    return f_exps, h_exps, j_exps


# ═══════════════════════════════════════════════════════════
# 7. ★★★ 可视化: h-space vs f-space 对 seen/unseen 的分布
# ═══════════════════════════════════════════════════════════

def visualize_h_vs_f(bbs, tl, ccc, etf, save_path, n_samples=2000):
    """
    对每个 client, 画 f-space 和 h-space 的 PCA 投影
    关键: 看 unseen 类样本在两个空间中的分布位置是否不同
    """
    ed = etf.to(device)
    K = len(bbs)

    # 收集测试集的一个子集
    all_x = []; all_y = []
    n = 0
    for x, y in tl:
        all_x.append(x); all_y.append(y)
        n += x.size(0)
        if n >= n_samples: break
    X = torch.cat(all_x, 0)[:n_samples]
    Y = torch.cat(all_y, 0)[:n_samples].numpy()

    # 选择2个有代表性的client (已见类最少的)
    client_ncls = [(k, sum(1 for c in range(10) if ccc[k].get(c, 0) > 0)) for k in range(K)]
    client_ncls.sort(key=lambda x: x[1])
    selected = [client_ncls[0][0], client_ncls[min(1, K-1)][0]]

    fig, axes = plt.subplots(len(selected), 2, figsize=(20, 8*len(selected)))
    if len(selected) == 1:
        axes = axes.reshape(1, -1)

    colors = plt.cm.tab10(np.arange(10))

    for row, k in enumerate(selected):
        bb = bbs[k]; bb.eval()
        seen_classes = set(c for c in range(10) if ccc[k].get(c, 0) > 100)

        with torch.no_grad():
            f_all, h_all, _ = bb.forward_with_h(X.to(device))
            f_np = f_all.cpu().numpy()
            h_np = h_all.cpu().numpy()

        # PCA
        pca_f = PCA(n_components=2).fit_transform(f_np)
        pca_h = PCA(n_components=2).fit_transform(h_np)

        # ── f-space ──
        ax = axes[row, 0]
        for c in range(10):
            mask = (Y == c)
            if not mask.any(): continue
            marker = 'o' if c in seen_classes else 'x'
            size = 15 if c in seen_classes else 25
            alpha = 0.6 if c in seen_classes else 0.8
            lw = 0 if c in seen_classes else 1.5
            ax.scatter(pca_f[mask, 0], pca_f[mask, 1], c=[colors[c]],
                      marker=marker, s=size, alpha=alpha, linewidths=lw,
                      label=f"{'★' if c in seen_classes else '✗'} {CIFAR10_CLASSES[c]}")
        # ETF directions in PCA space
        etf_f = ed.cpu().numpy()
        # etf_pca = PCA(n_components=2).fit(f_np).transform(etf_f)
        etf_pca = PCA(n_components=2).fit(f_np).transform(etf_f)

        for c in range(10):
            ax.annotate(f'{c}', etf_pca[c], fontsize=8, fontweight='bold',
                       color=colors[c], ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        ax.set_title(f'Client {k} — f-space (after FC+norm)\n'
                     f'Seen: {sorted(seen_classes)}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, ncol=2, loc='upper right')
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2')

        # ── h-space ──
        ax = axes[row, 1]
        for c in range(10):
            mask = (Y == c)
            if not mask.any(): continue
            marker = 'o' if c in seen_classes else 'x'
            size = 15 if c in seen_classes else 25
            alpha = 0.6 if c in seen_classes else 0.8
            lw = 0 if c in seen_classes else 1.5
            ax.scatter(pca_h[mask, 0], pca_h[mask, 1], c=[colors[c]],
                      marker=marker, s=size, alpha=alpha, linewidths=lw,
                      label=f"{'★' if c in seen_classes else '✗'} {CIFAR10_CLASSES[c]}")
        ax.set_title(f'Client {k} — h-space (before FC, 1024-dim)\n'
                     f'Seen: {sorted(seen_classes)}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, ncol=2, loc='upper right')
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2')

    plt.suptitle('f-space vs h-space: Unseen class (✗) distribution\n'
                 'Key question: Are unseen classes separable from seen classes in h-space?',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  可视化保存: {save_path}")


def visualize_error_distribution(bbs, f_exps_all, h_exps_all, tl, etf, ccc, save_path, n_samples=2000):
    """
    核心诊断图: 对每个 (client, class) 组合,
    比较 f-expert 和 h-expert 对 seen vs unseen 样本的误差分布
    """
    ed = etf.to(device); K = len(bbs)

    # 收集子集
    all_x = []; all_y = []; n = 0
    for x, y in tl:
        all_x.append(x); all_y.append(y); n += x.size(0)
        if n >= n_samples: break
    X = torch.cat(all_x, 0)[:n_samples].to(device)
    Y = torch.cat(all_y, 0)[:n_samples].numpy()

    # 选一个已见类少的 client
    k = min(range(K), key=lambda k: sum(1 for c in range(10) if ccc[k].get(c, 0) > 100))
    seen = sorted(c for c in range(10) if ccc[k].get(c, 0) > 100)
    unseen = sorted(c for c in range(10) if c not in seen)
    print(f"  诊断 Client {k}: seen={seen}, unseen={unseen}")

    bb = bbs[k]; bb.eval()
    with torch.no_grad():
        f_all, h_all, _ = bb.forward_with_h(X)

    # 对每个 seen class 的 expert, 看它在各个类上的误差
    if not seen:
        print("  No seen classes with >100 samples, skip"); return

    fig, axes = plt.subplots(len(seen), 2, figsize=(16, 4*len(seen)))
    if len(seen) == 1: axes = axes.reshape(1, -1)

    for row, c_exp in enumerate(seen):
        f_exp = f_exps_all[k][c_exp]
        h_exp = h_exps_all[k][c_exp]

        with torch.no_grad():
            cond = ed[c_exp].unsqueeze(0).expand(f_all.size(0), -1)
            f_rec, _ = f_exp(f_all, cond)
            f_err = ((f_all - f_rec)**2).mean(1).cpu().numpy()
            h_rec, _ = h_exp(h_all, cond)
            h_err = ((h_all - h_rec)**2).mean(1).cpu().numpy()

        # ── f-space errors ──
        ax = axes[row, 0]
        # 该 expert 对各个真实类的误差分布
        for c_true in range(10):
            mask = (Y == c_true)
            if not mask.any(): continue
            errs = f_err[mask]
            is_seen = c_true in seen
            is_self = c_true == c_exp
            color = 'green' if is_self else ('blue' if is_seen else 'red')
            label = f'c{c_true}{"★" if is_self else "○" if is_seen else "✗"}'
            ax.hist(errs, bins=30, alpha=0.4, color=color, label=label, density=True)
            ax.axvline(errs.mean(), color=color, linestyle='--', alpha=0.6)
        ax.set_title(f'f-expert({c_exp}) — error dist by true class', fontsize=10)
        ax.legend(fontsize=6, ncol=3); ax.set_xlabel('MSE'); ax.set_xlim(left=0)

        # ── h-space errors ──
        ax = axes[row, 1]
        for c_true in range(10):
            mask = (Y == c_true)
            if not mask.any(): continue
            errs = h_err[mask]
            is_seen = c_true in seen
            is_self = c_true == c_exp
            color = 'green' if is_self else ('blue' if is_seen else 'red')
            label = f'c{c_true}{"★" if is_self else "○" if is_seen else "✗"}'
            ax.hist(errs, bins=30, alpha=0.4, color=color, label=label, density=True)
            ax.axvline(errs.mean(), color=color, linestyle='--', alpha=0.6)
        ax.set_title(f'h-expert({c_exp}) — error dist by true class', fontsize=10)
        ax.legend(fontsize=6, ncol=3); ax.set_xlabel('MSE'); ax.set_xlim(left=0)

    plt.suptitle(f'Client {k} | Expert error: f-space vs h-space\n'
                 f'Green=self class, Blue=other seen, Red=unseen\n'
                 f'Key: h-expert should push red (unseen) errors higher than f-expert does',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  误差分布图保存: {save_path}")


# ═══════════════════════════════════════════════════════════
# 8. 推理评估
# ═══════════════════════════════════════════════════════════

def evaluate_experts(bbs, f_exps_all, h_exps_all, j_exps_all, tl, etf, ccc, nc=10):
    """对比三种 expert 的推理准确率"""
    K = len(bbs); ed = etf.to(device)

    all_f_err = []; all_h_err = []; all_j_err = []; all_labels = []

    with torch.no_grad():
        for x, y in tl:
            x_dev = x.to(device, non_blocking=True); bs = x.size(0)

            batch_f_err = torch.full((K, bs, nc), float('inf'))
            batch_h_err = torch.full((K, bs, nc), float('inf'))
            batch_j_err = torch.full((K, bs, nc), float('inf'))

            for k in range(K):
                f_k, h_k, _ = bbs[k].forward_with_h(x_dev)

                for c in f_exps_all[k]:
                    cond = ed[c].unsqueeze(0).expand(bs, -1)

                    # f-space
                    fr, _ = f_exps_all[k][c](f_k, cond)
                    batch_f_err[k, :, c] = ((f_k - fr)**2).mean(1).cpu()

                    # h-space
                    hr, _ = h_exps_all[k][c](h_k, cond)
                    batch_h_err[k, :, c] = ((h_k - hr)**2).mean(1).cpu()

                    # joint
                    f_rec, h_rec, _ = j_exps_all[k][c](f_k, h_k, cond)
                    joint_err = 0.3 * ((f_k - f_rec)**2).mean(1) + 0.7 * ((h_k - h_rec)**2).mean(1)
                    batch_j_err[k, :, c] = joint_err.cpu()

            all_f_err.append(batch_f_err)
            all_h_err.append(batch_h_err)
            all_j_err.append(batch_j_err)
            all_labels.append(y)

    f_errors = torch.cat(all_f_err, dim=1)
    h_errors = torch.cat(all_h_err, dim=1)
    j_errors = torch.cat(all_j_err, dim=1)
    labels = torch.cat(all_labels).numpy()
    N = len(labels)

    def _eval(errors, name):
        errors[errors == float('inf')] = 1e6
        best_err, _ = errors.min(dim=0)
        preds = best_err.argmin(1).numpy()
        acc = (preds == labels).mean()
        print(f"    {name:25s}: {acc:.2%}")
        return acc

    print(f"\n  === Expert 对比 (min_k argmin_c) ===")
    f_acc = _eval(f_errors.clone(), "f-space (original)")
    h_acc = _eval(h_errors.clone(), "h-space (new)")
    j_acc = _eval(j_errors.clone(), "joint f+h")

    # 按类别分析
    print(f"\n  === 按类别准确率 ===")
    print(f"    {'Class':<12s} {'f-space':>10s} {'h-space':>10s} {'joint':>10s} {'Δ(h-f)':>10s}")
    for c in range(nc):
        mask = (labels == c)
        f_e = f_errors.clone(); f_e[f_e==float('inf')] = 1e6
        h_e = h_errors.clone(); h_e[h_e==float('inf')] = 1e6
        j_e = j_errors.clone(); j_e[j_e==float('inf')] = 1e6
        f_pred = f_e.min(0)[0].argmin(1).numpy()
        h_pred = h_e.min(0)[0].argmin(1).numpy()
        j_pred = j_e.min(0)[0].argmin(1).numpy()
        fa = (f_pred[mask] == c).mean()
        ha = (h_pred[mask] == c).mean()
        ja = (j_pred[mask] == c).mean()
        delta = ha - fa
        marker = " ★" if delta > 0.02 else (" ✗" if delta < -0.02 else "")
        print(f"    {CIFAR10_CLASSES[c]:<12s} {fa:>10.2%} {ha:>10.2%} {ja:>10.2%} {delta:>+10.2%}{marker}")

    return f_acc, h_acc, j_acc


# ═══════════════════════════════════════════════════════════
# 9. Union (不变, 简化版)
# ═══════════════════════════════════════════════════════════

def union_aggregate(bbs, fd=256, thr=0.95):
    K = len(bbs); print(f"\n  [Union] {K} backbones, thr={thr}")
    ci = [0,4,8,12]; bi = [1,5,9,13]
    lp_list = []; alm = []; pm = None; pn = 3
    for li in range(4):
        af = []; ass = []; ab = []; abn = []
        for k, bb in enumerate(bbs):
            conv = bb.features[ci[li]]; bn = bb.features[bi[li]]
            w = conv.weight.data.cpu(); b_ = conv.bias.data.cpu(); Co, Ci = w.size(0), w.size(1)
            if li == 0: wr = w
            else:
                wr = torch.zeros(Co, pn, 3, 3)
                for l in range(Ci):
                    if l in pm[k]: wr[:, pm[k][l], :, :] = w[:, l, :, :]
            for i in range(Co):
                af.append(wr[i]); ass.append((k, i)); ab.append(b_[i])
                abn.append({c_: bn.__getattr__({'w':'weight','b':'bias','m':'running_mean','v':'running_var'}[c_]).data.cpu()[i] for c_ in 'wbmv'})
        if not af: continue
        st = torch.stack(af); N_ = st.size(0)
        sf = F.normalize(st.view(N_, -1).float(), dim=1); sim = sf @ sf.T
        assigned = [False]*N_; gf = []; fm = []
        norms = st.view(N_, -1).float().norm(dim=1)
        order = norms.argsort(descending=True).reshape(-1).tolist()
        for seed in order:
            if assigned[seed]: continue
            cluster = [seed]; assigned[seed] = True
            for j in order:
                if assigned[j]: continue
                if all(sim[j, c_] > thr for c_ in cluster): cluster.append(j); assigned[j] = True
            cf = st[cluster]; cn = norms[cluster]; ww = cn / (cn.sum() + 1e-8)
            gf.append((cf.float() * ww.view(-1, *([1]*(cf.dim()-1)))).sum(0))
            fm.append([ass[i] for i in cluster])
        No = len(gf)
        mb = []; mbn = {'w':[],'b':[],'m':[],'v':[]}
        for g, grp in enumerate(fm):
            idxs_ = [sum(bbs[kk].features[ci[li]].weight.size(0) for kk in range(ck))+ci_ for ck, ci_ in grp]
            mb.append(torch.stack([ab[i] for i in idxs_]).mean())
            for c_ in 'wbmv': mbn[c_].append(torch.stack([abn[i][c_] for i in idxs_]).mean())
        nm = [{} for _ in range(K)]
        for g, grp in enumerate(fm):
            for ck, ci_ in grp: nm[ck][ci_] = g
        oc = bbs[0].features[ci[li]].weight.size(0)
        print(f"    Conv{li+1}: {pn}→{No} (from {K*oc})")
        lp_list.append({'Ni':pn,'No':No,'f':torch.stack(gf),'b':torch.stack(mb),
                   'bw':torch.stack(mbn['w']),'bb':torch.stack(mbn['b']),
                   'bm':torch.stack(mbn['m']),'bv':torch.stack(mbn['v'])})
        alm.append(nm); pm = nm; pn = No
    Nf = pn; fi = Nf*4; mfw = torch.zeros(fd, fi); mfb = torch.zeros(fd); fcc = torch.zeros(fi)
    for k, bb in enumerate(bbs):
        fw = bb.fc.weight.data.cpu(); fb = bb.fc.bias.data.cpu(); c4 = bb.channels[3]
        for l in range(c4):
            if l not in pm[k]: continue
            g = pm[k][l]; mfw[:, g*4:(g+1)*4] += fw[:, l*4:(l+1)*4]; fcc[g*4:(g+1)*4] += 1
        mfb += fb/K
    fcc = fcc.clamp(min=1); mfw /= fcc.unsqueeze(0)
    chs = [l['No'] for l in lp_list]
    merged = Backbone(fd, chs)
    with torch.no_grad():
        for li_ in range(4):
            l = lp_list[li_]; Ni = l['Ni']; No = l['No']
            merged.features[ci[li_]] = nn.Conv2d(Ni, No, 3, padding=1)
            merged.features[ci[li_]].weight.copy_(l['f'][:,:Ni]); merged.features[ci[li_]].bias.copy_(l['b'])
            merged.features[bi[li_]] = nn.BatchNorm2d(No)
            merged.features[bi[li_]].weight.copy_(l['bw']); merged.features[bi[li_]].bias.copy_(l['bb'])
            merged.features[bi[li_]].running_mean.copy_(l['bm']); merged.features[bi[li_]].running_var.copy_(l['bv'])
        merged.fc = nn.Linear(fi, fd); merged.fc.weight.copy_(mfw); merged.fc.bias.copy_(mfb)
    print(f"    合并: {sum(p.numel() for p in merged.parameters()):,} params")
    return merged.to(device)


# ═══════════════════════════════════════════════════════════
# 10. 主实验
# ═══════════════════════════════════════════════════════════

def main(ALPHA=0.05):
    print("\n" + "=" * 80)
    print(f"MoE v16 — h-space Expert 实验 (α={ALPHA})")
    print("=" * 80)

    NC = 5; NL = 10; FD = 256; EPB = 600; EPE = 600

    os.makedirs('outputs', exist_ok=True)
    etf = generate_etf(NL, FD)
    cal, ccl, tl, ccc = prepare_data(NC, ALPHA, NL)

    # ══════════════════════════════════════════
    # Phase 1: 训练 Backbone (不变)
    # ══════════════════════════════════════════
    print(f"\n{'='*60}\n  Phase 1: 训练 Backbone\n{'='*60}")
    bbs = []; t0 = time.time()
    for k in range(NC):
        cls = sorted(ccc[k].keys())
        print(f"\n  Client {k}: {len(cls)} cls, {sum(ccc[k].values())} samp")
        bb = Backbone(FD)
        bb = train_bb(bb, cal[k], cls, etf, EPB)
        bbs.append(bb)
    print(f"\n  Backbone 训练: {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════
    # Phase 2: ★ 可视化 h vs f (关键诊断)
    # ══════════════════════════════════════════
    print(f"\n{'='*60}\n  Phase 2: 可视化 h-space vs f-space\n{'='*60}")
    visualize_h_vs_f(bbs, tl, ccc, etf,
                     f'outputs/h_vs_f_alpha{ALPHA}.png')

    # ══════════════════════════════════════════
    # Phase 3: 训练三种 Expert
    # ══════════════════════════════════════════
    print(f"\n{'='*60}\n  Phase 3: 训练 f/h/joint Expert\n{'='*60}")
    f_exps_all = {}; h_exps_all = {}; j_exps_all = {}
    t0 = time.time()
    for k in range(NC):
        cls = sorted(ccc[k].keys())
        print(f"\n  Client {k}: training experts for {len(cls)} classes")
        f_exps, h_exps, j_exps = train_all_experts(
            bbs[k], ccl[k], cls, etf, NL, FD, EPE)
        f_exps_all[k] = f_exps
        h_exps_all[k] = h_exps
        j_exps_all[k] = j_exps
    print(f"\n  Expert 训练: {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════
    # Phase 4: ★ 误差分布诊断
    # ══════════════════════════════════════════
    print(f"\n{'='*60}\n  Phase 4: 误差分布诊断\n{'='*60}")
    visualize_error_distribution(bbs, f_exps_all, h_exps_all, tl, etf, ccc,
                                 f'outputs/error_dist_alpha{ALPHA}.png')

    # ══════════════════════════════════════════
    # Phase 5: 推理对比
    # ══════════════════════════════════════════
    print(f"\n{'='*60}\n  Phase 5: 推理对比\n{'='*60}")
    f_acc, h_acc, j_acc = evaluate_experts(
        bbs, f_exps_all, h_exps_all, j_exps_all, tl, etf, ccc, NL)

    # Union baseline
    ubb = union_aggregate(bbs, FD, 0.95)
    ubb.eval()
    ed = etf.to(device)
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in tl:
            f = F.normalize(ubb(x.to(device)), dim=1)
            preds = torch.mm(f, ed.T).argmax(1).cpu()
            correct += (preds == y).sum().item(); total += y.size(0)
    u_acc = correct / total

    print(f"\n{'='*60}")
    print(f"  ★ 最终结果 (α={ALPHA})")
    print(f"{'='*60}")
    print(f"  Union:          {u_acc:.2%}")
    print(f"  f-expert:       {f_acc:.2%}")
    print(f"  h-expert:       {h_acc:.2%}  (vs f: {(h_acc-f_acc)*100:+.2f}pp)")
    print(f"  joint (f+h):    {j_acc:.2%}  (vs f: {(j_acc-f_acc)*100:+.2f}pp)")
    print(f"{'='*60}\n")

    return u_acc, f_acc, h_acc, j_acc


if __name__ == "__main__":
    results = {}
    for alpha in [0.05, 0.1, 0.3, 0.5]:
        results[alpha] = main(alpha)

    print("\n" + "=" * 80)
    print("  汇总")
    print("=" * 80)
    print(f"  {'α':<8s} {'Union':>10s} {'f-expert':>10s} {'h-expert':>10s} {'joint':>10s}")
    for alpha, (u, f, h, j) in results.items():
        print(f"  {alpha:<8.2f} {u:>10.2%} {f:>10.2%} {h:>10.2%} {j:>10.2%}")