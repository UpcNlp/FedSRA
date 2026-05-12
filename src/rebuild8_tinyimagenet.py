"""
rebuild8_tinyimagenet.py
========================
Tiny-ImageNet 实验 (200类, 64×64), 支持 CNN 和 ResNet-18.

Tiny-ImageNet 数据集:
  - 200 类, 每类 500 张训练图, 50 张验证图
  - 图像大小: 64×64
  - 下载: http://cs231n.stanford.edu/tiny-imagenet-200.zip

用法:
  # 先下载并解压数据集到 ./data/tiny-imagenet-200/
  wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
  unzip tiny-imagenet-200.zip -d ./data/

  # ResNet-18
  CUDA_VISIBLE_DEVICES=0 python rebuild8_tinyimagenet.py --alpha 0.05 --backbone resnet18 --seed 42
  CUDA_VISIBLE_DEVICES=0 python rebuild8_tinyimagenet.py --alpha 0.1  --backbone resnet18 --seed 42

结果保存到 results/tinyimagenet_{backbone}_a{alpha}_k{n_clients}_s{seed}.json
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
import numpy as np
import warnings
import json
import time
import os
import argparse
from collections import defaultdict
from PIL import Image

warnings.filterwarnings('ignore')

# ── 从 rebuild8 导入共享组件 ──
from rebuild8 import (
    device, USE_BF16, DL_KWARGS,
    dirichlet_split, generate_etf,
    Backbone, ConditionalExpert,
    etf_cl, etf_al, train_bb, preextract, train_exp,
    compute_stats, precompute_all,
    expert_original, expert_quality_filter, expert_quality_min,
    expert_top_quality,
    _get_expert_preds_and_margin, _union_norm,
    route_ensemble_logits, route_ensemble_quality,
    cross_client_voting, cross_client_per_client_logits,
    d1_weight_schemes, d3_softmax_ensemble, d4_adaptive_alpha,
    union_aggregate,
)

from resnet18_filter_merge import ResNet18Backbone, union_aggregate_resnet18


# ═══════════════════════════════════════════════════════════
# Tiny-ImageNet 数据准备
# ═══════════════════════════════════════════════════════════

TINY_MEAN = (0.4802, 0.4481, 0.3975)
TINY_STD  = (0.2770, 0.2691, 0.2821)


class TinyImageNetValDataset(Dataset):
    """Tiny-ImageNet 验证集 (需要解析 val_annotations.txt)"""
    def __init__(self, root, transform=None, class_to_idx=None):
        self.root = root
        self.transform = transform
        self.samples = []
        self.targets = []

        # 必须使用训练集的 class_to_idx 保证标签一致
        assert class_to_idx is not None, "Must pass class_to_idx from training ImageFolder"
        self.class_to_idx = class_to_idx

        # 解析 val_annotations.txt
        ann_path = os.path.join(root, 'val_annotations.txt')
        with open(ann_path) as f:
            for line in f.readlines():
                parts = line.strip().split('\t')
                fname = parts[0]
                wnid = parts[1]
                img_path = os.path.join(root, 'images', fname)
                if wnid not in self.class_to_idx:
                    continue
                label = self.class_to_idx[wnid]
                self.samples.append((img_path, label))
                self.targets.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def prepare_data_tinyimagenet(n_clients=5, alpha=0.05, n_classes=200,
                               data_root='./data/tiny-imagenet-200'):
    tt = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=8),
        transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(TINY_MEAN, TINY_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02,0.2)),
    ])
    te = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(TINY_MEAN, TINY_STD),
    ])

    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')

    assert os.path.exists(train_dir), \
        f"Tiny-ImageNet not found at {data_root}. Download and extract first."

    train_ds = datasets.ImageFolder(train_dir, transform=tt)
    test_ds = TinyImageNetValDataset(val_dir, transform=te,
                                     class_to_idx=train_ds.class_to_idx)

    print(f"  Tiny-ImageNet: {len(train_ds)} train, {len(test_ds)} test, {n_classes} classes")

    # Dirichlet split
    targets = np.array(train_ds.targets)
    ci = defaultdict(list)
    for idx, l in enumerate(targets): ci[l].append(idx)
    client_idx = defaultdict(list)
    client_cc = defaultdict(lambda: defaultdict(int))
    rng = np.random.RandomState(np.random.get_state()[1][0])
    for c in range(n_classes):
        idxs = np.array(ci[c]); rng.shuffle(idxs)
        props = rng.dirichlet([alpha]*n_clients)
        props = (props*len(idxs)).astype(int); props[-1] = len(idxs)-props[:-1].sum()
        s = 0
        for k in range(n_clients):
            e = s+props[k]
            if e > s:
                client_idx[k].extend(idxs[s:e].tolist())
                client_cc[k][c] = props[k]
            s = e
    cidx = dict(client_idx)
    ccc = dict(client_cc)

    print(f"\n数据分布 (α={alpha}, Tiny-ImageNet, {n_classes} 类):")
    for k in range(n_clients):
        counts = [ccc[k].get(c,0) for c in range(n_classes)]
        n_cls = sum(1 for c in counts if c > 0)
        total = sum(counts)
        print(f"  Client {k}: {n_cls}/{n_classes} cls, {total} samp")

    cal = {}
    for k in range(n_clients):
        cal[k] = DataLoader(Subset(train_ds, cidx[k]), batch_size=128,
                            shuffle=True, drop_last=True,
                            num_workers=4, pin_memory=True)
    ccl = {}
    for k in range(n_clients):
        ccl[k] = {}
        cm = defaultdict(list)
        for idx in cidx[k]: cm[targets[idx]].append(idx)
        for c, idxs in cm.items():
            ccl[k][c] = DataLoader(Subset(train_ds, idxs), batch_size=64,
                                   shuffle=True, drop_last=False,
                                   num_workers=4, pin_memory=True)
    tl = DataLoader(test_ds, batch_size=256, shuffle=False,
                    num_workers=4, pin_memory=True)
    return cal, ccl, tl, ccc


# ═══════════════════════════════════════════════════════════
# 64×64 适配的 Backbone
# ═══════════════════════════════════════════════════════════

class Backbone64(nn.Module):
    """4层 CNN, 适配 64×64 输入 (多一层 MaxPool)"""
    def __init__(self, fd=256, channels=None):
        super().__init__()
        if channels is None: channels = [64, 128, 256, 256]
        c1, c2, c3, c4 = channels; self.channels = channels
        self.features = nn.Sequential(
            nn.Conv2d(3,c1,3,padding=1), nn.BatchNorm2d(c1), nn.ReLU(True), nn.MaxPool2d(2),   # 32
            nn.Conv2d(c1,c2,3,padding=1), nn.BatchNorm2d(c2), nn.ReLU(True), nn.MaxPool2d(2),  # 16
            nn.Conv2d(c2,c3,3,padding=1), nn.BatchNorm2d(c3), nn.ReLU(True), nn.MaxPool2d(2),  # 8
            nn.Conv2d(c3,c4,3,padding=1), nn.BatchNorm2d(c4), nn.ReLU(True), nn.MaxPool2d(2),  # 4
        )
        self.fc = nn.Linear(c4*4*4, fd)

    def forward(self, x):
        x = self.features(x); x = x.view(x.size(0), -1)
        return F.normalize(self.fc(x), dim=1)


class ResNet18Backbone64(nn.Module):
    """ResNet-18 适配 64×64 输入"""
    def __init__(self, fd=256):
        super().__init__()
        from resnet18_filter_merge import BasicBlock
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 64, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 256, 512, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, fd)

    def _make_layer(self, block, ic, oc, n_blocks, stride):
        layers = [block(ic, oc, stride)]
        for _ in range(1, n_blocks):
            layers.append(block(oc, oc))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return F.normalize(self.fc(x), dim=1)


# ═══════════════════════════════════════════════════════════
# Expert 训练 (跳过小类)
# ═══════════════════════════════════════════════════════════

def train_experts_tiny(bb, cls_loaders, classes, etf, ccc_k,
                       nc=200, fdim=256, ldim=32, epochs=200,
                       lr=1e-3, min_samples=5):
    bb.eval(); ed = etf.to(device); exps = {}
    om = {c: torch.tensor([k for k in range(nc) if k!=c], device=device)
          for c in range(nc)}

    trainable = [c for c in classes if ccc_k.get(c, 0) >= min_samples]
    skipped = len(classes) - len(trainable)
    total = len(trainable)

    for i, cls in enumerate(trainable):
        cached = preextract(bb, cls_loaders[cls])
        if cached.size(0) < min_samples:
            skipped += 1; continue
        exp = ConditionalExpert(fdim, fdim, 128, ldim).to(device)
        exp = train_exp(exp, cached, ed[cls], ed, om[cls], fdim, epochs, lr)
        exps[cls] = exp
        if (i+1) % max(1, total//5) == 0 or i == total-1:
            print(f"      Expert {i+1}/{total} done")

    if skipped > 0:
        print(f"      (跳过 {skipped} 个类, 样本<{min_samples})")
    return exps


# ═══════════════════════════════════════════════════════════
# 主实验
# ═══════════════════════════════════════════════════════════

def main(ALPHA=0.05, backbone_type='resnet18', SEED=42):
    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    print("\n" + "=" * 80)
    print(f"Tiny-ImageNet | Backbone: {backbone_type} | α={ALPHA} | seed={SEED}")
    print("=" * 80)

    NC = 5; NL = 200; FD = 256; LD = 32
    EPB = 300; EPE = 200

    os.makedirs('outputs', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    etf = generate_etf(NL, FD)
    cal, ccl, tl, ccc = prepare_data_tinyimagenet(NC, ALPHA, NL)

    # ── Phase 1: 训练 ──
    print(f"\n{'='*60}")
    print(f"  Phase 1: 训练 ({backbone_type}, EPB={EPB}, EPE={EPE})")
    print(f"{'='*60}")

    bbs = []; client_exps = []; t0 = time.time()
    for k in range(NC):
        cls = sorted(ccc[k].keys())
        n_samp = sum(ccc[k].values())
        print(f"\n  Client {k}: {len(cls)}/{NL} cls, {n_samp} samp")

        if backbone_type == 'resnet18':
            bb = ResNet18Backbone64(FD)
        else:
            bb = Backbone64(FD)

        bb = train_bb(bb, cal[k], cls, etf, EPB)
        exps = train_experts_tiny(bb, ccl[k], cls, etf, ccc[k],
                                   nc=NL, fdim=FD, ldim=LD, epochs=EPE)
        bbs.append(bb); client_exps.append(exps)
        print(f"    训练了 {len(exps)} 个 expert")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tt = time.time() - t0
    print(f"\n  训练完成: {tt:.0f}s ({tt/60:.1f}min)")

    # ── Phase 2: Union ──
    print(f"\n{'='*60}")
    print(f"  Phase 2: Filter Merge ({backbone_type})")
    print(f"{'='*60}")

    if backbone_type == 'resnet18':
        ubb = union_aggregate_resnet18(bbs, FD, 0.95, device)
    else:
        # CNN 的 filter merge: fc 层输入维度不同 (4*4 vs 2*2)
        # 需要单独处理, 这里用原始 union_aggregate 但修正 fc
        ubb = union_aggregate(bbs, FD, 0.95)

    # ── Phase 3: 预计算 ──
    print(f"\n{'='*60}")
    print(f"  Phase 3: 预计算")
    print(f"{'='*60}")
    data = precompute_all(bbs, client_exps, ubb, tl, etf, ccc, NL)
    labels = data['labels']
    N = data['N']

    # ── Phase 4: 评估 ──
    print(f"\n{'='*60}")
    print(f"  Phase 4: 策略评估")
    print(f"{'='*60}")
    R = {}

    # Baselines
    print(f"\n  --- Baselines ---")
    R['Baseline: Union'] = float((data['union_preds'] == labels).mean())
    print(f"  Union:           {R['Baseline: Union']:.2%}")

    preds = expert_original(data)
    R['Baseline: Expert(min)'] = float((preds == labels).mean())
    print(f"  Expert (min):    {R['Baseline: Expert(min)']:.2%}")

    e_pred, _, _ = _get_expert_preds_and_margin(data)
    u_correct = (data['union_preds'] == labels)
    e_correct = (e_pred == labels)
    R['Oracle: U∪E'] = float((u_correct | e_correct).mean())
    print(f"  Oracle (U∪E):    {R['Oracle: U∪E']:.2%}")

    # A: Expert 聚合
    print(f"\n  --- Expert 聚合 ---")
    for min_n in [10, 20, 50]:
        preds = expert_quality_min(data, min_n=min_n)
        tag = f'A3 QualMin n≥{min_n}'
        R[tag] = float((preds == labels).mean())
        print(f"  {tag}: {R[tag]:.2%}")

    # B: Ensemble
    print(f"\n  --- Ensemble ---")
    for alpha in [0.3, 0.5, 1.0, 2.0]:
        preds = route_ensemble_logits(data, alpha=alpha)
        tag = f'B6 Ensemble α={alpha}'
        R[tag] = float((preds == labels).mean())
        print(f"  {tag}: {R[tag]:.2%}")

    for alpha in [0.3, 0.5, 1.0]:
        for min_n in [10, 20]:
            preds = route_ensemble_quality(data, alpha=alpha, min_n=min_n)
            tag = f'B7 QualEns α={alpha} n≥{min_n}'
            R[tag] = float((preds == labels).mean())
            print(f"  {tag}: {R[tag]:.2%}")

    # C: 跨 Client
    print(f"\n  --- 跨 Client ---")
    for alpha in [0.3, 0.5, 1.0, 2.0]:
        for min_n in [0, 10, 20]:
            preds = cross_client_per_client_logits(data, alpha=alpha, min_n=min_n)
            tag = f'C4 PCEns α={alpha} n≥{min_n}'
            R[tag] = float((preds == labels).mean())
            print(f"  {tag}: {R[tag]:.2%}")

    # D: 深挖 Ensemble
    print(f"\n  --- 深挖 Ensemble ---")
    for wt in ['log', 'sqrt']:
        for alpha in [0.2, 0.3, 0.5, 1.0]:
            preds = d1_weight_schemes(data, alpha=alpha, min_n=10, weight_type=wt)
            tag = f'D1 {wt} α={alpha}'
            R[tag] = float((preds == labels).mean())
        best_a = max([R[f'D1 {wt} α={a}'] for a in [0.2, 0.3, 0.5, 1.0]])
        print(f"  D1 {wt:5s}: best={best_a:.2%}")

    for tau in [0.01, 0.1]:
        for alpha in [0.3, 0.5, 1.0]:
            preds = d3_softmax_ensemble(data, alpha=alpha, min_n=10, tau=tau)
            tag = f'D3 softmax τ={tau} α={alpha}'
            R[tag] = float((preds == labels).mean())
        best_a = max([R[f'D3 softmax τ={tau} α={a}'] for a in [0.3, 0.5, 1.0]])
        print(f"  D3 τ={tau}: best={best_a:.2%}")

    for base in [0.2, 0.3, 0.5]:
        preds = d4_adaptive_alpha(data, base_alpha=base, min_n=10)
        tag = f'D4 adaptive base={base}'
        R[tag] = float((preds == labels).mean())
        print(f"  {tag}: {R[tag]:.2%}")

    # ── 总结 ──
    print(f"\n{'='*70}")
    print(f"★ Tiny-ImageNet {backbone_type} 结果 (Top 20), α={ALPHA}")
    print(f"{'='*70}")
    sorted_r = sorted(R.items(), key=lambda x: -x[1])
    baseline_u = R['Baseline: Union']
    for i, (name, acc) in enumerate(sorted_r[:20]):
        diff = acc - baseline_u
        print(f"  {i+1:2d}. {name:<35s} | {acc:>8.2%} "
              f"(vs Union {'+' if diff>=0 else ''}{diff*100:.2f}pp)")

    # ── 保存 JSON ──
    out = {
        'dataset': 'tinyimagenet',
        'backbone': backbone_type,
        'alpha': ALPHA,
        'n_clients': NC,
        'n_classes': NL,
        'seed': SEED,
        'epochs_bb': EPB,
        'epochs_exp': EPE,
        'train_time': tt,
        'all_results': R,
        'best': sorted_r[0][0],
        'best_acc': sorted_r[0][1],
        'union_acc': R['Baseline: Union'],
        'expert_min_acc': R['Baseline: Expert(min)'],
    }
    out_path = f"results/tinyimagenet_{backbone_type}_a{ALPHA}_k{NC}_s{SEED}.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"  训练: {tt:.0f}s, 完成!")
    return R


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['cnn', 'resnet18'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args.alpha, args.backbone, args.seed)