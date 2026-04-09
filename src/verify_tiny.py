"""
verify_tiny.py
==============
快速验证 Tiny-ImageNet 上的关键假设:
  1. FD=256 vs FD=512: ETF 维度是否是瓶颈
  2. thr=0.95 vs 0.90 vs 0.85: filter merge 阈值影响
  3. EPB=300 vs EPB=500: 训练 epoch 是否不足

只跑 α=0.05, seed=42, 一个配置约 5 小时.
所有配置共享数据划分, 结果保存到 results/verify_tiny_*.json

用法:
  # 验证 FD
  CUDA_VISIBLE_DEVICES=0 python verify_tiny.py --exp fd --fd 512
  CUDA_VISIBLE_DEVICES=0 python verify_tiny.py --exp fd --fd 256   # baseline

  # 验证 thr
  CUDA_VISIBLE_DEVICES=0 python verify_tiny.py --exp thr --thr 0.85
  CUDA_VISIBLE_DEVICES=0 python verify_tiny.py --exp thr --thr 0.90
  CUDA_VISIBLE_DEVICES=0 python verify_tiny.py --exp thr --thr 0.95  # baseline

  # 验证 epoch
  CUDA_VISIBLE_DEVICES=0 python verify_tiny.py --exp epoch --epb 500 --epe 300

  # 一次全跑 (串行)
  CUDA_VISIBLE_DEVICES=0 python verify_tiny.py --exp all
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import warnings
import json
import time
import os
import argparse
from collections import defaultdict

warnings.filterwarnings('ignore')

from rebuild8 import (
    device, USE_BF16,
    generate_etf, ConditionalExpert,
    etf_cl, etf_al, train_bb, preextract, train_exp,
    precompute_all,
    expert_original, _get_expert_preds_and_margin,
    cross_client_per_client_logits,
    d1_weight_schemes,
)

from resnet18_filter_merge import (
    BasicBlock, ResNet18Backbone, union_aggregate_resnet18,
)

from rebuild8_tinyimagenet import (
    TINY_MEAN, TINY_STD,
    TinyImageNetValDataset,
    prepare_data_tinyimagenet,
    ResNet18Backbone64,
    train_experts_tiny,
)


# ═══════════════════════════════════════════════════════════
# ResNet18 支持可变 FD (512 维)
# ═══════════════════════════════════════════════════════════

class ResNet18Backbone64V(nn.Module):
    """ResNet-18 for 64x64, 支持可变 FD"""
    def __init__(self, fd=256, internal_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, internal_dim, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(internal_dim, fd)

    def _make_layer(self, ic, oc, n_blocks, stride):
        layers = [BasicBlock(ic, oc, stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(oc, oc))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return F.normalize(self.fc(x), dim=1)


# ═══════════════════════════════════════════════════════════
# 单次实验
# ═══════════════════════════════════════════════════════════

def run_one(alpha, fd, thr, epb, epe, seed, tag):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    NC = 5; NL = 200; LD = 32

    print(f"\n{'='*70}")
    print(f"  [{tag}] α={alpha} FD={fd} thr={thr} EPB={epb} EPE={epe} seed={seed}")
    print(f"{'='*70}")

    etf = generate_etf(NL, fd)
    cal, ccl, tl, ccc = prepare_data_tinyimagenet(NC, alpha, NL)

    # ── 训练 ──
    bbs = []; client_exps = []; t0 = time.time()
    for k in range(NC):
        cls = sorted(ccc[k].keys())
        n_samp = sum(ccc[k].values())
        print(f"\n  Client {k}: {len(cls)}/{NL} cls, {n_samp} samp")

        bb = ResNet18Backbone64V(fd)
        bb = train_bb(bb, cal[k], cls, etf, epb)
        exps = train_experts_tiny(bb, ccl[k], cls, etf, ccc[k],
                                   nc=NL, fdim=fd, ldim=LD, epochs=epe)
        bbs.append(bb); client_exps.append(exps)
        print(f"    训练了 {len(exps)} 个 expert")
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    tt = time.time() - t0
    print(f"\n  训练完成: {tt:.0f}s ({tt/60:.1f}min)")

    # ── Filter Merge ──
    print(f"\n  Filter Merge (thr={thr})...")
    ubb = union_aggregate_resnet18(bbs, fd, thr, device)
    n_params = sum(p.numel() for p in ubb.parameters())

    # ── 预计算 ──
    data = precompute_all(bbs, client_exps, ubb, tl, etf, ccc, NL)
    labels = data['labels']

    # ── 评估 ──
    R = {}
    R['Union'] = float((data['union_preds'] == labels).mean())

    preds = expert_original(data)
    R['Expert(min)'] = float((preds == labels).mean())

    e_pred, _, _ = _get_expert_preds_and_margin(data)
    u_correct = (data['union_preds'] == labels)
    e_correct = (e_pred == labels)
    R['Oracle'] = float((u_correct | e_correct).mean())

    # C4 + D1 (关键策略)
    for a in [0.3, 0.5, 1.0]:
        for mn in [0, 10, 20]:
            preds = cross_client_per_client_logits(data, alpha=a, min_n=mn)
            R[f'C4 α={a} n≥{mn}'] = float((preds == labels).mean())

    for wt in ['log', 'sqrt']:
        for a in [0.3, 0.5, 1.0]:
            preds = d1_weight_schemes(data, alpha=a, min_n=10, weight_type=wt)
            R[f'D1 {wt} α={a}'] = float((preds == labels).mean())

    # 找最佳非 Oracle
    best_name = max((k for k in R if k != 'Oracle'), key=R.get)
    best_acc = R[best_name]

    print(f"\n  结果:")
    print(f"    Union:      {R['Union']:.2%}")
    print(f"    Expert:     {R['Expert(min)']:.2%}")
    print(f"    Oracle:     {R['Oracle']:.2%}")
    print(f"    Best:       {best_name} = {best_acc:.2%}")
    print(f"    合并参数:    {n_params:,}")

    # 保存
    out = {
        'tag': tag, 'alpha': alpha, 'fd': fd, 'thr': thr,
        'epb': epb, 'epe': epe, 'seed': seed,
        'n_params_merged': n_params,
        'train_time': tt,
        'results': R,
        'best': best_name, 'best_acc': best_acc,
    }
    out_path = f"results/verify_tiny_{tag}.json"
    os.makedirs('results', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {out_path}")
    return out


# ═══════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='fd',
                        choices=['fd', 'thr', 'epoch', 'all'],
                        help='验证哪个因素')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--fd', type=int, default=512)
    parser.add_argument('--thr', type=float, default=0.95)
    parser.add_argument('--epb', type=int, default=300)
    parser.add_argument('--epe', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    alpha = args.alpha; seed = args.seed

    if args.exp == 'fd':
        tag = f"fd{args.fd}_a{alpha}"
        run_one(alpha, args.fd, 0.95, 300, 200, seed, tag)

    elif args.exp == 'thr':
        tag = f"thr{args.thr}_a{alpha}"
        run_one(alpha, 256, args.thr, 300, 200, seed, tag)

    elif args.exp == 'epoch':
        tag = f"ep{args.epb}_a{alpha}"
        run_one(alpha, 256, 0.95, args.epb, args.epe, seed, tag)

    elif args.exp == 'all':
        results = []
        configs = [
            # baseline
            ('baseline_fd256', alpha, 256, 0.95, 300, 200),
            # FD 验证
            ('fd512', alpha, 512, 0.95, 300, 200),
            # thr 验证 (用 FD=256 隔离变量)
            ('thr090', alpha, 256, 0.90, 300, 200),
            ('thr085', alpha, 256, 0.85, 300, 200),
            ('thr080', alpha, 256, 0.80, 300, 200),
            # FD=512 + 低 thr (组合)
            ('fd512_thr085', alpha, 512, 0.85, 300, 200),
            # epoch 验证
            ('ep500', alpha, 256, 0.95, 500, 300),
        ]
        for tag, a, fd, thr, epb, epe in configs:
            out = run_one(a, fd, thr, epb, epe, seed, tag)
            results.append(out)

        # ── 汇总表 ──
        print(f"\n{'='*80}")
        print(f"  汇总: Tiny-ImageNet α={alpha}")
        print(f"{'='*80}")
        print(f"  {'配置':<25s} | {'FD':>4s} | {'thr':>5s} | {'EPB':>4s} | {'Union':>8s} | {'Expert':>8s} | {'Best':>8s} | {'合并参数':>12s}")
        print(f"  {'-'*95}")
        for r in results:
            print(f"  {r['tag']:<25s} | {r['fd']:>4d} | {r['thr']:>5.2f} | {r['epb']:>4d} | "
                  f"{r['results']['Union']:>7.2%} | {r['results']['Expert(min)']:>7.2%} | "
                  f"{r['best_acc']:>7.2%} | {r['n_params_merged']:>12,}")

        # 保存汇总
        summary = {r['tag']: {
            'fd': r['fd'], 'thr': r['thr'], 'epb': r['epb'],
            'union': r['results']['Union'],
            'expert': r['results']['Expert(min)'],
            'best': r['best_acc'],
            'best_method': r['best'],
            'n_params': r['n_params_merged'],
        } for r in results}
        with open(f'results/verify_tiny_summary_a{alpha}.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Summary saved: results/verify_tiny_summary_a{alpha}.json")


if __name__ == '__main__':
    main()
