"""
verify_tiny.py
==============
快速验证 Tiny-ImageNet 上的关键假设.

训练只做一次, 然后对多个 thr 值做 filter merge + 推理 (不重新训练).
不同 FD 需要重新训练, 所以分开跑.

用法:
  # FD=256 + 4个thr (训练1次, 推理4次)
  CUDA_VISIBLE_DEVICES=0 python verify_tiny.py --fd 256 --alpha 0.5

  # FD=512 + 4个thr
  CUDA_VISIBLE_DEVICES=0 python verify_tiny.py --fd 512 --alpha 0.5

  # 更多 epoch
  CUDA_VISIBLE_DEVICES=0 python verify_tiny.py --fd 256 --alpha 0.5 --epb 500 --epe 300
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
    BasicBlock, union_aggregate_resnet18,
)

from rebuild8_tinyimagenet import (
    TINY_MEAN, TINY_STD,
    TinyImageNetValDataset,
    prepare_data_tinyimagenet,
    train_experts_tiny,
)


# ═══════════════════════════════════════════════════════════
# ResNet18 支持可变 FD
# ═══════════════════════════════════════════════════════════

class ResNet18Backbone64V(nn.Module):
    """ResNet-18 for 64x64, 支持可变 FD"""
    def __init__(self, fd=256):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, fd)

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
# 评估函数
# ═══════════════════════════════════════════════════════════

def evaluate(data, labels):
    R = {}
    R['Union'] = float((data['union_preds'] == labels).mean())

    preds = expert_original(data)
    R['Expert(min)'] = float((preds == labels).mean())

    e_pred, _, _ = _get_expert_preds_and_margin(data)
    u_correct = (data['union_preds'] == labels)
    e_correct = (e_pred == labels)
    R['Oracle'] = float((u_correct | e_correct).mean())

    for a in [0.3, 0.5, 1.0, 2.0]:
        for mn in [0, 10, 20]:
            preds = cross_client_per_client_logits(data, alpha=a, min_n=mn)
            R[f'C4 α={a} n≥{mn}'] = float((preds == labels).mean())

    for wt in ['log', 'sqrt']:
        for a in [0.3, 0.5, 1.0]:
            preds = d1_weight_schemes(data, alpha=a, min_n=10, weight_type=wt)
            R[f'D1 {wt} α={a}'] = float((preds == labels).mean())

    best_name = max((k for k in R if k != 'Oracle'), key=R.get)
    return R, best_name, R[best_name]


# ═══════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--fd', type=int, default=256)
    parser.add_argument('--epb', type=int, default=300)
    parser.add_argument('--epe', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--thrs', type=str, default='0.95,0.90,0.85,0.80',
                        help='逗号分隔的 thr 值, 训练一次全部测试')
    args = parser.parse_args()

    thrs = [float(x) for x in args.thrs.split(',')]
    alpha = args.alpha; fd = args.fd; seed = args.seed
    epb = args.epb; epe = args.epe
    NC = 5; NL = 200; LD = 32

    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    print(f"\n{'='*70}")
    print(f"  Tiny-ImageNet 验证")
    print(f"  α={alpha} FD={fd} EPB={epb} EPE={epe} seed={seed}")
    print(f"  测试 thr: {thrs}")
    print(f"{'='*70}")

    os.makedirs('results', exist_ok=True)
    etf = generate_etf(NL, fd)
    cal, ccl, tl, ccc = prepare_data_tinyimagenet(NC, alpha, NL)

    # ══════════════════════════════════════════════════
    # Phase 1: 训练 (只做一次)
    # ══════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  Phase 1: 训练 (FD={fd}, EPB={epb}, EPE={epe})")
    print(f"{'='*60}")

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

    # ══════════════════════════════════════════════════
    # Phase 2: 对每个 thr 做 Filter Merge + 推理
    # ══════════════════════════════════════════════════
    all_results = {}

    for thr in thrs:
        print(f"\n{'='*60}")
        print(f"  thr={thr}: Filter Merge + 推理")
        print(f"{'='*60}")

        ubb = union_aggregate_resnet18(bbs, fd, thr, device)
        n_params = sum(p.numel() for p in ubb.parameters())

        data = precompute_all(bbs, client_exps, ubb, tl, etf, ccc, NL)
        labels = data['labels']

        R, best_name, best_acc = evaluate(data, labels)

        print(f"    Union:    {R['Union']:.2%}")
        print(f"    Expert:   {R['Expert(min)']:.2%}")
        print(f"    Oracle:   {R['Oracle']:.2%}")
        print(f"    Best:     {best_name} = {best_acc:.2%}")
        print(f"    参数:      {n_params:,}")

        all_results[f'thr={thr}'] = {
            'thr': thr, 'n_params': n_params,
            'union': R['Union'], 'expert': R['Expert(min)'],
            'oracle': R['Oracle'], 'best': best_acc,
            'best_method': best_name, 'all': R,
        }

        del ubb, data
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════
    # 汇总
    # ══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  汇总: α={alpha} FD={fd} EPB={epb}")
    print(f"{'='*70}")
    print(f"  {'thr':>5s} | {'Union':>8s} | {'Expert':>8s} | {'Best':>8s} | {'Oracle':>8s} | {'参数':>12s} | Best method")
    print(f"  {'-'*85}")
    for key, r in all_results.items():
        print(f"  {r['thr']:>5.2f} | {r['union']:>7.2%} | {r['expert']:>7.2%} | "
              f"{r['best']:>7.2%} | {r['oracle']:>7.2%} | {r['n_params']:>12,} | {r['best_method']}")

    fafi = {0.05: 36.96, 0.1: 43.62, 0.3: 53.32, 0.5: 56.48}
    if alpha in fafi:
        print(f"\n  FAFI (α={alpha}): {fafi[alpha]:.2f}%")
        best_ours = max(r['best'] for r in all_results.values())
        print(f"  Ours best:     {best_ours:.2%}")
        print(f"  差距:           {best_ours*100 - fafi[alpha]:+.2f}pp")

    out = {
        'alpha': alpha, 'fd': fd, 'epb': epb, 'epe': epe, 'seed': seed,
        'train_time': tt,
        'results': all_results,
    }
    tag = f"fd{fd}_ep{epb}_a{alpha}_s{seed}"
    out_path = f"results/verify_tiny_{tag}.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
