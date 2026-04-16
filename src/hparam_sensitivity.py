"""
hparam_sensitivity.py
=====================
超参敏感性分析. 训练一次, 扫描多个超参.

扫描的超参:
  1. α_fuse (C4 融合权重): 0.2, 0.3, 0.5, 1.0, 2.0
  2. min_n (expert 样本阈值): 0, 5, 10, 20, 50
  3. thr (filter merge 阈值): 0.80, 0.85, 0.90, 0.95

用法:
  python hparam_sensitivity.py --alpha 0.05 --dataset cifar10 --seed 42
  python hparam_sensitivity.py --alpha 0.1 --dataset cifar100 --seed 42

结果保存到 results/hparam_{dataset}_a{alpha}_s{seed}.json
"""

import torch
import numpy as np
import json
import time
import os
import argparse

from rebuild8 import (
    device, generate_etf,
    prepare_data, train_bb, train_experts,
    precompute_all,
    cross_client_per_client_logits,
)
from resnet18_filter_merge import ResNet18Backbone, union_aggregate_resnet18


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    if args.dataset == 'cifar10':
        NC=5; NL=10; FD=256; LD=32; EPB=600; EPE=600
        cal, ccl, tl, ccc = prepare_data(NC, args.alpha, NL)
    else:
        from rebuild8_cifar100 import prepare_data_cifar100, train_experts_cifar100
        NC=5; NL=100; FD=256; LD=32; EPB=300; EPE=200
        cal, ccl, tl, ccc = prepare_data_cifar100(NC, args.alpha, NL)

    print(f"\n{'='*70}")
    print(f"  超参敏感性: {args.dataset} α={args.alpha} seed={args.seed}")
    print(f"{'='*70}")

    os.makedirs('results', exist_ok=True)
    etf = generate_etf(NL, FD)

    # ── 训练 ──
    print(f"\n训练...")
    bbs = []; client_exps = []; t0 = time.time()
    for k in range(NC):
        cls = sorted(ccc[k].keys())
        print(f"  Client {k}: {len(cls)} cls")
        bb = ResNet18Backbone(FD)
        bb = train_bb(bb, cal[k], cls, etf, EPB)
        if args.dataset == 'cifar10':
            exps = train_experts(bb, ccl[k], cls, etf, NL, FD, LD, EPE)
        else:
            exps = train_experts_cifar100(bb, ccl[k], cls, etf, ccc[k], NL, FD, LD, EPE)
        bbs.append(bb); client_exps.append(exps)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    tt = time.time() - t0
    print(f"训练: {tt:.0f}s ({tt/60:.1f}min)")

    # ── 扫描 thr (每个 thr 需要重新做 filter merge + precompute) ──
    print(f"\n扫描 thr...")
    thr_results = {}  # thr -> {best_acc, best_cfg}
    data_by_thr = {}  # 缓存 data 用于后续扫描

    for thr in [0.80, 0.85, 0.90, 0.95]:
        ubb = union_aggregate_resnet18(bbs, FD, thr, device)
        data = precompute_all(bbs, client_exps, ubb, tl, etf, ccc, NL)
        data_by_thr[thr] = data

        # 在默认超参下测一下
        labels = data['labels']
        preds = cross_client_per_client_logits(data, alpha=1.0, min_n=10)
        acc = float((preds == labels).mean())
        thr_results[thr] = acc
        print(f"  thr={thr}: {acc:.2%}")

        del ubb
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ── 扫描 α_fuse (用默认 thr=0.95 的 data) ──
    print(f"\n扫描 α_fuse...")
    alpha_results = {}
    data = data_by_thr[0.95]
    labels = data['labels']
    for a in [0.2, 0.3, 0.5, 1.0, 2.0]:
        preds = cross_client_per_client_logits(data, alpha=a, min_n=10)
        acc = float((preds == labels).mean())
        alpha_results[a] = acc
        print(f"  α_fuse={a}: {acc:.2%}")

    # ── 扫描 min_n ──
    print(f"\n扫描 min_n...")
    minn_results = {}
    for mn in [0, 5, 10, 20, 50]:
        preds = cross_client_per_client_logits(data, alpha=1.0, min_n=mn)
        acc = float((preds == labels).mean())
        minn_results[mn] = acc
        print(f"  min_n={mn}: {acc:.2%}")

    # ── 保存 ──
    out = {
        'dataset': args.dataset,
        'alpha': args.alpha,
        'seed': args.seed,
        'train_time': tt,
        'thr_scan': thr_results,        # 固定 α_fuse=1.0, min_n=10
        'alpha_fuse_scan': alpha_results,  # 固定 thr=0.95, min_n=10
        'min_n_scan': minn_results,     # 固定 thr=0.95, α_fuse=1.0
    }
    out_path = f"results/hparam_{args.dataset}_a{args.alpha}_s{args.seed}.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
