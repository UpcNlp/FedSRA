"""
ablation_fm_only.py
===================
只跑 "G: FM only (CE backbone)" 消融。
用标准 CE loss 训练 backbone → filter merge → ETF 分类器评估。

用法:
  python ablation_fm_only.py --alpha 0.05 --dataset cifar10 --seed 42

结果追加到 results/ablation_components_{dataset}_a{alpha}_s{seed}.json
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import os
import argparse

from rebuild8 import (
    device, USE_BF16,
    generate_etf, prepare_data,
)
from resnet18_filter_merge import ResNet18Backbone, union_aggregate_resnet18


def train_bb_ce(bb, loader, classes, num_classes, fd=256, epochs=600, lr=1e-3):
    """用标准 CE loss 训练 backbone (不用 ETF)"""
    bb = bb.to(device)
    cls_head = nn.Linear(fd, num_classes).to(device)
    bb.train(); cls_head.train()
    params = list(bb.parameters()) + list(cls_head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    amp = (torch.amp.autocast('cuda', dtype=torch.bfloat16) if USE_BF16
           else torch.amp.autocast('cuda', enabled=False))
    for ep in range(epochs):
        el = 0; nb = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with amp:
                f = bb(x)
                logits = cls_head(f)
                loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step()
            el += loss.item(); nb += 1
        sch.step()
        if (ep+1) % 100 == 0 or ep == 0:
            print(f"      BB-CE {ep+1}/{epochs} loss={el/max(nb,1):.4f}")
    bb.eval()
    return bb


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
        NC = 5; NL = 10; FD = 256; EPB = 600
        cal, ccl, tl, ccc = prepare_data(NC, args.alpha, NL)
    else:
        from rebuild8_cifar100 import prepare_data_cifar100
        NC = 5; NL = 100; FD = 256; EPB = 300
        cal, ccl, tl, ccc = prepare_data_cifar100(NC, args.alpha, NL)

    print(f"\n{'='*60}")
    print(f"  FM only (CE backbone): {args.dataset} α={args.alpha} seed={args.seed}")
    print(f"{'='*60}")

    etf = generate_etf(NL, FD)

    # 训练 CE backbone
    print(f"\n训练 CE backbone...")
    ce_bbs = []; t0 = time.time()
    for k in range(NC):
        cls = sorted(ccc[k].keys())
        print(f"  Client {k}: {len(cls)} cls, {sum(ccc[k].values())} samp")
        ce_bb = ResNet18Backbone(FD)
        ce_bb = train_bb_ce(ce_bb, cal[k], cls, NL, FD, EPB)
        ce_bbs.append(ce_bb)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    tt = time.time() - t0
    print(f"训练: {tt:.0f}s")

    # Filter merge
    print(f"\nFilter merge...")
    ubb = union_aggregate_resnet18(ce_bbs, FD, 0.95, device)
    ubb.eval()

    # ETF 分类器评估
    print(f"\n评估...")
    ed = etf.to(device)
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in tl:
            x = x.to(device)
            f = ubb(x)
            preds = torch.mm(f, ed.T).argmax(1).cpu()
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"\n  G: FM only (CE backbone) = {acc:.2%}")

    # 追加到已有的 ablation JSON
    os.makedirs('results', exist_ok=True)
    json_path = f"results/ablation_components_{args.dataset}_a{args.alpha}_s{args.seed}.json"

    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        data['results']['G: FM only (CE backbone)'] = acc
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  追加到: {json_path}")
    else:
        data = {
            'dataset': args.dataset,
            'alpha': args.alpha,
            'seed': args.seed,
            'results': {'G: FM only (CE backbone)': acc},
            'train_time_ce': tt,
        }
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  保存到: {json_path}")


if __name__ == '__main__':
    main()
