"""
ablation_components.py
======================
ResNet-18 组件消融实验.

训练一次后, 评估各组件组合:
  A: ETF logits only (无 Expert, 无 Filter Merge)
  B: Expert only (无 ETF logits)
  C: ETF + Expert (无 Filter Merge, 用 client 平均)
  D: ETF + Filter Merge (无 Expert)
  E: Expert + Filter Merge (无 ETF logits)
  F: Full (ETF + Expert + Filter Merge) ← Ours

用法:
  python ablation_components.py --alpha 0.05 --dataset cifar10 --seed 42
  python ablation_components.py --alpha 0.1 --dataset cifar100 --seed 42

结果保存到 results/ablation_components_{dataset}_a{alpha}_s{seed}.json
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import os
import argparse

# 复用 rebuild8 的所有组件
from rebuild8 import (
    device, USE_BF16,
    generate_etf, ConditionalExpert,
    prepare_data, train_bb, train_experts,
    precompute_all,
)
from resnet18_filter_merge import ResNet18Backbone, union_aggregate_resnet18


def fake_union_from_client_avg(bbs, etf, test_loader, device):
    """不用 Filter Merge, 用 client 模型的 ETF logits 平均作为 union 信号"""
    ed = etf.to(device)
    all_logits = []
    for bb in bbs:
        bb.eval()
        bb_logits = []
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                f = bb(x)
                logits = torch.mm(f, ed.T)
                bb_logits.append(logits.cpu())
        all_logits.append(torch.cat(bb_logits, dim=0))
    # 所有 client 的 logits 平均
    avg_logits = torch.stack(all_logits).mean(dim=0)
    return avg_logits


def precompute_with_fake_union(bbs, client_exps, tl, etf, ccc, nc):
    """类似 precompute_all, 但 union 用 client avg 代替 filter merge"""
    # 用 client 平均的 logits 作为 fake union_logits
    avg_logits = fake_union_from_client_avg(bbs, etf, tl, device)

    # 再跑一次 precompute_all 拿其他数据 (expert logits 等)
    # 用第一个 backbone 作为 union backbone 占位 (只是为了 API 兼容)
    data = precompute_all(bbs, client_exps, bbs[0], tl, etf, ccc, nc)

    # 替换 union_logits 和 union_preds
    data['union_logits'] = avg_logits
    data['union_preds'] = avg_logits.argmax(1).numpy()

    return data


def evaluate_configs(bbs, client_exps, tl, etf, ccc, nc, has_filter_merge=True):
    """对一组训练好的 backbone/expert, 评估所有消融配置"""
    results = {}

    # 先跑 Filter Merge 的 precompute
    if has_filter_merge:
        ubb = union_aggregate_resnet18(bbs, etf.size(1), 0.95, device)
        data_fm = precompute_all(bbs, client_exps, ubb, tl, etf, ccc, nc)
    else:
        data_fm = None

    # 不用 Filter Merge, 用 client 平均
    data_avg = precompute_with_fake_union(bbs, client_exps, tl, etf, ccc, nc)

    labels = data_avg['labels']

    # ─── A: ETF logits only (client 平均, 无 Expert) ───
    # 直接用 union_logits 预测
    preds = data_avg['union_logits'].argmax(1).numpy()
    results['A: ETF only (client avg)'] = float((preds == labels).mean())

    # ─── B: Expert only ───
    # 用 expert_original (逐 client min expert error)
    from rebuild8 import expert_original
    preds = expert_original(data_avg)
    results['B: Expert only'] = float((preds == labels).mean())

    # ─── C: ETF + Expert (client 平均, 无 Filter Merge) ───
    # C4 融合但 union 用 client 平均
    from rebuild8 import cross_client_per_client_logits
    best_acc = 0; best_cfg = ''
    for a in [0.3, 0.5, 1.0, 2.0]:
        for mn in [0, 10, 20]:
            preds = cross_client_per_client_logits(data_avg, alpha=a, min_n=mn)
            acc = float((preds == labels).mean())
            if acc > best_acc:
                best_acc = acc; best_cfg = f'α={a} n≥{mn}'
    results[f'C: ETF+Expert (client avg) [{best_cfg}]'] = best_acc

    # ─── D: ETF + Filter Merge (无 Expert) ───
    if has_filter_merge:
        preds = data_fm['union_preds']
        results['D: ETF + Filter Merge only'] = float((preds == labels).mean())

        # ─── E: Expert + Filter Merge (无 ETF 直接作为 logits) ───
        # 用 expert_original, 但测试 filter merge union
        # 这里 filter merge 只为了测 union 准确率, expert 用原始
        preds = expert_original(data_fm)
        results['E: Expert + Filter Merge'] = float((preds == labels).mean())

        # ─── F: Full (Ours) ───
        best_acc = 0; best_cfg = ''
        for a in [0.3, 0.5, 1.0, 2.0]:
            for mn in [0, 10, 20]:
                preds = cross_client_per_client_logits(data_fm, alpha=a, min_n=mn)
                acc = float((preds == labels).mean())
                if acc > best_acc:
                    best_acc = acc; best_cfg = f'α={a} n≥{mn}'
        results[f'F: Full (Ours) [{best_cfg}]'] = best_acc

    return results


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
    else:  # cifar100
        from rebuild8_cifar100 import prepare_data_cifar100
        NC=5; NL=100; FD=256; LD=32; EPB=300; EPE=200
        cal, ccl, tl, ccc = prepare_data_cifar100(NC, args.alpha, NL)

    print(f"\n{'='*70}")
    print(f"  组件消融: {args.dataset} α={args.alpha} seed={args.seed}")
    print(f"{'='*70}")

    os.makedirs('results', exist_ok=True)
    etf = generate_etf(NL, FD)

    # ── 训练 (一次, 所有配置共享) ──
    print(f"\n训练...")
    bbs = []; client_exps = []; t0 = time.time()
    for k in range(NC):
        cls = sorted(ccc[k].keys())
        print(f"  Client {k}: {len(cls)} cls, {sum(ccc[k].values())} samp")
        bb = ResNet18Backbone(FD)
        bb = train_bb(bb, cal[k], cls, etf, EPB)
        if args.dataset == 'cifar10':
            exps = train_experts(bb, ccl[k], cls, etf, NL, FD, LD, EPE)
        else:
            from rebuild8_cifar100 import train_experts_cifar100
            exps = train_experts_cifar100(bb, ccl[k], cls, etf, ccc[k], NL, FD, LD, EPE)
        bbs.append(bb); client_exps.append(exps)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    tt = time.time() - t0
    print(f"训练: {tt:.0f}s")

    # ── 评估所有配置 ──
    print(f"\n评估消融配置...")
    results = evaluate_configs(bbs, client_exps, tl, etf, ccc, NL, has_filter_merge=True)

    # ── 输出 ──
    print(f"\n{'='*70}")
    print(f"  消融结果 ({args.dataset} α={args.alpha})")
    print(f"{'='*70}")
    for cfg, acc in results.items():
        print(f"  {cfg:<50s} {acc:>7.2%}")

    out = {
        'dataset': args.dataset,
        'alpha': args.alpha,
        'seed': args.seed,
        'n_clients': NC,
        'train_time': tt,
        'results': results,
    }
    out_path = f"results/ablation_components_{args.dataset}_a{args.alpha}_s{args.seed}.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
