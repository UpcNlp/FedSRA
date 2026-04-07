"""
rebuild8_resnet18.py
====================
和 rebuild8.py 完全相同的实验流程, 但使用 ResNet-18 backbone + ResNet-18 filter merge.

不修改 rebuild8.py 的任何代码.

用法:
  python rebuild8_resnet18.py --alpha 0.05 --gpu 0
  python rebuild8_resnet18.py --alpha 0.1 --gpu 0

  # 串行跑所有 alpha
  python rebuild8_resnet18.py --alpha 0.05 --gpu 0 && \
  python rebuild8_resnet18.py --alpha 0.1  --gpu 0 && \
  python rebuild8_resnet18.py --alpha 0.3  --gpu 0 && \
  python rebuild8_resnet18.py --alpha 0.5  --gpu 0 && \
  python rebuild8_resnet18.py --alpha 1.0  --gpu 0

结果保存到 results/resnet18_a{alpha}_k{n_clients}_s42.json
"""

import argparse
import json
import os
import time

# ── 从 rebuild8 导入所有共享组件 ──
from rebuild8 import (
    # 全局变量
    device, USE_BF16, DL_KWARGS,
    # 数据
    dirichlet_split, prepare_data, generate_etf,
    # 模型 (只导入 Expert, 不导入 Backbone)
    ConditionalExpert,
    # 训练
    etf_cl, etf_al, train_bb, preextract, train_exp, train_experts, compute_stats,
    # 预计算
    precompute_all,
    # 推理策略
    expert_original, expert_quality_filter, expert_weighted_avg,
    expert_quality_min, expert_top_quality, expert_weighted_min,
    _get_expert_preds_and_margin, _union_norm,
    route_ensemble_logits, route_ensemble_quality,
    cross_client_voting, cross_client_weighted_vote,
    cross_client_per_client_logits,
    d1_weight_schemes, d3_softmax_ensemble, d4_adaptive_alpha,
    # 可视化
    plot_results,
)
import torch
import numpy as np

# ── 从 resnet18_filter_merge 导入 ResNet-18 组件 ──
from resnet18_filter_merge import ResNet18Backbone, union_aggregate_resnet18


def main(ALPHA=0.05, GPU=0):
    print("\n" + "=" * 80)
    print("rebuild8 + ResNet-18 Filter Merge")
    print("=" * 80)

    NC=5; NL=10; FD=256; LD=32; EPB=600; EPE=600

    os.makedirs('outputs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    etf = generate_etf(NL, FD)
    cal, ccl, tl, ccc = prepare_data(NC, ALPHA, NL)

    # ── Phase 1: 训练 (ResNet-18 代替 CNN) ──
    print(f"\n{'='*60}\n  Phase 1: 训练 (ResNet-18)\n{'='*60}")
    bbs = []; client_exps = []; t0 = time.time()
    for k in range(NC):
        cls = sorted(ccc[k].keys())
        print(f"\n  Client {k}: {len(cls)} cls, {sum(ccc[k].values())} samp")

        # ★ 这里用 ResNet18Backbone 代替 Backbone
        bb = ResNet18Backbone(FD)
        bb = train_bb(bb, cal[k], cls, etf, EPB)
        exps = train_experts(bb, ccl[k], cls, etf, NL, FD, LD, EPE)
        for c in cls:
            mu, _ = compute_stats(bb, exps[c], ccl[k][c], etf[c])
            print(f"    c{c}: n={ccc[k][c]:5d} μ={mu:.6f}")
        bbs.append(bb); client_exps.append(exps)
    tt = time.time() - t0
    print(f"\n  训练: {tt:.1f}s")

    # ── Phase 2: Union (ResNet-18 Filter Merge) ──
    print(f"\n{'='*60}\n  Phase 2: ResNet-18 Filter Merge\n{'='*60}")

    # ★ 这里用 union_aggregate_resnet18 代替 union_aggregate
    ubb = union_aggregate_resnet18(bbs, FD, 0.95, device)

    # ── Phase 3: 预计算 ──
    print(f"\n{'='*60}\n  Phase 3: 预计算\n{'='*60}")
    data = precompute_all(bbs, client_exps, ubb, tl, etf, ccc, NL)
    labels = data['labels']
    N = data['N']

    # ── Phase 4: 评估 (和 rebuild8 完全相同的策略) ──
    print(f"\n{'='*60}\n  Phase 4: 策略评估\n{'='*60}")
    R = {}

    # Baselines
    print(f"\n  --- Baselines ---")
    R['Baseline: Union'] = float((data['union_preds'] == labels).mean())
    print(f"  Union:                 {R['Baseline: Union']:.2%}")

    preds = expert_original(data)
    R['Baseline: Expert(min)'] = float((preds == labels).mean())
    print(f"  Expert (original min): {R['Baseline: Expert(min)']:.2%}")

    # Oracle
    e_pred, _, _ = _get_expert_preds_and_margin(data)
    u_correct = (data['union_preds'] == labels)
    e_correct = (e_pred == labels)
    R['Oracle: U∪E'] = float((u_correct | e_correct).mean())
    print(f"  Oracle (U∪E):          {R['Oracle: U∪E']:.2%}")

    # ── A: Expert 聚合 ──
    print(f"\n  --- Expert 聚合 ---")
    for min_n in [50, 100, 200]:
        preds = expert_quality_min(data, min_n=min_n)
        R[f'A3 QualMin n≥{min_n}'] = float((preds == labels).mean())
        print(f"  A3 QualMin n≥{min_n:3d}: {R[f'A3 QualMin n≥{min_n}']:.2%}")

    preds = expert_top_quality(data)
    R['A4 TopQuality'] = float((preds == labels).mean())
    print(f"  A4 TopQuality:    {R['A4 TopQuality']:.2%}")

    # ── B: Ensemble ──
    print(f"\n  --- Ensemble ---")
    for alpha in [0.3, 0.5, 1.0, 2.0]:
        preds = route_ensemble_logits(data, alpha=alpha)
        R[f'B6 Ensemble α={alpha}'] = float((preds == labels).mean())
        print(f"  B6 Ensemble α={alpha}: {R[f'B6 Ensemble α={alpha}']:.2%}")

    for alpha in [0.3, 0.5, 1.0]:
        for min_n in [50, 100]:
            preds = route_ensemble_quality(data, alpha=alpha, min_n=min_n)
            tag = f'B7 QualEns α={alpha} n≥{min_n}'
            R[tag] = float((preds == labels).mean())
            print(f"  {tag}: {R[tag]:.2%}")

    # ── C: 跨 Client ──
    print(f"\n  --- 跨 Client ---")
    for min_n in [0, 50, 100]:
        preds = cross_client_voting(data, min_n=min_n)
        R[f'C1 Vote n≥{min_n}'] = float((preds == labels).mean())
        print(f"  C1 Vote n≥{min_n}: {R[f'C1 Vote n≥{min_n}']:.2%}")

    for alpha in [0.3, 0.5, 1.0, 2.0]:
        for min_n in [0, 50, 100]:
            preds = cross_client_per_client_logits(data, alpha=alpha, min_n=min_n)
            tag = f'C4 PCEns α={alpha} n≥{min_n}'
            R[tag] = float((preds == labels).mean())
            print(f"  {tag}: {R[tag]:.2%}")

    # ── D: 深挖 Ensemble ──
    print(f"\n  --- 深挖 Ensemble ---")
    for wt in ['log', 'sqrt']:
        for alpha in [0.2, 0.3, 0.5]:
            preds = d1_weight_schemes(data, alpha=alpha, min_n=100, weight_type=wt)
            tag = f'D1 {wt} α={alpha}'
            R[tag] = float((preds == labels).mean())
        best_a = max([R[f'D1 {wt} α={a}'] for a in [0.2, 0.3, 0.5]])
        print(f"  D1 {wt:5s}: best={best_a:.2%}")

    for tau in [0.01, 0.1]:
        for alpha in [0.3, 0.5, 1.0]:
            preds = d3_softmax_ensemble(data, alpha=alpha, min_n=100, tau=tau)
            tag = f'D3 softmax τ={tau} α={alpha}'
            R[tag] = float((preds == labels).mean())
        best_a = max([R[f'D3 softmax τ={tau} α={a}'] for a in [0.3, 0.5, 1.0]])
        print(f"  D3 τ={tau}: best={best_a:.2%}")

    for base in [0.2, 0.3, 0.5]:
        preds = d4_adaptive_alpha(data, base_alpha=base, min_n=100)
        R[f'D4 adaptive base={base}'] = float((preds == labels).mean())
        print(f"  D4 base={base}: {R[f'D4 adaptive base={base}']:.2%}")

    # ── 总结 ──
    print(f"\n{'='*70}")
    print(f"★ ResNet-18 结果 (Top 20), α={ALPHA}")
    print(f"{'='*70}")
    sorted_r = sorted(R.items(), key=lambda x: -x[1])
    baseline_u = R['Baseline: Union']
    for i, (name, acc) in enumerate(sorted_r[:20]):
        diff = acc - baseline_u
        print(f"  {i+1:2d}. {name:<35s} | {acc:>8.2%} "
              f"(vs Union {'+' if diff>=0 else ''}{diff*100:.2f}pp)")

    # ── 保存 JSON ──
    out = {
        'backbone': 'resnet18',
        'alpha': ALPHA,
        'n_clients': NC,
        'seed': 42,
        'train_time': tt,
        'all_results': R,
        'best': sorted_r[0][0],
        'best_acc': sorted_r[0][1],
        'union_acc': R['Baseline: Union'],
        'expert_min_acc': R['Baseline: Expert(min)'],
    }
    out_path = f"results/resnet18_a{ALPHA}_k{NC}_s42.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")

    # 对比 CNN 结果 (如果存在)
    cnn_path = f"results/ablation_fusion_a{ALPHA}_k{NC}_s42.json"
    if os.path.exists(cnn_path):
        with open(cnn_path) as f:
            cnn = json.load(f)
        cnn_best = max(cnn['methods'].items(), key=lambda x: x[1]['c4_best'])
        print(f"\n  对比 CNN:")
        print(f"    CNN best:     {cnn_best[0]} = {cnn_best[1]['c4_best']:.2%}")
        print(f"    ResNet18 best: {sorted_r[0][0]} = {sorted_r[0][1]:.2%}")
        diff = sorted_r[0][1] - cnn_best[1]['c4_best']
        print(f"    差距: {diff:+.2%}")

    plot_results(dict(sorted_r), f'outputs/resnet18_a{ALPHA}.png')
    print(f"\n  完成!")
    return R


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    # 设置 GPU (rebuild8 里用全局 device)
    if args.gpu != 0:
        import rebuild8
        rebuild8.device = torch.device(f'cuda:{args.gpu}')

    main(args.alpha, args.gpu)
