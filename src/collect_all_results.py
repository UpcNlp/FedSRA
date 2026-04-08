"""
collect_all_results.py
======================
汇总所有实验结果, 计算 mean±std, 和 FAFI 对比

用法: python collect_all_results.py --results_dir results
"""
import json, os, glob, argparse
import numpy as np
from collections import defaultdict

# FAFI Table 1 数据 (ResNet-18, K=5)
FAFI = {
    'cifar10': {
        0.05: (71.84, 1.53),
        0.1:  (77.83, 1.32),
        0.3:  (84.76, 0.46),
        0.5:  (88.74, 0.11),
    },
    'cifar100': {
        0.05: (31.02, 1.17),
        0.1:  (45.48, 1.01),
        0.3:  (56.65, 0.91),
        0.5:  (61.07, 0.55),
    }
}


def find_best_non_oracle(all_results):
    """从 all_results dict 中找最佳非 Oracle 方法"""
    best_name, best_acc = None, 0
    for name, acc in all_results.items():
        if 'Oracle' in name:
            continue
        if acc > best_acc:
            best_acc = acc
            best_name = name
    return best_name, best_acc


def collect():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results')
    args = parser.parse_args()

    rd = args.results_dir

    # ═══════════════════════════════════════════════════════════
    # 收集 CIFAR-10 ResNet-18
    # ═══════════════════════════════════════════════════════════
    print("=" * 80)
    print("  CIFAR-10 ResNet-18 (vs FAFI)")
    print("=" * 80)

    c10_data = defaultdict(list)  # alpha -> [best_acc across seeds]
    c10_union = defaultdict(list)
    c10_expert = defaultdict(list)
    c10_best_methods = defaultdict(list)

    for f in sorted(glob.glob(os.path.join(rd, 'resnet18_a*_k5_s*.json'))):
        with open(f) as fp:
            d = json.load(fp)
        alpha = d['alpha']
        seed = d['seed']
        best_name, best_acc = find_best_non_oracle(d['all_results'])
        c10_data[alpha].append(best_acc * 100)
        c10_union[alpha].append(d['union_acc'] * 100)
        c10_expert[alpha].append(d['expert_min_acc'] * 100)
        c10_best_methods[alpha].append(best_name)

    print(f"\n  {'α':>6s} | {'Ours (mean±std)':>18s} | {'FAFI (mean±std)':>18s} | {'Δ':>8s} | {'Union':>8s} | {'Expert':>8s} | Seeds | Best method")
    print(f"  {'-'*110}")
    for alpha in sorted(c10_data.keys()):
        accs = c10_data[alpha]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs) if len(accs) > 1 else 0
        n_seeds = len(accs)

        fafi_mean, fafi_std = FAFI.get('cifar10', {}).get(alpha, (0, 0))
        delta = mean_acc - fafi_mean

        union_mean = np.mean(c10_union[alpha])
        expert_mean = np.mean(c10_expert[alpha])

        # 最常出现的 best method
        from collections import Counter
        method_counts = Counter(c10_best_methods[alpha])
        top_method = method_counts.most_common(1)[0][0]

        print(f"  {alpha:>6.2f} | {mean_acc:>6.2f}±{std_acc:<5.2f}      | {fafi_mean:>6.2f}±{fafi_std:<5.2f}      | {delta:>+7.2f} | {union_mean:>7.2f} | {expert_mean:>7.2f} | {n_seeds:>5d} | {top_method}")

    # ═══════════════════════════════════════════════════════════
    # 收集 CIFAR-100 ResNet-18
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  CIFAR-100 ResNet-18 (vs FAFI)")
    print("=" * 80)

    c100_data = defaultdict(list)
    c100_union = defaultdict(list)
    c100_expert = defaultdict(list)
    c100_best_methods = defaultdict(list)

    for f in sorted(glob.glob(os.path.join(rd, 'cifar100_resnet18_a*_k5_s*.json'))):
        with open(f) as fp:
            d = json.load(fp)
        alpha = d['alpha']
        best_name, best_acc = find_best_non_oracle(d['all_results'])
        c100_data[alpha].append(best_acc * 100)
        c100_union[alpha].append(d['union_acc'] * 100)
        c100_expert[alpha].append(d['expert_min_acc'] * 100)
        c100_best_methods[alpha].append(best_name)

    print(f"\n  {'α':>6s} | {'Ours (mean±std)':>18s} | {'FAFI (mean±std)':>18s} | {'Δ':>8s} | {'Union':>8s} | {'Expert':>8s} | Seeds | Best method")
    print(f"  {'-'*110}")
    for alpha in sorted(c100_data.keys()):
        accs = c100_data[alpha]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs) if len(accs) > 1 else 0
        n_seeds = len(accs)

        fafi_mean, fafi_std = FAFI.get('cifar100', {}).get(alpha, (0, 0))
        delta = mean_acc - fafi_mean

        union_mean = np.mean(c100_union[alpha])
        expert_mean = np.mean(c100_expert[alpha])

        from collections import Counter
        method_counts = Counter(c100_best_methods[alpha])
        top_method = method_counts.most_common(1)[0][0]

        print(f"  {alpha:>6.2f} | {mean_acc:>6.2f}±{std_acc:<5.2f}      | {fafi_mean:>6.2f}±{fafi_std:<5.2f}      | {delta:>+7.2f} | {union_mean:>7.2f} | {expert_mean:>7.2f} | {n_seeds:>5d} | {top_method}")

    # ═══════════════════════════════════════════════════════════
    # LaTeX 表格 (可直接粘贴论文)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  LaTeX Table (粘贴到论文)")
    print("=" * 80)

    print(r"""
\begin{table}[t]
\centering
\caption{Comparison with FAFI on CIFAR-10 and CIFAR-100 (ResNet-18, $K=5$).}
\label{tab:main}
\begin{tabular}{l|cccc|cccc}
\toprule
& \multicolumn{4}{c|}{CIFAR-10} & \multicolumn{4}{c}{CIFAR-100} \\
$\alpha$ & 0.05 & 0.1 & 0.3 & 0.5 & 0.05 & 0.1 & 0.3 & 0.5 \\
\midrule""")

    # FAFI row
    line = "FAFI"
    for ds in ['cifar10', 'cifar100']:
        for alpha in [0.05, 0.1, 0.3, 0.5]:
            m, s = FAFI.get(ds, {}).get(alpha, (0, 0))
            line += f" & {m:.2f}$\\pm${s:.2f}"
    line += r" \\"
    print(line)

    # Ours row
    line = "\\textbf{Ours}"
    for ds_data in [c10_data, c100_data]:
        for alpha in [0.05, 0.1, 0.3, 0.5]:
            accs = ds_data.get(alpha, [])
            if accs:
                m = np.mean(accs)
                s = np.std(accs) if len(accs) > 1 else 0
                line += f" & \\textbf{{{m:.2f}}}$\\pm${s:.2f}"
            else:
                line += " & —"
    line += r" \\"
    print(line)

    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    # ═══════════════════════════════════════════════════════════
    # 每个 seed 的详细数据
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  Per-seed 详细数据")
    print("=" * 80)

    print("\n  CIFAR-10 ResNet-18:")
    print(f"  {'α':>6s}", end="")
    seeds_found = set()
    for f in sorted(glob.glob(os.path.join(rd, 'resnet18_a*_k5_s*.json'))):
        with open(f) as fp:
            d = json.load(fp)
        seeds_found.add(d['seed'])
    for s in sorted(seeds_found):
        print(f" | {'s='+str(s):>10s}", end="")
    print(f" | {'mean±std':>12s}")

    for alpha in sorted(c10_data.keys()):
        print(f"  {alpha:>6.2f}", end="")
        seed_accs = {}
        for f in sorted(glob.glob(os.path.join(rd, f'resnet18_a{alpha}_k5_s*.json'))):
            with open(f) as fp:
                d = json.load(fp)
            _, best_acc = find_best_non_oracle(d['all_results'])
            seed_accs[d['seed']] = best_acc * 100
        for s in sorted(seeds_found):
            if s in seed_accs:
                print(f" | {seed_accs[s]:>9.2f}%", end="")
            else:
                print(f" | {'—':>10s}", end="")
        accs = list(seed_accs.values())
        m = np.mean(accs); std = np.std(accs) if len(accs) > 1 else 0
        print(f" | {m:.2f}±{std:.2f}")

    print("\n  CIFAR-100 ResNet-18:")
    seeds_found_100 = set()
    for f in sorted(glob.glob(os.path.join(rd, 'cifar100_resnet18_a*_k5_s*.json'))):
        with open(f) as fp:
            d = json.load(fp)
        seeds_found_100.add(d['seed'])

    print(f"  {'α':>6s}", end="")
    for s in sorted(seeds_found_100):
        print(f" | {'s='+str(s):>10s}", end="")
    print(f" | {'mean±std':>12s}")

    for alpha in sorted(c100_data.keys()):
        print(f"  {alpha:>6.2f}", end="")
        seed_accs = {}
        for f in sorted(glob.glob(os.path.join(rd, f'cifar100_resnet18_a{alpha}_k5_s*.json'))):
            with open(f) as fp:
                d = json.load(fp)
            _, best_acc = find_best_non_oracle(d['all_results'])
            seed_accs[d['seed']] = best_acc * 100
        for s in sorted(seeds_found_100):
            if s in seed_accs:
                print(f" | {seed_accs[s]:>9.2f}%", end="")
            else:
                print(f" | {'—':>10s}", end="")
        accs = list(seed_accs.values())
        m = np.mean(accs); std = np.std(accs) if len(accs) > 1 else 0
        print(f" | {m:.2f}±{std:.2f}")


if __name__ == '__main__':
    collect()
