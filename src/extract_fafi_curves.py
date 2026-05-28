"""
extract_fafi_curves.py - 从 FAFI 的 round-by-round result yaml 抽 epoch sensitivity 曲线
=========================================================================================
FAFI 配置 local_epochs=1, num_rounds=N → 每个 epoch 测一次, list 每元素 = epoch acc。
读 baselines_<method>_<dataset>_results.yaml, 转成跟 epoch_analysis_joint_*.json
对得上的格式, 方便一起画图。

用法 (cluster 上):
  python extract_fafi_curves.py \
      --base /public/home/dongshou/fedETF/FAFI_ICML25-master-orgin/checkpoints \
      --dataset CIFAR10 --K 10 --out_dir results

会扫所有 CIFAR10_alpha*_K10 目录, 解析每个的 baselines yaml, 写出
  results/fafi_epoch_curve_a{ALPHA}_k{K}_s{SEED}.json
"""
import argparse, glob, json, os, re, sys
import yaml


def find_fafi_runs(base_dir, dataset, K):
    """扫 base_dir 下的 <dataset>_alpha*_K{K}* 子目录."""
    pattern = os.path.join(base_dir, f"{dataset}_alpha*_K{K}*")
    runs = sorted(glob.glob(pattern))
    return runs


def parse_alpha_from_dirname(d):
    """从目录名抽 α, e.g. CIFAR10_alpha0.05_K10 → 0.05."""
    m = re.search(r'alpha([\d.]+)', os.path.basename(d))
    return float(m.group(1)) if m else None


def parse_seed_from_dirname(d, default=42):
    m = re.search(r'[_-]s(\d+)', os.path.basename(d))
    return int(m.group(1)) if m else default


def load_fafi_yaml(yaml_path):
    """yaml 里是 PyYAML 序列化的 defaultdict, 需要 unsafe_load."""
    with open(yaml_path) as f:
        data = yaml.unsafe_load(f)
    # data: defaultdict(list, {'<method_name>': [acc per epoch, ...]})
    if not data:
        return None, None
    method_name = list(data.keys())[0]
    accs = list(data[method_name])
    return method_name, accs


def find_baseline_yaml(run_dir, dataset):
    """优先找 OneShotOurs+Ensemble (FAFI 自家方法), 找不到就拿任何 baselines_*_results.yaml."""
    candidates = [
        f"baselines_OneShotOurs+Ensemble_{dataset}_results.yaml",
        f"baselines_Ensemble_{dataset}_results.yaml",
        f"baselines_OneshotFedavg_{dataset}_results.yaml",
    ]
    found = {}
    for c in candidates:
        p = os.path.join(run_dir, c)
        if os.path.exists(p):
            # method label: 去掉前后缀
            label = c.replace(f"baselines_", "").replace(f"_{dataset}_results.yaml", "")
            found[label] = p
    # 兜底: 任何 baselines_*_results.yaml
    if not found:
        for p in glob.glob(os.path.join(run_dir, f"baselines_*_{dataset}_results.yaml")):
            label = os.path.basename(p).replace("baselines_", "").replace(f"_{dataset}_results.yaml", "")
            found[label] = p
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True, help='FAFI checkpoints 根目录')
    parser.add_argument('--dataset', default='CIFAR10')
    parser.add_argument('--K', type=int, required=True)
    parser.add_argument('--out_dir', default='results')
    parser.add_argument('--default_seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = find_fafi_runs(args.base, args.dataset, args.K)
    if not runs:
        print(f"[!] no runs found at {args.base} matching {args.dataset}_alpha*_K{args.K}*")
        sys.exit(1)

    print(f"Found {len(runs)} candidate run dir(s):")
    for r in runs:
        print(f"  {r}")

    for run_dir in runs:
        alpha = parse_alpha_from_dirname(run_dir)
        if alpha is None:
            print(f"  [skip] cannot parse alpha from: {run_dir}")
            continue
        seed = parse_seed_from_dirname(run_dir, args.default_seed)
        yamls = find_baseline_yaml(run_dir, args.dataset)
        if not yamls:
            print(f"  [skip] no baseline yaml under {run_dir}")
            continue

        for method_label, yp in yamls.items():
            method_name, accs = load_fafi_yaml(yp)
            if accs is None:
                print(f"    [skip] empty {yp}")
                continue
            n_epochs = len(accs)
            print(f"  α={alpha} seed={seed} method={method_label} → {n_epochs} epochs, "
                  f"final acc={accs[-1]*100:.2f}%")

            # 转成 epoch_analysis_joint 同结构 (snapshot_epochs + results dict)
            snapshot_epochs = list(range(1, n_epochs + 1))
            results = {
                str(ep): {
                    'union': float(accs[ep - 1]),  # FAFI 单值, 三列都填同一个数
                    'expert': float(accs[ep - 1]),
                    'full': float(accs[ep - 1]),
                    'best_alpha': None,
                } for ep in snapshot_epochs
            }
            out = {
                'experiment': 'fafi_epoch_curve',
                'source_yaml': yp,
                'method_label': method_label,
                'method_name_in_yaml': method_name,
                'dataset': args.dataset,
                'alpha': alpha, 'n_clients': args.K, 'seed': seed,
                'snapshot_epochs': snapshot_epochs,
                'n_epochs_recorded': n_epochs,
                'results': results,
            }
            # 输出文件名带 method_label, 避免 OneShotOurs+Ensemble vs Ensemble 冲突
            safe_label = method_label.replace('+', '_').replace(' ', '_')
            out_path = os.path.join(
                args.out_dir,
                f"fafi_epoch_curve_{safe_label}_a{alpha}_k{args.K}_s{seed}.json"
            )
            json.dump(out, open(out_path, 'w'), indent=2)
            print(f"    → {out_path}")


if __name__ == '__main__':
    main()
