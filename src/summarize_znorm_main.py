"""
汇总主表 OURS znorm 数据
扫 ETF-pesuade/results/ 下:
  - znorm_scale_*.json       (CIFAR-10)
  - znorm_cifar100_*.json    (CIFAR-100)
  - znorm_tiny_*.json        (Tiny)
按 dataset × alpha × seed 输出表格 + mean±std
"""
import os, json, re
from collections import defaultdict
import numpy as np

RESULTS_DIR = "/home/huanbao/huanbao/ct/fedETF/ETF-pesuade/results"

PATTERNS = {
    'CIFAR-10':  (re.compile(r'^znorm_scale_a([\d.]+)_k5_s(\d+)\.json$'),     'acc_full', '0.3'),
    'CIFAR-100': (re.compile(r'^znorm_cifar100_a([\d.]+)_k5_s(\d+)\.json$'),  'acc_full', '0.3'),
    'Tiny':      (re.compile(r'^znorm_tiny_a([\d.]+)_k5_s(\d+)\.json$'),      'acc_full', '0.3'),
}

ALPHAS = [0.05, 0.1, 0.3, 0.5]
SEEDS  = [0, 42, 123]

def main():
    summary = defaultdict(lambda: defaultdict(dict))  # [dataset][alpha][seed] = acc

    for f in os.listdir(RESULTS_DIR):
        for dset, (pat, key, subkey) in PATTERNS.items():
            m = pat.match(f)
            if not m: continue
            a = float(m.group(1)); s = int(m.group(2))
            try:
                d = json.load(open(os.path.join(RESULTS_DIR, f)))
                v = d[key][subkey]
                summary[dset][a][s] = v * 100
            except Exception as e:
                print(f"  ⚠️ Failed parse {f}: {e}")

    print("=" * 80)
    print("OURS znorm 主表 (K=5)")
    print("=" * 80)

    for dset in ['CIFAR-10', 'CIFAR-100', 'Tiny']:
        print(f"\n--- {dset} ---")
        print(f"{'α':<8}{'s=0':<10}{'s=42':<10}{'s=123':<10}{'mean±std':<14}{'n':<5}")
        for a in ALPHAS:
            sd = summary[dset][a]
            cells = []
            for s in SEEDS:
                if s in sd: cells.append(f"{sd[s]:.2f}")
                else: cells.append(" ❌ ")
            vals = [sd[s] for s in SEEDS if s in sd]
            ms = f"{np.mean(vals):.2f}±{np.std(vals):.2f}" if vals else "N/A"
            print(f"a={a:<6}{cells[0]:<10}{cells[1]:<10}{cells[2]:<10}{ms:<14}{len(vals):<5}")

    # 总进度
    total_done = sum(len(summary[d][a]) for d in ['CIFAR-10','CIFAR-100','Tiny'] for a in ALPHAS)
    print(f"\n=== 总进度: {total_done}/36 cells ===")

if __name__ == '__main__':
    main()
