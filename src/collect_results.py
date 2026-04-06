"""
collect_results.py
==================
汇总所有 results/grid_a*_k*_s*.json，输出汇总表格。

用法：
  1. 把各机器上的 results/ 文件夹合并到同一个目录下
  2. python collect_results.py [--results_dir results] [--output summary.csv]

输出：
  - 终端打印 alpha × K 的汇总表（含均值±标准差）
  - summary.csv  ：每行一个实验点的原始记录
  - summary_agg.csv：按 (alpha, K) 聚合后的均值±std
  - 如果安装了 matplotlib，还会生成 summary_heatmap.png
"""

import os
import json
import glob
import argparse
from collections import defaultdict

import numpy as np

# ============================================================
# 1. 收集所有 JSON
# ============================================================
def load_all(results_dir):
    pattern = os.path.join(results_dir, "grid_a*_k*_s*.json")
    files   = sorted(glob.glob(pattern))
    records = []
    for fp in files:
        try:
            with open(fp) as f:
                d = json.load(f)
            d["file"] = os.path.basename(fp)
            records.append(d)
        except Exception as e:
            print(f"  [WARN] skip {fp}: {e}")
    return records


# ============================================================
# 2. 按 (alpha, K) 聚合
# ============================================================
def aggregate(records):
    groups = defaultdict(list)
    for r in records:
        key = (r["alpha"], r["n_clients"])
        groups[key].append(r)

    rows = []
    for (alpha, K), recs in sorted(groups.items()):
        rels = [r["acc_relational"] for r in recs if r.get("acc_relational") is not None]
        ints = [r["acc_intrinsic"]  for r in recs if r.get("acc_intrinsic")  is not None]
        gaps = [r["gap"]            for r in recs if r.get("gap")            is not None]

        row = {
            "alpha":    alpha,
            "K":        K,
            "n_seeds":  len(recs),
            "seeds":    sorted([r["seed"] for r in recs]),
            # relational
            "rel_n":    len(rels),
            "rel_mean": np.mean(rels)  if rels else None,
            "rel_std":  np.std(rels)   if len(rels) > 1 else 0.0,
            # intrinsic
            "int_n":    len(ints),
            "int_mean": np.mean(ints)  if ints else None,
            "int_std":  np.std(ints)   if len(ints) > 1 else 0.0,
            # gap
            "gap_mean": np.mean(gaps)  if gaps else None,
            "gap_std":  np.std(gaps)   if len(gaps) > 1 else 0.0,
            # status
            "missing_rel": [r["seed"] for r in recs if r.get("acc_relational") is None],
            "missing_int": [r["seed"] for r in recs if r.get("acc_intrinsic")  is None],
        }
        rows.append(row)
    return rows


# ============================================================
# 3. 终端打印
# ============================================================
def fmt(val, std=None):
    if val is None:
        return "  ---  "
    s = f"{val:.2%}"
    if std is not None and std > 0:
        s += f"±{std:.2%}"
    return s


def print_table(agg_rows):
    alphas = sorted(set(r["alpha"] for r in agg_rows))
    Ks     = sorted(set(r["K"]     for r in agg_rows))
    lookup = {(r["alpha"], r["K"]): r for r in agg_rows}

    sep = "-" * (18 + 28 * len(Ks))

    # header
    print(f"\n{'':>18s}", end="")
    for K in Ks:
        print(f"{'K=' + str(K):^28s}", end="")
    print()
    print(sep)

    # sub-header
    print(f"{'alpha':>18s}", end="")
    for K in Ks:
        print(f"{'Rel':>14s}{'Int':>14s}", end="")
    print()
    print(sep)

    for a in alphas:
        print(f"{a:>18.3f}", end="")
        for K in Ks:
            r = lookup.get((a, K))
            if r is None:
                print(f"{'---':>14s}{'---':>14s}", end="")
            else:
                print(f"{fmt(r['rel_mean'], r['rel_std']):>14s}"
                      f"{fmt(r['int_mean'], r['int_std']):>14s}", end="")
        print()

    print(sep)

    # 缺失汇总
    missing_any = False
    for r in agg_rows:
        if r["missing_rel"] or r["missing_int"]:
            if not missing_any:
                print("\n  缺失实验：")
                missing_any = True
            parts = []
            if r["missing_rel"]:
                parts.append(f"Rel seeds={r['missing_rel']}")
            if r["missing_int"]:
                parts.append(f"Int seeds={r['missing_int']}")
            print(f"    alpha={r['alpha']}, K={r['K']}  →  {', '.join(parts)}")
    if not missing_any:
        print("\n  所有实验点均已完成！")
    print()


# ============================================================
# 4. CSV 输出
# ============================================================
def save_csv(records, agg_rows, out_prefix):
    import csv

    # 原始记录
    raw_path = f"{out_prefix}.csv"
    fields   = ["alpha", "n_clients", "seed",
                "acc_relational", "acc_intrinsic", "gap",
                "epochs_ce", "epochs_ssl", "time_total", "file"]
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(records, key=lambda x: (x["alpha"], x["n_clients"], x["seed"])):
            w.writerow(r)
    print(f"  原始记录 → {raw_path}  ({len(records)} rows)")

    # 聚合表
    agg_path = f"{out_prefix}_agg.csv"
    agg_fields = ["alpha", "K", "n_seeds",
                  "rel_mean", "rel_std", "int_mean", "int_std",
                  "gap_mean", "gap_std"]
    with open(agg_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields, extrasaction="ignore")
        w.writeheader()
        for r in agg_rows:
            w.writerow(r)
    print(f"  聚合结果 → {agg_path}  ({len(agg_rows)} rows)")


# ============================================================
# 5. 热力图 (可选)
# ============================================================
def plot_heatmaps(agg_rows, out_prefix):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [INFO] matplotlib 未安装，跳过热力图")
        return

    alphas = sorted(set(r["alpha"] for r in agg_rows))
    Ks     = sorted(set(r["K"]     for r in agg_rows))
    lookup = {(r["alpha"], r["K"]): r for r in agg_rows}

    def build_matrix(key):
        mat = np.full((len(alphas), len(Ks)), np.nan)
        for i, a in enumerate(alphas):
            for j, K in enumerate(Ks):
                r = lookup.get((a, K))
                if r and r.get(key) is not None:
                    mat[i, j] = r[key]
        return mat

    mat_rel = build_matrix("rel_mean")
    mat_int = build_matrix("int_mean")
    mat_gap = build_matrix("gap_mean")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, mat, title, cmap in [
        (axes[0], mat_rel * 100, "Relational (CE) Acc %", "Blues"),
        (axes[1], mat_int * 100, "Intrinsic (SSL) Acc %", "Greens"),
        (axes[2], mat_gap * 100, "Gap (Rel - Int) %",     "RdYlGn"),
    ]:
        im = ax.imshow(mat, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(Ks)))
        ax.set_xticklabels([str(k) for k in Ks])
        ax.set_yticks(range(len(alphas)))
        ax.set_yticklabels([str(a) for a in alphas])
        ax.set_xlabel("K (clients)")
        ax.set_ylabel("alpha")
        ax.set_title(title)
        # 在格子里标注数值
        for i in range(len(alphas)):
            for j in range(len(Ks)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            fontsize=9, color="white" if abs(v) > 50 else "black")
                else:
                    ax.text(j, i, "?", ha="center", va="center",
                            fontsize=9, color="gray")
        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    fig_path = f"{out_prefix}_heatmap.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  热力图   → {fig_path}")
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="汇总 run_grid 实验结果")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="存放 grid_*.json 的目录（可传多个用逗号分隔）")
    parser.add_argument("--output",      type=str, default="summary",
                        help="输出文件前缀 (会生成 .csv, _agg.csv, _heatmap.png)")
    args = parser.parse_args()

    # 支持逗号分隔的多目录
    dirs = [d.strip() for d in args.results_dir.split(",")]

    all_records = []
    for d in dirs:
        if not os.path.isdir(d):
            print(f"  [WARN] 目录不存在: {d}")
            continue
        recs = load_all(d)
        print(f"  从 {d} 加载了 {len(recs)} 个结果文件")
        all_records.extend(recs)

    if not all_records:
        print("  未找到任何结果文件！"); return

    # 去重（同一 alpha+K+seed 取最新/更完整的）
    seen = {}
    for r in all_records:
        key = (r["alpha"], r["n_clients"], r["seed"])
        if key not in seen:
            seen[key] = r
        else:
            old = seen[key]
            # 优先保留两个 pipeline 都有结果的
            old_complete = (old.get("acc_relational") is not None) + (old.get("acc_intrinsic") is not None)
            new_complete = (r.get("acc_relational") is not None)   + (r.get("acc_intrinsic") is not None)
            if new_complete > old_complete:
                seen[key] = r
            elif new_complete == old_complete:
                # 合并：取非 None 的值
                merged = dict(old)
                if merged.get("acc_relational") is None and r.get("acc_relational") is not None:
                    merged["acc_relational"] = r["acc_relational"]
                if merged.get("acc_intrinsic") is None and r.get("acc_intrinsic") is not None:
                    merged["acc_intrinsic"] = r["acc_intrinsic"]
                if merged["acc_relational"] is not None and merged["acc_intrinsic"] is not None:
                    merged["gap"] = merged["acc_relational"] - merged["acc_intrinsic"]
                seen[key] = merged

    deduped = list(seen.values())
    print(f"\n  去重后共 {len(deduped)} 个实验点")

    agg_rows = aggregate(deduped)
    print_table(agg_rows)
    save_csv(deduped, agg_rows, args.output)
    plot_heatmaps(agg_rows, args.output)


if __name__ == "__main__":
    main()
