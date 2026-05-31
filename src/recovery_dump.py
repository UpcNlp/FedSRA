"""
Correct-pipeline numbers for the figure's panel (b): single-client vs GPA-aggregated
own-prototype cosine, under equal / sqrt(n) / linear-n weights. Raw fc feature
(eval_ablation_RIJ.forward_features), NOT the double-normalized feats_sub.
"""
import json, argparse
import numpy as np, torch
from rebuild8 import prepare_data, generate_etf
from eval_ablation_RIJ import load_backbones, forward_features

NL, FD = 10, 256


def l2(a, ax=-1):
    return a / (np.linalg.norm(a, axis=ax, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save_dir', required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    etf = generate_etf(NL, FD); etfn = l2(etf.numpy().astype(np.float32))
    _, _, tl, ccc = prepare_data(args.K, args.alpha, NL)
    bbs = load_backbones(args.save_dir, args.K, FD)
    all_raw, labels, _ = forward_features(bbs, [{} for _ in range(args.K)], tl, etf, args.K, FD, NL, use_experts=False)
    N = len(labels); ey = etfn[labels]
    nk = np.array([sum(ccc.get(k, {}).values()) for k in range(args.K)], np.float32)

    # single-client own-prototype cosine (mean over clients & samples)
    sv = []
    for k in range(args.K):
        if all_raw[k] is None: continue
        f = l2(all_raw[k].numpy().astype(np.float32))
        sv.append((f * ey).sum(1).mean())
    single = float(np.mean(sv))

    def agg_own(weights):
        feat = np.zeros((N, FD), np.float32); wsum = 0.0
        for k in range(args.K):
            if all_raw[k] is None: continue
            f = all_raw[k].numpy().astype(np.float32)
            fz = (f - f.mean(0, keepdims=True)) / (f.std(0, keepdims=True) + 1e-8)
            feat += fz * weights[k]; wsum += weights[k]
        fa = l2(feat / wsum)
        return float((fa * ey).sum(1).mean())

    equal = agg_own(np.ones(args.K))
    sqrtn = agg_own(np.sqrt(nk))
    lin = agg_own(nk)
    print(f"a={args.alpha} K={args.K} | single={single:.4f} equal={equal:.4f} sqrtn={sqrtn:.4f} linear={lin:.4f}", flush=True)


if __name__ == '__main__':
    main()
