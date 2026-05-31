"""
Dump cos-to-ETF of the GPA-aggregated feature for ALL 10000 test samples, via the
CORRECT eval_ablation_RIJ pipeline (raw fc feature). For the V31 representation figure.
Works for ERL backbones and for CE ("w/o ETF") backbones (no experts needed).
"""
import os, argparse
import numpy as np, torch
from rebuild8 import prepare_data, generate_etf
from eval_ablation_RIJ import load_backbones, forward_features, znorm_sqrt_aggregate

NL, FD = 10, 256


def l2(a, ax=-1):
    return a / (np.linalg.norm(a, axis=ax, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--K', type=int, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    NC = args.K
    etf = generate_etf(NL, FD); etfn = l2(etf.numpy().astype(np.float32))
    _, _, tl, ccc = prepare_data(NC, args.alpha, NL)
    bbs = load_backbones(args.save_dir, NC, FD)
    all_raw, labels, _ = forward_features(bbs, [{} for _ in range(NC)], tl, etf, NC, FD, NL,
                                          use_experts=False)
    N = len(labels)
    feat_agg = znorm_sqrt_aggregate(all_raw, ccc, NC, N, FD).numpy().astype(np.float32)
    cos = (l2(feat_agg) @ etfn.T).astype(np.float16)
    own = (l2(feat_agg)[np.arange(N), :] * etfn[labels]).sum(1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, cos=cos, labels=labels.astype(np.int16))
    print(f"saved {args.out} cos{cos.shape} mean_align={own.mean():.3f}", flush=True)


if __name__ == '__main__':
    main()
