"""
Dump cos-to-ETF for the two roles, via the CORRECT eval_ablation_RIJ pipeline
(raw fc feature). For a stratified test subset (PER_CLASS/class):
  cos_single[K, M, C] : each client's per-sample feature cos to the C ETF prototypes
                        (Role 1: ERL aligns each client to the shared ETF -- but noisy)
  cos_agg[M, C]       : the GPA-aggregated feature cos to the ETF prototypes
                        (Role 2: geometric aggregation sharpens onto the prototype)
  labels[M]
"""
import os, json, argparse
import numpy as np, torch, torch.nn.functional as F
from rebuild8 import prepare_data, generate_etf
from eval_ablation_RIJ import load_backbones, forward_features, znorm_sqrt_aggregate

NL, FD = 10, 256
PER_CLASS = 80


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
    etf = generate_etf(NL, FD); etf_np = etf.numpy().astype(np.float32)
    _, _, tl, ccc = prepare_data(NC, args.alpha, NL)
    bbs = load_backbones(args.save_dir, NC, FD)
    all_raw, labels, _ = forward_features(bbs, [{} for _ in range(NC)], tl, etf, NC, FD, NL,
                                          use_experts=False)
    N = len(labels)
    feat_agg = znorm_sqrt_aggregate(all_raw, ccc, NC, N, FD).numpy().astype(np.float32)

    rng = np.random.RandomState(0)
    idx = np.concatenate([rng.choice(np.where(labels == c)[0],
                          size=min(PER_CLASS, int((labels == c).sum())), replace=False)
                          for c in range(NL)])
    lab = labels[idx]
    En = l2(etf_np)
    cos_agg = (l2(feat_agg[idx]) @ En.T).astype(np.float16)
    cos_single = np.stack([(l2(all_raw[k].numpy()[idx].astype(np.float32)) @ En.T)
                           for k in range(NC) if all_raw[k] is not None]).astype(np.float16)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, cos_single=cos_single, cos_agg=cos_agg,
                        labels=lab.astype(np.int16), alpha=args.alpha, K=NC)
    print(f"saved {args.out} single{cos_single.shape} agg{cos_agg.shape}", flush=True)


if __name__ == '__main__':
    main()
