"""
Experiment 2: sensitivity of GPA's z-score standardization to the inference batch size.

Inference-only. Loads already-trained ETF-anchored backbones (NO training) and
re-runs ONLY the aggregation under different sources of standardization statistics.

GPA standardizes each client's features with a per-feature mean/std before the
sqrt(n_k)-weighted sum + post-L2 (see eval_ablation_RIJ.znorm_sqrt_aggregate).
Those statistics are normally computed over the WHOLE test set. In online /
streaming serving only a small batch of B samples is available at a time. We
compare three ways to obtain the stats, all evaluated on the SAME held-out split
so the accuracies are directly comparable:

  batch-B : stats computed WITHIN each incoming batch of B samples
            (naive streaming -- the failure mode a reviewer will attack).
            B = eval_n reproduces the paper's "global" full-test-set number.
  frozen  : per-client mean/std estimated ONCE from a small unlabeled calibration
            pool, then frozen and applied per-sample. Batch-size independent --
            the robust deployment we propose.

Feature extraction (the only GPU-heavy step) runs once; the B sweep is pure
re-aggregation on cached features, so the whole experiment is cheap.

Usage:
  python eval_batch_zscore.py --dataset cifar10  --NL 10  --alpha 0.05 --K 10 \
      --save_dir saved_models/a0.05_k10_s42
"""
import os, argparse, json
import numpy as np, torch, torch.nn.functional as F
from rebuild8 import generate_etf
from eval_ablation_RIJ import load_backbones, forward_features, get_test_loader_and_ccc

EPS = 1e-8


def zscore(f, mu, sd):
    # nan_to_num makes the B=1 / zero-variance case well-defined: it collapses the
    # standardized feature to 0 (no usable signal), faithfully representing the
    # degeneracy of per-batch stats when the batch is too small to estimate a std.
    return torch.nan_to_num((f - mu) / (sd + EPS), nan=0.0, posinf=0.0, neginf=0.0)


def _aggregate(all_raw, valid, w, etf, idx, mu_sd=None):
    """Weighted sum of z-scored client features over rows `idx`, then post-L2 + ETF sim.
    If mu_sd is given (dict k -> (mu, sd)), use those FROZEN stats; otherwise compute
    mean/std over `idx` itself (i.e. stats scoped to this very block)."""
    it = torch.as_tensor(idx)
    feat, wsum = None, 0.0
    for k in valid:
        f = all_raw[k][it]
        if mu_sd is None:
            mu, sd = f.mean(0, keepdim=True), f.std(0, keepdim=True)
        else:
            mu, sd = mu_sd[k]
        fz = zscore(f, mu, sd) * w[k]
        feat = fz if feat is None else feat + fz
        wsum += w[k]
    logits = F.normalize(feat / max(wsum, 1e-12), dim=1) @ etf.T
    return logits.argmax(1).numpy()


def acc_batch(all_raw, valid, w, etf, labels, eval_idx, B):
    """Per-batch stats: chunk eval_idx into blocks of B, z-score within each block."""
    preds = np.empty(len(eval_idx), dtype=np.int64)
    for s in range(0, len(eval_idx), B):
        blk = eval_idx[s:s + B]
        preds[s:s + len(blk)] = _aggregate(all_raw, valid, w, etf, blk)
    return float((preds == labels[eval_idx]).mean())


def acc_frozen(all_raw, valid, w, etf, labels, eval_idx, calib_idx):
    """Frozen stats from a disjoint calibration pool; applied per-sample (B-independent)."""
    ct = torch.as_tensor(calib_idx)
    mu_sd = {k: (all_raw[k][ct].mean(0, keepdim=True),
                 all_raw[k][ct].std(0, keepdim=True)) for k in valid}
    preds = _aggregate(all_raw, valid, w, etf, eval_idx, mu_sd=mu_sd)
    return float((preds == labels[eval_idx]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--K', type=int, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    ap.add_argument('--NL', type=int, default=10, help='num classes (10 / 100)')
    ap.add_argument('--FD', type=int, default=256)
    ap.add_argument('--calib_n', type=int, default=1024,
                    help='size of the unlabeled calibration pool for frozen stats')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    NC, NL, FD = args.K, args.NL, args.FD
    etf = generate_etf(NL, FD)
    tl, ccc = get_test_loader_and_ccc(args.dataset, NC, args.alpha, NL)
    bbs = load_backbones(args.save_dir, NC, FD)
    all_raw, labels, _ = forward_features(bbs, [{} for _ in range(NC)], tl, etf,
                                          NC, FD, NL, use_experts=False)
    labels = np.asarray(labels)
    N = len(labels)
    valid = [k for k in range(NC) if all_raw[k] is not None]
    if not valid:
        raise SystemExit(f"No backbones found under {args.save_dir}")
    w = {k: float(np.sqrt(sum(ccc.get(k, {}).values()))) for k in valid}

    # deterministic calibration / evaluation split (same eval set across all modes)
    perm = np.random.RandomState(args.seed).permutation(N)
    calib_idx = perm[:args.calib_n]
    eval_idx = perm[args.calib_n:]
    ne = len(eval_idx)

    Bs = [b for b in (1, 8, 32, 128, 512) if b < ne] + [ne]   # B=ne == global full-test stats
    batch_acc = {B: acc_batch(all_raw, valid, w, etf, labels, eval_idx, B) for B in Bs}
    frozen = acc_frozen(all_raw, valid, w, etf, labels, eval_idx, calib_idx)

    print(f"\n[{args.dataset}] alpha={args.alpha} K={NC} | eval_n={ne} "
          f"calib_n={len(calib_idx)} | clients {len(valid)}/{NC}")
    print("  per-batch-stats z-score (naive streaming):")
    for B in Bs:
        tag = "full (=global)" if B == ne else f"B={B}"
        print(f"    {tag:>15s}: {batch_acc[B] * 100:6.2f}%")
    print(f"  frozen-calib z-score (any B):  {frozen * 100:6.2f}%")

    out = args.out or f"results/batchz_{args.dataset}_a{args.alpha}_k{NC}_s{args.seed}.json"
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    json.dump({
        'dataset': args.dataset, 'alpha': args.alpha, 'K': NC, 'seed': args.seed,
        'eval_n': int(ne), 'calib_n': int(len(calib_idx)),
        'batch_acc': {str(B): batch_acc[B] for B in Bs},
        'global_acc': batch_acc[ne],
        'frozen_acc': frozen,
    }, open(out, 'w'), indent=2)
    print(f"  saved {out}")


if __name__ == '__main__':
    main()
