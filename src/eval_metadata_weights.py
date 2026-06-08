"""
Experiment 3: reliability of the class-coverage metadata prior in GPA aggregation.

Inference-only. Loads trained ETF-anchored backbones (NO training), extracts each
client's features ONCE, z-scores them with global (full-test) statistics -- exactly
as the deployed method -- and then re-runs ONLY the weighted aggregation under
different client-weighting schemes. The features are shared, so the whole sweep is
near-instant.

GPA aggregates  feat = L2( sum_k w_k * z(h_k) )  and predicts by cosine to the ETF.
Only w_k changes here:

  uniform     : w_k = 1                       (no metadata -- lower anchor)
  count       : w_k = n_k                     (sample-count weight, linear)
  sqrt_count  : w_k = sqrt(n_k)               (THE DEPLOYED METHOD)
  coverage    : w_k = |S_k|                   (class-coverage |observed classes|, the paper's stated prior)

Robustness:
  coverage @ noisy-p : drop a fraction p of the observed (client,class) metadata
                       records, recompute the coverage weight, average over seeds.
                       Tests whether the prior survives imperfect metadata.

Upper bound:
  oracle (seen-only) : for each test sample, aggregate ONLY the clients that have
                       actually observed its true class (uses labels -> not
                       deployable; an upper bound on reliability-aware routing).

Usage:
  python eval_metadata_weights.py --dataset cifar10 --NL 10 --alpha 0.05 --K 10 \
      --save_dir saved_models/a0.05_k10_s42
"""
import os, argparse, json
import numpy as np, torch, torch.nn.functional as F
from rebuild8 import generate_etf
from eval_ablation_RIJ import load_backbones, forward_features, get_test_loader_and_ccc

EPS = 1e-8
SEEN_THR = 1          # a class counts as "observed" by client k if it has >= SEEN_THR samples


def zscore_global(all_raw, valid, FD, N):
    """Per-client global (full-test) z-score, shared across all weighting schemes."""
    fz = {}
    for k in valid:
        f = all_raw[k]
        fz[k] = (f - f.mean(0, keepdim=True)) / (f.std(0, keepdim=True) + EPS)
    return fz


def predict(fz, valid, w, etf):
    feat, wsum = None, 0.0
    for k in valid:
        if w[k] == 0.0:
            continue
        feat = fz[k] * w[k] if feat is None else feat + fz[k] * w[k]
        wsum += w[k]
    logits = F.normalize(feat / max(wsum, 1e-12), dim=1) @ etf.T
    return logits.argmax(1).numpy()


def make_weights(ccc, valid, scheme):
    w = {}
    for k in valid:
        cnt = ccc.get(k, {})
        n_k = float(sum(cnt.values()))
        cov = float(sum(1 for c, v in cnt.items() if v >= SEEN_THR))   # |S_k|
        if scheme == 'uniform':
            w[k] = 1.0
        elif scheme == 'count':
            w[k] = n_k
        elif scheme == 'sqrt_count':
            w[k] = float(np.sqrt(n_k))
        elif scheme == 'coverage':
            w[k] = cov
        else:
            raise ValueError(scheme)
    return w


def noisy_coverage_acc(fz, valid, ccc, etf, labels, NL, drop_frac, n_seed=5):
    """Drop drop_frac of observed (k,c) metadata records, recompute coverage weight."""
    entries = np.array([(k, c) for k in valid for c, v in ccc.get(k, {}).items()
                        if v >= SEEN_THR], dtype=np.int64)
    accs = []
    for s in range(n_seed):
        rng = np.random.RandomState(1000 + s)
        nd = int(round(drop_frac * len(entries)))
        drop = set(map(tuple, entries[rng.permutation(len(entries))[:nd]].tolist())) if nd else set()
        cor = {k: dict(ccc.get(k, {})) for k in valid}
        for (k, c) in drop:
            cor[k].pop(c, None)
        w = make_weights(cor, valid, 'coverage')
        if sum(w.values()) == 0:           # degenerate: all metadata dropped
            accs.append(float((np.zeros_like(labels) == labels).mean())); continue
        accs.append(float((predict(fz, valid, w, etf) == labels).mean()))
    return float(np.mean(accs)), float(np.std(accs))


def oracle_seen_acc(fz, valid, ccc, etf, labels, NL, FD):
    """Upper bound: per sample, aggregate only clients that have seen its true class
    (sqrt_count weight among those clients). Uses labels -> not deployable."""
    N = len(labels)
    seen = np.array([[ccc.get(k, {}).get(c, 0) >= SEEN_THR for c in range(NL)]
                     for k in valid], dtype=bool)              # [V, NL]
    feat = torch.zeros(N, FD)
    wsum = torch.zeros(N, 1)
    for idx, k in enumerate(valid):
        m = torch.tensor(seen[idx, labels].astype(np.float32)).unsqueeze(1)   # [N,1]
        wk = float(np.sqrt(sum(ccc.get(k, {}).values())))
        feat = feat + fz[k] * (wk * m)
        wsum = wsum + wk * m
    logits = F.normalize(feat / wsum.clamp(min=1e-9), dim=1) @ etf.T
    return float((logits.argmax(1).numpy() == labels).mean())


def acc_perclass(fz, valid, ccc, etf, labels, NL, FD, mode='seen', thr=SEEN_THR):
    """Per-class reliability routing from per-class counts n_{k,c} (NO labels).
    For each candidate class c, aggregate with a per-class weight a_{k,c} and score by
    cosine to e_c; predict argmax_c. This is the deployable approximation of the
    seen-only oracle, and REQUIRES per-class counts (uniform / n_k / |S_k| cannot do it).
      mode='seen'  : a_{k,c} = sqrt(n_k) if n_{k,c}>=thr else 0   (hard per-class mask)
      mode='sqrt'  : a_{k,c} = sqrt(n_{k,c})                      (soft, continuous)
      mode='linear': a_{k,c} = n_{k,c}
    """
    N = len(labels)
    En = F.normalize(etf, dim=1)                      # [C, FD]
    scores = torch.full((N, NL), -1e9)
    for c in range(NL):
        vc, wsum = None, 0.0
        for k in valid:
            nkc = ccc.get(k, {}).get(c, 0)
            if mode == 'seen':
                a = float(np.sqrt(sum(ccc[k].values()))) if nkc >= thr else 0.0
            elif mode == 'sqrt':
                a = float(np.sqrt(nkc))
            else:  # linear
                a = float(nkc)
            if a == 0.0:
                continue
            vc = fz[k] * a if vc is None else vc + fz[k] * a
            wsum += a
        if vc is not None:
            scores[:, c] = F.normalize(vc / max(wsum, 1e-12), dim=1) @ En[c]
    return float((scores.argmax(1).numpy() == labels).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--K', type=int, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    ap.add_argument('--NL', type=int, default=10)
    ap.add_argument('--FD', type=int, default=256)
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
    valid = [k for k in range(NC) if all_raw[k] is not None]
    if not valid:
        raise SystemExit(f"No backbones under {args.save_dir}")
    fz = zscore_global(all_raw, valid, FD, len(labels))

    schemes = ['uniform', 'count', 'sqrt_count', 'coverage']
    acc = {s: float((predict(fz, valid, make_weights(ccc, valid, s), etf) == labels).mean())
           for s in schemes}
    noisy = {}
    for p in (0.1, 0.2):
        m, sd = noisy_coverage_acc(fz, valid, ccc, etf, labels, NL, p)
        noisy[f'coverage_noisy{int(p*100)}'] = {'mean': m, 'std': sd}
    # per-class reliability routing from n_{k,c} (label-free oracle approximation)
    perclass = {f'perclass_{m}': acc_perclass(fz, valid, ccc, etf, labels, NL, FD, mode=m)
                for m in ('seen', 'sqrt', 'linear')}
    oracle = oracle_seen_acc(fz, valid, ccc, etf, labels, NL, FD)

    print(f"\n[{args.dataset}] alpha={args.alpha} K={NC} | clients {len(valid)}/{NC}")
    print("  per-client scalar weights:")
    for s in schemes:
        star = "  <- deployed method" if s == 'sqrt_count' else (
               "  <- paper's stated prior" if s == 'coverage' else "")
        print(f"    {s:18s}: {acc[s]*100:6.2f}%{star}")
    print("  per-class routing from n_{k,c} (label-free):")
    for kk, v in perclass.items():
        print(f"    {kk:18s}: {v*100:6.2f}%")
    print("  noisy-metadata robustness (coverage weight):")
    for kk, v in noisy.items():
        print(f"    {kk:18s}: {v['mean']*100:6.2f}% (+/-{v['std']*100:.2f})")
    print(f"    {'oracle (seen)':18s}: {oracle*100:6.2f}%  <- upper bound (uses labels)")

    out = args.out or f"results/metaw_{args.dataset}_a{args.alpha}_k{NC}_s{args.seed}.json"
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    json.dump({
        'dataset': args.dataset, 'alpha': args.alpha, 'K': NC, 'seed': args.seed,
        'n_clients': len(valid), 'acc': acc, 'perclass': perclass,
        'noisy': noisy, 'oracle': oracle,
    }, open(out, 'w'), indent=2)
    print(f"  saved {out}")


if __name__ == '__main__':
    main()
