#!/usr/bin/env python
"""
RGA noise-cancellation diagnostics.

Inference only. Loads the FULL ETF backbones for one (dataset, alpha, K) cell,
standardizes each client's features exactly as RGA does (per-feature z-score over
the test set, Eq.(4)), and tests whether the Proposition-1 residual model holds:
for a class-c sample, seen clients contribute mu*e_c + eps_k and unseen clients
contribute a zero-mean, weakly correlated residual eps_k.

Per class c we measure, using the STANDARDIZED features h~_k:
  signal_mu     : mean projection <h~_k, e_c> over SEEN clients   (Prop.1 mu; >0).
  bias_ratio    : ||mean_x h~_k(x)|| / rms  over UNSEEN clients. ~0 => residuals are
                  zero-mean (assumption holds); large => systematic bias.
  bias_offtarget: max_{c'!=c} cos(mean_x h~_k(x), e_c') over unseen clients. Large =>
                  unseen clients systematically point class c at ANOTHER prototype
                  (a representative failure mode: "cat -> dog").
  rho_centered  : mean pairwise correlation of the RANDOM part of unseen residuals
                  (the Prop.1 rho; should be small for cancellation to work).
  rho_raw       : same but WITHOUT removing the bias (captures shared systematic
                  error across clients).
  seen_frac     : seen-weight fraction w_S / W (drives the signal term in Prop.1).

Also reports the cell's RGA accuracy as a cross-check against the paper.
Output: results/residdiag_{dataset}_a{alpha}_k{K}_s{seed}.json
"""
import os, json, argparse, itertools
import numpy as np, torch, torch.nn.functional as F
from rebuild8 import generate_etf
from eval_ablation_RIJ import (
    load_backbones, forward_features, get_test_loader_and_ccc, znorm_sqrt_aggregate,
)

EPS = 1e-8


def standardize(all_raw, NC):
    H = []
    for k in range(NC):
        if all_raw[k] is None:
            H.append(None); continue
        f = all_raw[k]
        H.append((f - f.mean(0, keepdim=True)) / (f.std(0, keepdim=True) + EPS))
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['cifar10', 'cifar100'])
    ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--K', type=int, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    NL = 10 if args.dataset == 'cifar10' else 100
    FD = 256; NC = args.K; N = 10000
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    etf = generate_etf(NL, FD)                       # [NL, FD]
    tl, ccc = get_test_loader_and_ccc(args.dataset, NC, args.alpha, NL)
    bbs = load_backbones(args.save_dir, NC, FD)
    n_loaded = sum(b is not None for b in bbs)
    if n_loaded == 0:
        print(f"[skip] no backbones at {args.save_dir}"); return

    all_raw, labels, _ = forward_features(bbs, [{} for _ in range(NC)], tl, etf,
                                          NC, FD, NL, N)
    labels = torch.as_tensor(labels)

    feat_agg = znorm_sqrt_aggregate(all_raw, ccc, NC, N, FD)
    acc = float((feat_agg @ etf.T).argmax(1).eq(labels).float().mean())

    H = standardize(all_raw, NC)
    etf_n = F.normalize(etf, dim=1)
    w = {k: float(np.sqrt(sum(ccc.get(k, {}).values())))
         for k in range(NC) if all_raw[k] is not None}
    W = sum(w.values())

    per_class = []
    for c in range(NL):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        if idx.numel() < 5:
            continue
        e_c = etf_n[c]
        seen = [k for k in range(NC)
                if all_raw[k] is not None and ccc.get(k, {}).get(c, 0) > 0]
        unseen = [k for k in range(NC)
                  if all_raw[k] is not None and ccc.get(k, {}).get(c, 0) == 0]

        signal_mu = (float(np.mean([float((H[k][idx] @ e_c).mean()) for k in seen]))
                     if seen else None)

        bias_ratio, bias_off, cent = [], [], {}
        for k in unseen:
            fk = H[k][idx]                          # [n, FD]
            b = fk.mean(0)                          # [FD]
            rms = float(fk.pow(2).sum(1).mean().sqrt())
            bias_ratio.append(float(b.norm()) / (rms + EPS))
            cos = F.normalize(b, dim=0) @ etf_n.T   # [NL]
            cos[c] = -9.0
            bias_off.append(float(cos.max()))
            cent[k] = fk - b

        rho_raw, rho_cent = [], []
        for j, k in itertools.combinations(unseen, 2):
            fj, fk = H[j][idx], H[k][idx]
            den = (fj.pow(2).sum(1).mean().sqrt() * fk.pow(2).sum(1).mean().sqrt())
            rho_raw.append(float((fj * fk).sum(1).mean() / (den + EPS)))
            cj, ck = cent[j], cent[k]
            den2 = (cj.pow(2).sum(1).mean().sqrt() * ck.pow(2).sum(1).mean().sqrt())
            rho_cent.append(float((cj * ck).sum(1).mean() / (den2 + EPS)))

        wS = sum(w[k] for k in seen if k in w)
        per_class.append({
            'c': c, 'n_seen': len(seen), 'n_unseen': len(unseen),
            'signal_mu': signal_mu,
            'bias_ratio': float(np.mean(bias_ratio)) if bias_ratio else None,
            'bias_offtarget': float(np.mean(bias_off)) if bias_off else None,
            'rho_raw': float(np.mean(rho_raw)) if rho_raw else None,
            'rho_centered': float(np.mean(rho_cent)) if rho_cent else None,
            'seen_frac': (wS / W) if W > 0 else None,
        })

    def agg(key):
        vals = [p[key] for p in per_class if p[key] is not None]
        return float(np.mean(vals)) if vals else None

    summary = {k: agg(k) for k in
               ['signal_mu', 'bias_ratio', 'bias_offtarget',
                'rho_raw', 'rho_centered', 'seen_frac']}
    out = {'cell': {'dataset': args.dataset, 'alpha': args.alpha,
                    'K': NC, 'seed': args.seed},
           'n_clients_loaded': n_loaded, 'rga_acc': acc,
           'summary': summary, 'per_class': per_class}
    os.makedirs('results', exist_ok=True)
    path = args.out or f"results/residdiag_{args.dataset}_a{args.alpha}_k{NC}_s{args.seed}.json"
    json.dump(out, open(path, 'w'), indent=2)

    def s(x): return f"{x:.3f}" if x is not None else "  -  "
    print(f"[{args.dataset} a{args.alpha} K{NC}] acc={acc*100:5.2f}% | "
          f"mu={s(summary['signal_mu'])} bias_ratio={s(summary['bias_ratio'])} "
          f"bias_off={s(summary['bias_offtarget'])} "
          f"rho_c={s(summary['rho_centered'])} rho_raw={s(summary['rho_raw'])} "
          f"seenfrac={s(summary['seen_frac'])}")
    print("saved", path)


if __name__ == '__main__':
    main()
