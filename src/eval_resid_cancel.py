"""
Direct measurement of residual cancellation in RGA (not just the conditions).

Inference-only. For each test sample, consider ONLY the clients that did not see
the sample's true class. Let r_k be that client's RGA-standardized feature and
w_k = sqrt(n_k). We measure, per sample:

    R_before = sum_k w_k * ||r_k||        (magnitudes added, no cancellation)
    R_after  = || sum_k w_k * r_k ||      (vector sum, cancellation allowed)
    reduction = 1 - R_after / R_before

reduction ~ 0  => unseen clients point the same wrong way (no cancellation)
reduction ~ 1  => unseen outputs cancel during aggregation

We also report the effective SNR per sample,
    SNR = <sum_{seen} w_k r_k, e_true> / R_after ,
and (across cells) its rank/linear correlation with accuracy.

Output: results/residcancel_{dataset}_a{alpha}_k{K}_s{seed}.json
"""
import os, argparse, json
import numpy as np, torch, torch.nn.functional as F
from rebuild8 import generate_etf
from eval_ablation_RIJ import (load_backbones, forward_features,
                               get_test_loader_and_ccc, znorm_sqrt_aggregate)
EPS = 1e-8


def standardize(all_raw, NC):
    H = []
    for k in range(NC):
        f = all_raw[k]
        if f is None:
            H.append(None); continue
        H.append((f - f.mean(0, keepdim=True)) / (f.std(0, keepdim=True) + EPS))
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--K', type=int, required=True)
    ap.add_argument('--NL', type=int, required=True)
    ap.add_argument('--FD', type=int, default=256)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    NC, NL, FD = a.K, a.NL, a.FD

    etf = generate_etf(NL, FD)
    etf_n = F.normalize(etf, dim=1)                       # [NL, FD]
    tl, ccc = get_test_loader_and_ccc(a.dataset, NC, a.alpha, NL)
    bbs = load_backbones(a.save_dir, NC, FD)
    all_raw, labels, _ = forward_features(bbs, [{} for _ in range(NC)], tl, etf,
                                          NC, FD, NL, use_experts=False)
    labels = torch.as_tensor(np.asarray(labels)).long()
    N = len(labels)
    valid = [k for k in range(NC) if all_raw[k] is not None]

    # full-aggregate accuracy (for the SNR-accuracy relation)
    feat_agg = znorm_sqrt_aggregate(all_raw, ccc, NC, N, FD)
    preds = (torch.as_tensor(feat_agg) @ etf.T).argmax(1)
    correct = preds.eq(labels).float()
    acc = float(correct.mean())

    H = standardize(all_raw, NC)
    w = {k: float(np.sqrt(sum(ccc.get(k, {}).values()))) for k in valid}
    saw = {k: set(c for c, v in ccc.get(k, {}).items() if v > 0) for k in valid}

    # per-client, per-sample: unseen mask U[k] ([N]), weighted feats, weighted norms
    R_before = torch.zeros(N)
    S_unseen = torch.zeros(N, FD)     # sum_{unseen} w_k r_k
    S_seen = torch.zeros(N, FD)       # sum_{seen}   w_k r_k
    lab = labels.numpy()
    for k in valid:
        Uk = torch.as_tensor(np.array([0.0 if c in saw[k] else 1.0 for c in lab]))  # 1 if unseen
        wk = w[k]
        Hk = H[k]
        nk = Hk.norm(dim=1)                          # ||r_k(i)||
        R_before += Uk * wk * nk
        S_unseen += (Uk * wk).unsqueeze(1) * Hk
        S_seen += ((1.0 - Uk) * wk).unsqueeze(1) * Hk

    R_after = S_unseen.norm(dim=1)                    # ||sum_{unseen} w_k r_k||
    valid_s = R_before > EPS
    reduction = torch.zeros(N)
    reduction[valid_s] = 1.0 - R_after[valid_s] / R_before[valid_s]

    # signal = <sum_{seen} w_k r_k, e_true>;  SNR = signal / R_after
    e_true = etf_n[labels]                            # [N, FD]
    signal = (S_seen * e_true).sum(1)
    snr = signal / (R_after + EPS)

    has_unseen = valid_s.numpy()
    out = {
        'dataset': a.dataset, 'alpha': a.alpha, 'K': NC, 'seed': a.seed,
        'acc': acc, 'n_eval': int(N), 'frac_with_unseen': float(has_unseen.mean()),
        'reduction_mean': float(reduction[valid_s].mean()),
        'reduction_median': float(reduction[valid_s].median()),
        'R_before_mean': float(R_before[valid_s].mean()),
        'R_after_mean': float(R_after[valid_s].mean()),
        'snr_mean': float(snr[valid_s].mean()),
        'snr_median': float(snr[valid_s].median()),
        # per-sample point-biserial: does higher SNR go with correct predictions?
        'snr_corr_pointbiserial': float(np.corrcoef(
            snr[valid_s].numpy(), correct[valid_s].numpy())[0, 1]),
    }
    print(f"[{a.dataset}] a={a.alpha} K={NC} | acc={acc*100:.2f} | "
          f"reduction={out['reduction_mean']*100:.1f}% (med {out['reduction_median']*100:.1f}) | "
          f"R_before={out['R_before_mean']:.2f} R_after={out['R_after_mean']:.2f} | "
          f"SNR={out['snr_mean']:.3f}", flush=True)
    path = a.out or f"results/residcancel_{a.dataset}_a{a.alpha}_k{NC}_s{a.seed}.json"
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    json.dump(out, open(path, 'w'), indent=2)


if __name__ == '__main__':
    main()
