"""
measure_fednc.py — Federated Neural Collapse measurement for the
fednc_ablation 1×4 figure.

For one (dataset, α, K, seed, variant) cell, loads per-client backbones
of the given loss variant (J/I/R) and computes:

  Per-client (over the FULL test set, evaluated through each client's bb):
    NC1 = tr(Σ_W) / tr(Σ_B)     # intra/inter-class scatter ratio
    NC2 = mean_c cos(mean_features_in_class_c, ETF[c])

  Post-aggregation (only meaningful for J backbones, but recorded for all):
    GPA aggregation (znorm per-client + sqrt(n) weighted sum + post-L2):
      NC1, NC2, NCC-accuracy
    preL2 aggregation (L2 first then weighted sum, NO post-L2):
      NC1, NC2, NCC-accuracy

Output: results/fednc_meta_{dataset}_{variant}_a{α}_k{K}_s{seed}.json

Reused infrastructure:
  - rebuild8.generate_etf, eval_ablation_RIJ.{load_backbones, forward_features,
    get_test_loader_and_ccc} — same pipeline as recovery_dump.py / eval_ablation_RIJ.py
"""
import torch, torch.nn.functional as F
import numpy as np, json, os, argparse
from rebuild8 import generate_etf
from eval_ablation_RIJ import (
    get_test_loader_and_ccc, load_backbones, forward_features,
)


def l2(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-9)


def compute_nc2(features, labels, etf_n, NL):
    """NC2 = mean over classes c of cos(class_mean_c, ETF[c]).

    features: (N, FD) numpy (raw or L2-normalized; class means are L2-normalized
              before cosine so it doesn't matter)
    labels:   (N,) int numpy
    etf_n:    (NL, FD) numpy, L2-normalized rows
    """
    NL_ = etf_n.shape[0]; FD = features.shape[1]
    mu_c = np.zeros((NL_, FD), np.float32)
    seen = np.zeros(NL_, bool)
    for c in range(NL_):
        m = labels == c
        if m.any():
            mu_c[c] = features[m].mean(0); seen[c] = True
    mu_c_n = l2(mu_c)
    cos_vec = (mu_c_n * etf_n).sum(1)
    if seen.any():
        return float(cos_vec[seen].mean()), cos_vec.tolist()
    return float('nan'), cos_vec.tolist()


def compute_nc1(features, labels, NL):
    """NC1 = tr(Σ_W) / tr(Σ_B).

    Σ_W trace = (1/N) Σ_i ||f_i − μ_{y_i}||²
    Σ_B trace = (1/C) Σ_c ||μ_c − μ_global||²

    Lower = better intra-class collapse. Uses uniform class weighting for
    Σ_B (the NC literature convention), so missing classes are penalized.
    """
    N = features.shape[0]
    mu_g = features.mean(0)
    sw_tr = 0.0
    mu_c = np.zeros((NL, features.shape[1]), np.float32)
    for c in range(NL):
        m = labels == c
        if not m.any(): continue
        f_c = features[m]
        mu = f_c.mean(0)
        mu_c[c] = mu
        sw_tr += ((f_c - mu) ** 2).sum()
    sw_tr /= float(N)
    sb_tr = ((mu_c - mu_g) ** 2).sum() / float(NL)
    return float(sw_tr / (sb_tr + 1e-12))


def gpa_aggregate(all_raw, NC, N, FD, weights):
    """GPA family: znorm per client + weighted sum + post-L2."""
    feat = np.zeros((N, FD), np.float32); wsum = 0.0
    for k in range(NC):
        if all_raw[k] is None: continue
        f = all_raw[k].numpy().astype(np.float32)
        fz = (f - f.mean(0, keepdims=True)) / (f.std(0, keepdims=True) + 1e-8)
        feat += fz * weights[k]; wsum += weights[k]
    return l2(feat / max(wsum, 1e-12))


def preL2_aggregate(all_raw, NC, N, FD, weights):
    """Principle-violating: L2 per client, weighted sum, NO post-L2."""
    feat = np.zeros((N, FD), np.float32); wsum = 0.0
    for k in range(NC):
        if all_raw[k] is None: continue
        f = all_raw[k].numpy().astype(np.float32)
        feat += l2(f) * weights[k]; wsum += weights[k]
    return feat / max(wsum, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['cifar10', 'cifar100'])
    ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--K', type=int, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--variant', required=True, choices=['J', 'I', 'R'])
    ap.add_argument('--save_dir', required=True,
                    help='backbone dir, e.g. saved_models/ablation_cifar100/J_a0.05_k10_s42')
    ap.add_argument('--out', required=True, help='output JSON path')
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    NL = 10 if args.dataset == 'cifar10' else 100
    FD = 256
    NC = args.K

    etf = generate_etf(NL, FD)
    etf_np = etf.numpy().astype(np.float32)
    etf_n = l2(etf_np)

    tl, ccc = get_test_loader_and_ccc(args.dataset, NC, args.alpha, NL)
    bbs = load_backbones(args.save_dir, NC, FD)
    n_loaded = sum(1 for b in bbs if b is not None)
    print(f"[load] {n_loaded}/{NC} clients from {args.save_dir}", flush=True)

    all_raw, labels, _ = forward_features(
        bbs, [{} for _ in range(NC)], tl, etf, NC, FD, NL, use_experts=False)
    N = len(labels)

    # Per-client NC1, NC2
    pc_nc1, pc_nc2 = [], []
    for k in range(NC):
        if all_raw[k] is None:
            pc_nc1.append(None); pc_nc2.append(None); continue
        f_np = all_raw[k].numpy().astype(np.float32)
        nc1_k = compute_nc1(f_np, labels, NL)
        nc2_k, _ = compute_nc2(f_np, labels, etf_n, NL)
        pc_nc1.append(nc1_k); pc_nc2.append(nc2_k)

    # Aggregated NC1, NC2, NCC-acc under GPA vs preL2 (using sqrt(n_k) weights)
    nk = np.array([float(sum(ccc.get(k, {}).values())) for k in range(NC)], np.float32)
    w_sqrtn = np.sqrt(np.maximum(nk, 0.0)) + 1e-9

    gpa_feat = gpa_aggregate(all_raw, NC, N, FD, w_sqrtn)
    pre_feat = preL2_aggregate(all_raw, NC, N, FD, w_sqrtn)

    nc2_gpa, _ = compute_nc2(gpa_feat, labels, etf_n, NL)
    nc1_gpa = compute_nc1(gpa_feat, labels, NL)
    acc_gpa = float(((gpa_feat @ etf_n.T).argmax(1) == labels).mean())

    pre_feat_n = l2(pre_feat)
    nc2_pre, _ = compute_nc2(pre_feat_n, labels, etf_n, NL)
    nc1_pre = compute_nc1(pre_feat, labels, NL)  # raw (un-normalized) for NC1
    acc_pre = float(((pre_feat_n @ etf_n.T).argmax(1) == labels).mean())

    pc_nc1_clean = [x for x in pc_nc1 if x is not None]
    pc_nc2_clean = [x for x in pc_nc2 if x is not None]

    out = {
        'dataset': args.dataset, 'alpha': args.alpha, 'K': args.K,
        'seed': args.seed, 'variant': args.variant,
        'save_dir': args.save_dir, 'n_loaded': n_loaded,
        'NL': NL, 'FD': FD, 'N_test': N,
        'per_client': {
            'nc1': pc_nc1, 'nc2': pc_nc2,
            'nc1_mean': float(np.mean(pc_nc1_clean)) if pc_nc1_clean else None,
            'nc1_std':  float(np.std(pc_nc1_clean))  if pc_nc1_clean else None,
            'nc2_mean': float(np.mean(pc_nc2_clean)) if pc_nc2_clean else None,
            'nc2_std':  float(np.std(pc_nc2_clean))  if pc_nc2_clean else None,
        },
        'aggregated': {
            'gpa':   {'nc1': nc1_gpa, 'nc2': nc2_gpa, 'acc': acc_gpa},
            'preL2': {'nc1': nc1_pre, 'nc2': nc2_pre, 'acc': acc_pre},
        },
        'meta': {
            'aggregation_weights': 'sqrt(n_k)',
            'nc2_def': 'mean over present classes of cos(L2(mean_features_in_class_c), L2(ETF[c]))',
            'nc1_def': 'tr(Sigma_W)/tr(Sigma_B), uniform class weighting on Sigma_B',
        },
    }
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fp:
        json.dump(out, fp, indent=2)

    print(f"[{args.variant} a={args.alpha} K={args.K}] "
          f"per-client μ: NC1={out['per_client']['nc1_mean']:.4f} "
          f"NC2={out['per_client']['nc2_mean']:.4f} | "
          f"GPA: NC1={nc1_gpa:.4f} NC2={nc2_gpa:.4f} acc={acc_gpa:.4f} | "
          f"preL2: NC1={nc1_pre:.4f} NC2={nc2_pre:.4f} acc={acc_pre:.4f}",
          flush=True)
    print(f"[save] {args.out}", flush=True)


if __name__ == '__main__':
    main()
