"""
dump_fednc_overview.py — produce data for the fednc_overview 2×3 figure.

Three settings on CIFAR-10, all rendered through the SAME 2D projection plane:
  centralized : single ETF-anchored ResNet18 trained on full CIFAR-10 (K=1)
  preL2       : 10 J-backbones (K=10) aggregated via preL2 (L2 then mean, NO post-L2)
  fedDSI      : same 10 backbones aggregated via GPA (znorm + sqrt(n) + post-L2)

For each setting we save:
  feats_2d  (N_sub, 2)  — features projected onto the shared 2D ETF plane
  labels    (N_sub,)    — class labels of those samples
  class_means_256 (10, 256) — class means in feature space
  class_means_2d  (10, 2)  — projected class means
  nc_matrix       (10, 10) — pairwise cosine of class means
  setting   (str)

The shared 2D plane is PCA-top-2 of the 10×256 ETF matrix; the ETF prototypes
themselves project to ten well-separated points in that plane. We also save
the prototype 2D coordinates and the projection matrix for reproducibility.

Output:  results/fednc_overview_{setting}_a{α}_k{K}_s{seed}.npz
Usage on cluster (after centralized model is trained):
  source /opt/dtk/env.sh
  HIP_VISIBLE_DEVICES=0 python -u dump_fednc_overview.py --alpha 0.05 --seed 42
"""
import argparse, os
import numpy as np
import torch
from rebuild8 import generate_etf
from eval_ablation_RIJ import (
    get_test_loader_and_ccc, load_backbones, forward_features,
)


def l2(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-9)


def pca_plane(etf_np):
    """PCA top-2 of the ETF matrix (10×256). Returns (2, 256) projection
    matrix W such that f_2d = f @ W.T."""
    centered = etf_np - etf_np.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt[:2]  # (2, 256)


def class_means(feats, labels, NL):
    means = np.zeros((NL, feats.shape[1]), np.float32)
    for c in range(NL):
        m = labels == c
        if m.any(): means[c] = feats[m].mean(0)
    return means


def cos_matrix(class_means_np):
    mu = l2(class_means_np)
    return mu @ mu.T


def aggregate_gpa(all_raw, NC, N, FD, weights):
    feat = np.zeros((N, FD), np.float32); wsum = 0.0
    for k in range(NC):
        if all_raw[k] is None: continue
        f = all_raw[k].numpy().astype(np.float32)
        fz = (f - f.mean(0, keepdims=True)) / (f.std(0, keepdims=True) + 1e-8)
        feat += fz * weights[k]; wsum += weights[k]
    return l2(feat / max(wsum, 1e-12))


def aggregate_preL2(all_raw, NC, N, FD, weights):
    feat = np.zeros((N, FD), np.float32); wsum = 0.0
    for k in range(NC):
        if all_raw[k] is None: continue
        f = all_raw[k].numpy().astype(np.float32)
        feat += l2(f) * weights[k]; wsum += weights[k]
    return feat / max(wsum, 1e-12)  # NO post-L2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n_sub', type=int, default=5000,
                    help='samples per setting to keep (uniform per class)')
    ap.add_argument('--centralized_dir',
                    default='saved_models/centralized_cifar10_s42/client_0')
    ap.add_argument('--federated_dir_template',
                    default='saved_models/a{alpha}_k{K}_s{seed}')
    ap.add_argument('--out_dir', default='results')
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    NL = 10; FD = 256
    etf_t = generate_etf(NL, FD)
    etf_np = etf_t.numpy().astype(np.float32)
    W = pca_plane(etf_np)             # (2, 256)  — shared plane
    proto_2d = etf_np @ W.T           # (10, 2)   — prototype anchors

    tl, ccc = get_test_loader_and_ccc('cifar10', args.K, args.alpha, NL)

    # ---------- common subsample (same indices across settings) ----------
    # First pass: snapshot labels and pick a stratified subset.
    rng = np.random.default_rng(args.seed)
    # We'll get labels from the first forward (using centralized) then reuse.

    os.makedirs(args.out_dir, exist_ok=True)

    def write_setting(setting, feats_full, labels):
        mus = class_means(feats_full, labels, NL)
        nc_mat = cos_matrix(mus)
        # stratified subsample n_sub/NL per class
        per_cls = args.n_sub // NL
        keep = []
        for c in range(NL):
            idxs = np.where(labels == c)[0]
            keep.append(rng.choice(idxs, size=min(per_cls, len(idxs)), replace=False))
        keep = np.concatenate(keep)
        feats_sub = feats_full[keep]
        feats_sub_n = l2(feats_sub)            # 256-D, unit-norm; figure can re-project
        feats_2d = feats_sub @ W.T             # legacy ETF-PCA 2D (back-compat)
        mus_2d = mus @ W.T
        out = os.path.join(
            args.out_dir,
            f"fednc_overview_{setting}_a{args.alpha}_k{args.K}_s{args.seed}.npz")
        np.savez_compressed(
            out,
            feats_2d=feats_2d.astype(np.float32),
            feats_256=feats_sub_n.astype(np.float32),   # NEW: per-sample L2-norm 256-D
            labels=labels[keep].astype(np.int32),
            class_means_256=mus.astype(np.float32),
            class_means_2d=mus_2d.astype(np.float32),
            nc_matrix=nc_mat.astype(np.float32),
            etf_2d=proto_2d.astype(np.float32),
            projection_W=W.astype(np.float32),
            etf_256=etf_np.astype(np.float32),
            setting=setting,
            alpha=args.alpha, K=args.K, seed=args.seed,
        )
        print(f"[save] {out}  (feats_256 shape {feats_sub_n.shape})", flush=True)
        print(f"   NC matrix diag mean = {np.diag(nc_mat).mean():.3f}, "
              f"off-diag mean = {nc_mat[~np.eye(NL, dtype=bool)].mean():.3f} "
              f"(ideal off-diag = {-1/(NL-1):.3f})", flush=True)

    # ---------- 1. centralized ----------
    print(f"\n=== Centralized (CIFAR-10, K=1, full data) ===", flush=True)
    bbs_c = load_backbones(args.centralized_dir + '/..', 1, FD)
    # ^ load_backbones expects a save_dir/client_0/backbone.pt structure; we
    # pass the *parent* of centralized_dir so it looks at client_0.
    if bbs_c[0] is None:
        raise FileNotFoundError(f"centralized backbone not found at {args.centralized_dir}/backbone.pt")
    raw_c, labels, _ = forward_features(
        bbs_c, [{}], tl, etf_t, 1, FD, NL, use_experts=False)
    feats_c = raw_c[0].numpy().astype(np.float32)
    write_setting('centralized', feats_c, labels)

    # ---------- 2 & 3. federated preL2 / GPA ----------
    fed_dir = args.federated_dir_template.format(
        alpha=args.alpha, K=args.K, seed=args.seed)
    print(f"\n=== Federated K={args.K} α={args.alpha} from {fed_dir} ===",
          flush=True)
    bbs_f = load_backbones(fed_dir, args.K, FD)
    n_loaded = sum(1 for b in bbs_f if b is not None)
    if n_loaded == 0:
        raise FileNotFoundError(f"no federated backbones in {fed_dir}")
    print(f"   loaded {n_loaded}/{args.K} clients", flush=True)
    raw_f, labels_f, _ = forward_features(
        bbs_f, [{} for _ in range(args.K)], tl, etf_t, args.K, FD, NL,
        use_experts=False)
    N = len(labels_f)
    nk = np.array([float(sum(ccc.get(k, {}).values()))
                   for k in range(args.K)], np.float32)
    w_sqrtn = np.sqrt(np.maximum(nk, 0.0)) + 1e-9

    print("Aggregating: preL2 ...", flush=True)
    feats_pre = aggregate_preL2(raw_f, args.K, N, FD, w_sqrtn)
    write_setting('preL2', feats_pre, labels_f)

    print("Aggregating: GPA  ...", flush=True)
    feats_gpa = aggregate_gpa(raw_f, args.K, N, FD, w_sqrtn)
    write_setting('fedDSI', feats_gpa, labels_f)


if __name__ == '__main__':
    main()
