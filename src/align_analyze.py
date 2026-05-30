"""
Cluster-side analysis for representation figures. Reads an align_*.npz
(per-client test features + centroids + etf) and emits a compact plotdata_*.npz
that the local matplotlib scripts consume. Heavy compute (t-SNE) lives here
because the cluster env has scikit-learn.

Outputs in plotdata:
  # (1) per-client joint t-SNE  -> cross-client alignment (color by client/class)
  pc_xy[N,2], pc_client[N], pc_label[N]
  # (2) GPA-aggregated feature t-SNE -> the space the method classifies in
  agg_xy[M,2], agg_label[M]
  # (3) C x C class-centroid cosine (global, per-client averaged over seen)
  cxc[C,C]
  # (4) alignment distributions
  a2etf_vals[...], a2etf_cls[...]   cos(centroid_{k,c}, e_c) for seen (k,c)
  xclient_vals[...]                 cos(centroid_{k,c}, centroid_{k',c})
"""
import json, argparse
import numpy as np
from sklearn.manifold import TSNE

C, FD = 10, 256
SUB_PER_CC = 30   # points per (client,class) kept for the per-client t-SNE


def l2(a, axis=-1, eps=1e-9):
    return a / (np.linalg.norm(a, axis=axis, keepdims=True) + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.inp, allow_pickle=True)
    cen = d['centroids']            # (K,C,FD)
    seen = d['seen']                # (K,C)
    etf = d['etf'].astype(np.float32)   # (C,FD)
    feats = d['feats_sub'].astype(np.float32)   # (K,M,FD)
    lab = d['labels_sub']           # (M,)
    K, M, _ = feats.shape
    ccc = json.loads(str(d['ccc']))
    n_k = np.array([sum(v.values()) for v in ccc], dtype=np.float32)  # client sizes

    # ---- (3) C x C global centroid cosine (seen-averaged) ----
    glob = np.full((C, FD), np.nan, np.float32)
    for c in range(C):
        ks = np.where(seen[:, c])[0]
        if len(ks):
            glob[c] = np.nanmean(cen[ks, c, :], axis=0)
    M_ = l2(np.nan_to_num(glob))
    cxc = (M_ @ M_.T).astype(np.float32)

    # ---- (4) alignment distributions ----
    a2etf_vals, a2etf_cls, xclient_vals = [], [], []
    for c in range(C):
        ks = [k for k in range(K) if seen[k, c] and not np.isnan(cen[k, c, 0])]
        for k in ks:
            a2etf_vals.append(float(l2(cen[k, c]) @ l2(etf[c])))
            a2etf_cls.append(c)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                xclient_vals.append(float(l2(cen[ks[i], c]) @ l2(cen[ks[j], c])))

    # ---- (1) per-client joint t-SNE (subsample per client,class) ----
    rng = np.random.RandomState(args.seed)
    X, cl, yl = [], [], []
    for k in range(K):
        for c in range(C):
            idx = np.where(lab == c)[0]
            if len(idx) == 0:
                continue
            take = idx[rng.choice(len(idx), size=min(SUB_PER_CC, len(idx)), replace=False)]
            X.append(feats[k, take]); cl += [k] * len(take); yl += [c] * len(take)
    X = np.concatenate(X); cl = np.array(cl); yl = np.array(yl)
    pc_xy = TSNE(n_components=2, init='pca', perplexity=30,
                 random_state=args.seed, metric='cosine').fit_transform(X)

    # ---- (2) GPA-aggregated feature, then t-SNE ----
    # per-client z-score over the subset, sqrt(n) weight, sum, post-L2
    zf = np.zeros_like(feats)
    for k in range(K):
        mu = feats[k].mean(0, keepdims=True)
        sd = feats[k].std(0, keepdims=True) + 1e-6
        zf[k] = (feats[k] - mu) / sd
    w = np.sqrt(n_k)[:, None, None]
    agg = (zf * w).sum(0) / w.sum()      # (M,FD)
    agg = l2(agg)
    agg_xy = TSNE(n_components=2, init='pca', perplexity=30,
                  random_state=args.seed, metric='cosine').fit_transform(agg)

    np.savez_compressed(
        args.out,
        tag=str(d['tag']), alpha=float(d['alpha']), K=K,
        pc_xy=pc_xy.astype(np.float32), pc_client=cl, pc_label=yl,
        agg_xy=agg_xy.astype(np.float32), agg_label=lab,
        cxc=cxc,
        a2etf_vals=np.array(a2etf_vals, np.float32), a2etf_cls=np.array(a2etf_cls),
        xclient_vals=np.array(xclient_vals, np.float32),
        n_k=n_k, seen=seen,
    )
    print(f"saved {args.out} | a2etf mean={np.mean(a2etf_vals):.3f} "
          f"xclient mean={np.mean(xclient_vals):.3f} "
          f"cxc offdiag={cxc[~np.eye(C,dtype=bool)].mean():.3f}", flush=True)


if __name__ == '__main__':
    main()
