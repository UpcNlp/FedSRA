"""Probe: which feature space gives the prettiest separable t-SNE?
Runs on cluster (sklearn, no torch). Uses cached align_*.npz features.
Candidates per cell: aggregated backbone feature (256-d) vs relational logits (C-d).
"""
import numpy as np, json, warnings, matplotlib
warnings.filterwarnings("ignore"); matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

TMP = "/public/home/dongshou/fedETF/ETF-pesuade/tmp"


def l2(a, ax=-1):
    return a / (np.linalg.norm(a, axis=ax, keepdims=True) + 1e-9)


def load(tag, K, A):
    d = np.load(f"{TMP}/align_{tag}_a{A}_k{K}.npz", allow_pickle=True)
    f = d['feats_sub'].astype(np.float32); y = d['labels_sub']; etf = d['etf'].astype(np.float32)
    nk = np.array([[v.get(str(c), 0) for c in range(10)]
                   for v in json.loads(str(d['ccc']))], np.float32)
    return f, y, etf, nk


def agg_feat(f, nk):
    zf = np.zeros_like(f)
    for k in range(f.shape[0]):
        zf[k] = (f[k] - f[k].mean(0, keepdims=True)) / (f[k].std(0, keepdims=True) + 1e-6)
    w = np.sqrt(nk.sum(1))[:, None, None]
    return l2((zf * w).sum(0) / w.sum())


def tsne(X):
    return TSNE(n_components=2, perplexity=30, init='pca', random_state=42, max_iter=700).fit_transform(X)


cands = [
    ("CE  aggfeat  a0.05 K20", "CE", 20, "0.05", "agg"),
    ("CE  rellogit a0.05 K20", "CE", 20, "0.05", "logit"),
    ("ERL aggfeat  a0.05 K20", "ERL", 20, "0.05", "agg"),
    ("ERL rellogit a0.05 K20", "ERL", 20, "0.05", "logit"),
    ("ERL aggfeat  a0.5  K10", "ERL", 10, "0.5", "agg"),
    ("ERL rellogit a0.5  K10", "ERL", 10, "0.5", "logit"),
]
cmap = plt.cm.tab10(np.arange(10))
fig, axes = plt.subplots(2, 3, figsize=(13, 8)); axes = axes.ravel()
for i, (title, tag, K, A, mode) in enumerate(cands):
    f, y, etf, nk = load(tag, K, A)
    aggf = agg_feat(f, nk)
    X = aggf if mode == "agg" else aggf @ l2(etf).T
    XY = tsne(X)
    try:
        sil = silhouette_score(XY, y)
    except Exception:
        sil = float('nan')
    ax = axes[i]
    for c in range(10):
        m = y == c
        ax.scatter(XY[m, 0], XY[m, 1], s=6, color=cmap[c], alpha=0.6, lw=0)
    ax.set_title(f"{title}  | silhouette={sil:.3f}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    print(f"{title}: silhouette={sil:.3f}", flush=True)
fig.tight_layout()
fig.savefig(f"{TMP}/tsne_contact.png", dpi=100)
print("saved", f"{TMP}/tsne_contact.png")
