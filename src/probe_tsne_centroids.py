"""Probe: per-client class-CENTROID t-SNE (denoised) -- the cleanest honest
'cluster-like' alignment view. Few points (K x classes-seen), colored by class;
if ERL aligns features, same-class centroids from different clients group.
Uses cached align_*.npz centroids. sklearn, no torch.
"""
import numpy as np, warnings, matplotlib
warnings.filterwarnings("ignore"); matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

TMP = "/public/home/dongshou/fedETF/ETF-pesuade/tmp"
CELLS = [("CE", "0.05", 20), ("ERL", "0.05", 20), ("ERL", "0.05", 10), ("ERL", "0.5", 10)]
cmap = plt.cm.tab10(np.arange(10))


def l2(a, ax=-1):
    return a / (np.linalg.norm(a, axis=ax, keepdims=True) + 1e-9)


fig, axes = plt.subplots(1, len(CELLS), figsize=(len(CELLS) * 3.2, 3.3))
for j, (tag, a, K) in enumerate(CELLS):
    d = np.load(f"{TMP}/align_{tag}_a{a}_k{K}.npz", allow_pickle=True)
    cen = d["centroids"].astype(np.float32); seen = d["seen"]
    pts, labs = [], []
    for k in range(K):
        for c in range(10):
            if seen[k, c]:
                pts.append(l2(cen[k, c])); labs.append(c)
    pts = np.array(pts); labs = np.array(labs)
    perp = max(5, min(30, len(pts) // 4))
    XY = TSNE(n_components=2, perplexity=perp, init="pca", random_state=42).fit_transform(pts)
    try:
        sil = silhouette_score(XY, labs)
    except Exception:
        sil = float("nan")
    ax = axes[j]
    for c in range(10):
        m = labs == c
        ax.scatter(XY[m, 0], XY[m, 1], s=55, color=cmap[c], alpha=0.85, edgecolor="white", lw=0.4)
    ax.set_title(f"{tag} a{a} K{K} | n={len(pts)} | sil={sil:.3f}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    print(f"{tag} a{a} K{K}: n={len(pts)} sil={sil:.3f}", flush=True)
fig.tight_layout()
fig.savefig(f"{TMP}/tsne_centroid_contact.png", dpi=130)
print("saved", f"{TMP}/tsne_centroid_contact.png")
