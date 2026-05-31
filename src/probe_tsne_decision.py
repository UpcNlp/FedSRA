"""Probe: which DECISION representation t-SNEs into the prettiest clusters?
Loads decision_*.npz (rel_feat / rel_logits / int_logits / fused) and renders a
contact sheet with silhouette scores so we can pick the best 'cluster-like' view.
Runs on cluster (sklearn). No torch.
"""
import numpy as np, warnings, matplotlib
warnings.filterwarnings("ignore"); matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

TMP = "/public/home/dongshou/fedETF/ETF-pesuade/tmp"
CELLS = [("0.05", 10), ("0.5", 10), ("0.05", 20), ("0.3", 20), ("0.5", 20)]
REPS = ["rel_logits", "int_logits", "fused"]
cmap = plt.cm.tab10(np.arange(10))


def tsne(X):
    return TSNE(n_components=2, perplexity=30, init="pca", random_state=42, max_iter=700).fit_transform(X)


fig, axes = plt.subplots(len(CELLS), len(REPS), figsize=(len(REPS) * 3.2, len(CELLS) * 3.0))
for i, (a, K) in enumerate(CELLS):
    try:
        d = np.load(f"{TMP}/decision_a{a}_k{K}.npz", allow_pickle=True)
    except Exception as e:
        print("skip", a, K, e); continue
    y = d["labels"].astype(int)
    for j, rep in enumerate(REPS):
        X = d[rep].astype(np.float32)
        XY = tsne(X)
        try:
            sil = silhouette_score(XY, y)
        except Exception:
            sil = float("nan")
        ax = axes[i, j]
        for c in range(10):
            m = y == c
            ax.scatter(XY[m, 0], XY[m, 1], s=4, color=cmap[c], alpha=0.55, lw=0)
        ax.set_title(f"a{a} K{K} | {rep} | sil={sil:.3f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        print(f"a{a} K{K} {rep}: sil={sil:.3f}", flush=True)
fig.tight_layout()
fig.savefig(f"{TMP}/tsne_decision_contact.png", dpi=95)
print("saved", f"{TMP}/tsne_decision_contact.png")
