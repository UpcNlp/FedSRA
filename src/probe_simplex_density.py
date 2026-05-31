"""Probe: ETF-anchored SIMPLEX DENSITY (no t-SNE).
Project each sample onto the polygon of the C fixed ETF prototypes via its
cosine-similarity profile (softmax barycentric), then draw per-class KDE
density. Coordinate frame = the fixed ETF anchors (what classification uses).
Density (not scatter) avoids overplotting. Compares relational feature across
heterogeneity. sklearn-free (numpy + scipy + matplotlib).
"""
import numpy as np, warnings, matplotlib
warnings.filterwarnings("ignore"); matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

TMP = "/public/home/dongshou/fedETF/ETF-pesuade/tmp"
C, BETA = 10, 6.0
ang = 2 * np.pi * np.arange(C) / C + np.pi / 2
ANCHOR = np.c_[np.cos(ang), np.sin(ang)]
cmap = plt.cm.tab10(np.arange(C))


def l2(a, ax=-1):
    return a / (np.linalg.norm(a, axis=ax, keepdims=True) + 1e-9)


def etf():
    # ETF prototypes are stored in align files; reuse one
    d = np.load(f"{TMP}/align_ERL_a0.05_k20.npz", allow_pickle=True)
    return l2(d["etf"].astype(np.float32))


ETFn = etf()


def barycentric(feat):
    cs = l2(feat.astype(np.float32)) @ ETFn.T
    w = np.exp(BETA * cs); w /= w.sum(1, keepdims=True)
    return w @ ANCHOR


CELLS = [("0.5", 20), ("0.3", 20), ("0.1", 20), ("0.05", 20)]
fig, axes = plt.subplots(1, len(CELLS), figsize=(len(CELLS) * 3.0, 3.2))
gx, gy = np.mgrid[-1.25:1.25:160j, -1.25:1.25:160j]
grid = np.vstack([gx.ravel(), gy.ravel()])
for j, (a, K) in enumerate(CELLS):
    d = np.load(f"{TMP}/decision_a{a}_k{K}.npz", allow_pickle=True)
    feat = d["rel_feat"].astype(np.float32); y = d["labels"].astype(int)
    P = barycentric(feat)
    ax = axes[j]
    # faint polygon + anchors
    poly = np.vstack([ANCHOR, ANCHOR[:1]])
    ax.plot(poly[:, 0], poly[:, 1], color="#cccccc", lw=0.7, zorder=1)
    for c in range(C):
        ax.text(ANCHOR[c, 0] * 1.16, ANCHOR[c, 1] * 1.16, f"{c}", color="#999",
                fontsize=6, ha="center", va="center")
    # per-class KDE: one filled contour level in class color
    for c in range(C):
        pts = P[y == c]
        if len(pts) < 20:
            continue
        try:
            k = gaussian_kde(pts.T, bw_method=0.25)
            z = k(grid).reshape(gx.shape)
            ax.contourf(gx, gy, z, levels=[z.max() * 0.35, z.max()],
                        colors=[cmap[c]], alpha=0.45, zorder=2)
        except Exception:
            ax.scatter(pts[:, 0], pts[:, 1], s=3, color=cmap[c], alpha=0.3)
    ax.set_title(f"relational feat | a{a} K{K}", fontsize=8.5)
    ax.set_xlim(-1.3, 1.35); ax.set_ylim(-1.3, 1.4); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(f"{TMP}/simplex_density_contact.png", dpi=120)
print("saved", f"{TMP}/simplex_density_contact.png")
