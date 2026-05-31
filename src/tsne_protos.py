"""
FedETF-style: t-SNE of per-client class PROTOTYPES (client k's centroid for each
class it observed). Few points (~C x clients), colored by class. Dumps 2D coords
for CE and ERL so local matplotlib can style the figure.
"""
import json, argparse
import numpy as np
from sklearn.manifold import TSNE

C = 10


def collect(path):
    d = np.load(path, allow_pickle=True)
    cen, seen = d['centroids'], d['seen']
    P, lab, cli = [], [], []
    for k in range(cen.shape[0]):
        for c in range(C):
            if seen[k, c] and not np.isnan(cen[k, c, 0]):
                v = cen[k, c]; P.append(v / (np.linalg.norm(v) + 1e-9))
                lab.append(c); cli.append(k)
    return np.array(P, np.float32), np.array(lab), np.array(cli)


ap = argparse.ArgumentParser()
ap.add_argument('--k', type=int, default=20)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', required=True)
args = ap.parse_args()

out = {}
for tag in ('CE', 'ERL'):
    P, lab, cli = collect(f"tmp/align_{tag}_a0.05_k{args.k}.npz")
    perp = max(5, min(20, (len(P) - 1) // 3))
    xy = TSNE(n_components=2, init='pca', perplexity=perp, random_state=args.seed,
              metric='cosine', early_exaggeration=12).fit_transform(P)
    out[f"{tag}_xy"] = xy.astype(np.float32)
    out[f"{tag}_lab"] = lab
    out[f"{tag}_cli"] = cli
    print(f"{tag}: n={len(P)} perp={perp}")

np.savez_compressed(args.out, **out)
print("saved", args.out)
