"""Extract compact figure data for the ETF-anchored simplex density plot:
per cell, the cosine-to-ETF-prototype matrix (N x C) + labels. Small (~0.4MB/cell)
so it can live in the paper repo as figure data. Reads decision_*.npz.
"""
import numpy as np
TMP = "/public/home/dongshou/fedETF/ETF-pesuade/tmp"
K = 20
ALPHAS = ["0.5", "0.3", "0.1", "0.05"]


def l2(a, ax=-1):
    return a / (np.linalg.norm(a, axis=ax, keepdims=True) + 1e-9)


etf = l2(np.load(f"{TMP}/align_ERL_a0.05_k20.npz", allow_pickle=True)["etf"].astype(np.float32))
out = {"etf_dim": etf.shape}
# ERL across alpha + a CE (w/o ETF) baseline at alpha=0.05
SOURCES = [(f"decision_a{a}_k{K}.npz", a) for a in ALPHAS] + [("decision_CE_a0.05_k20.npz", "CE")]
for fname, key in SOURCES:
    d = np.load(f"{TMP}/{fname}", allow_pickle=True)
    cos = (l2(d["rel_feat"].astype(np.float32)) @ etf.T).astype(np.float16)
    out[f"cos_{key}"] = cos
    out[f"lab_{key}"] = d["labels"].astype(np.int16)
    out[f"covacc_{key}"] = np.array([float(d["coverage"]), float(d["acc_union"])], np.float32)
    print(key, "cos", cos.shape, "acc_union", float(d["acc_union"]), flush=True)
np.savez_compressed(f"{TMP}/simplex_coords_k{K}.npz", **out)
print("saved", f"{TMP}/simplex_coords_k{K}.npz")
