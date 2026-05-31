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
for a in ALPHAS:
    d = np.load(f"{TMP}/decision_a{a}_k{K}.npz", allow_pickle=True)
    cos = (l2(d["rel_feat"].astype(np.float32)) @ etf.T).astype(np.float16)
    out[f"cos_{a}"] = cos
    out[f"lab_{a}"] = d["labels"].astype(np.int16)
    out[f"covacc_{a}"] = np.array([float(d["coverage"]), float(d["acc_union"])], np.float32)
    print(a, "cos", cos.shape, "acc_union", float(d["acc_union"]), flush=True)
np.savez_compressed(f"{TMP}/simplex_coords_k{K}.npz", **out)
print("saved", f"{TMP}/simplex_coords_k{K}.npz")
