#!/usr/bin/env python3
"""RGA noise-cancellation diagnostics on Fed-ISIC2019.

Inference only. Loads the trained per-center ETF backbones, standardizes each
client's features per-feature over the test set (Eq. 4), and measures the
Proposition-1 residual quantities per class, exactly matching
ETF-pesuade/eval_residual_diag.py so the numbers are comparable to CIFAR.
"""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from medmnist_fedsra import generate_etf
from fedisic_fedsra import Backbone, IsicParquetDataset, build_transform, make_loader, N_CLASSES

EPS = 1e-8


def per_class_diag(H, labels, etf_n, ccc, w, NL, NC):
    W = sum(w.values())
    per = []
    for c in range(NL):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        if idx.numel() < 5:
            continue
        e_c = etf_n[c]
        seen = [k for k in range(NC) if H[k] is not None and ccc.get(k, {}).get(c, 0) > 0]
        unseen = [k for k in range(NC) if H[k] is not None and ccc.get(k, {}).get(c, 0) == 0]
        signal_mu = (float(np.mean([float((H[k][idx] @ e_c).mean()) for k in seen])) if seen else None)
        bias_ratio, bias_off, cent = [], [], {}
        for k in unseen:
            fk = H[k][idx]; b = fk.mean(0); rms = float(fk.pow(2).sum(1).mean().sqrt())
            bias_ratio.append(float(b.norm()) / (rms + EPS))
            cos = F.normalize(b, dim=0) @ etf_n.T; cos[c] = -9.0; bias_off.append(float(cos.max()))
            cent[k] = fk - b
        rho_raw, rho_cent = [], []
        for j, k in itertools.combinations(unseen, 2):
            fj, fk = H[j][idx], H[k][idx]
            den = (fj.pow(2).sum(1).mean().sqrt() * fk.pow(2).sum(1).mean().sqrt())
            rho_raw.append(float((fj * fk).sum(1).mean() / (den + EPS)))
            cj, ck = cent[j], cent[k]
            den2 = (cj.pow(2).sum(1).mean().sqrt() * ck.pow(2).sum(1).mean().sqrt())
            rho_cent.append(float((cj * ck).sum(1).mean() / (den2 + EPS)))
        wS = sum(w[k] for k in seen if k in w)
        per.append({'c': c, 'n_seen': len(seen), 'n_unseen': len(unseen), 'signal_mu': signal_mu,
                    'bias_ratio': float(np.mean(bias_ratio)) if bias_ratio else None,
                    'bias_offtarget': float(np.mean(bias_off)) if bias_off else None,
                    'rho_raw': float(np.mean(rho_raw)) if rho_raw else None,
                    'rho_centered': float(np.mean(rho_cent)) if rho_cent else None,
                    'seen_frac': (wS / W) if W > 0 else None})
    def agg(key):
        v = [p[key] for p in per if p[key] is not None]; return float(np.mean(v)) if v else None
    return {k: agg(k) for k in ['signal_mu','bias_ratio','bias_offtarget','rho_raw','rho_centered','seen_frac']}, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True, help="dir with center_{0..5}.pt")
    ap.add_argument("--train_parquet", type=Path, required=True)
    ap.add_argument("--test_parquet", type=Path, required=True)
    ap.add_argument("--image_size", type=int, default=144)
    ap.add_argument("--feature_dim", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df = pd.read_parquet(args.train_parquet); test_df = pd.read_parquet(args.test_parquet)
    centers = sorted(int(c) for c in train_df["center"].unique())
    NC = len(centers)
    ccc = {i: {int(l): int(n) for l, n in train_df[train_df["center"] == centers[i]]["label"].value_counts().items()}
           for i in range(NC)}
    etf = generate_etf(N_CLASSES, args.feature_dim, 42); etf_n = F.normalize(etf, dim=1)

    ds = IsicParquetDataset(test_df.reset_index(drop=True), build_transform(args.image_size, False))
    loader = make_loader(ds, 128, 4)
    all_raw = [None] * NC; labels = None
    for i in range(NC):
        m = Backbone(args.feature_dim, pretrained=False)
        saved = torch.load(f"{args.ckpt_dir}/center_{centers[i]}.pt", map_location="cpu", weights_only=False)
        m.load_state_dict(saved["model"]); m.to(device).eval()
        feats, ys = [], []
        with torch.no_grad():
            for x, y in loader:
                feats.append(m.forward_raw(x.to(device)).float().cpu())
                if i == 0: ys.append(y)
        all_raw[i] = torch.cat(feats)
        if i == 0: labels = torch.cat(ys)
        m.cpu()
    H = [(f - f.mean(0, keepdim=True)) / (f.std(0, keepdim=True) + EPS) for f in all_raw]
    w = {i: float(np.sqrt(sum(ccc[i].values()))) for i in range(NC)}
    summary, per = per_class_diag(H, labels, etf_n, ccc, w, N_CLASSES, NC)
    out = {"dataset": "fed-isic2019", "n_clients": NC, "summary": summary, "per_class": per}
    json.dump(out, open(args.out, "w"), indent=2)
    def s(x): return f"{x:.3f}" if x is not None else "  -  "
    print(f"[Fed-ISIC] mu={s(summary['signal_mu'])} bias_ratio={s(summary['bias_ratio'])} "
          f"bias_off={s(summary['bias_offtarget'])} rho_cent={s(summary['rho_centered'])} "
          f"rho_raw={s(summary['rho_raw'])} seen_frac={s(summary['seen_frac'])}")
    print("saved", args.out)


if __name__ == "__main__":
    main()
