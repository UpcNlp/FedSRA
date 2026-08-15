"""
Direct residual-cancellation measurement on the real Fed-ISIC federation.
Inference-only, same metric as src/eval_resid_cancel.py: for each test sample,
over the centers that did not see its true class, with r_i the center's
standardized feature and w_i = sqrt(n_i):
    R_before = sum_i w_i ||r_i||,  R_after = ||sum_i w_i r_i||,
    reduction = 1 - R_after / R_before.
"""
import argparse, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from fedisic_fedsra import N_CLASSES, Backbone, IsicParquetDataset, build_transform
from medmnist_fedsra import generate_etf
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8


@torch.no_grad()
def extract(models, loader):
    raws = []; labels = None
    for i, m in enumerate(models):
        m.to(device).eval(); feats = []; ys = []
        for x, y in loader:
            feats.append(m.forward_raw(x.to(device, non_blocking=True)).float().cpu())
            if i == 0: ys.append(y)
        raws.append(torch.cat(feats))
        if i == 0: labels = torch.cat(ys).numpy()
        m.cpu(); torch.cuda.empty_cache()
    return raws, labels


def run_seed(ckpt_dir, centers, ccc, loader, fd):
    cps = [ckpt_dir / f"center_{c}.pt" for c in centers]
    models, counts = [], []
    for p in cps:
        d = torch.load(p, map_location="cpu", weights_only=False)
        m = Backbone(fd, pretrained=False)
        m.load_state_dict({k: v.float() for k, v in d["model"].items()})
        models.append(m); counts.append(int(d["meta"]["n"]))
    raws, labels = extract(models, loader)
    NC = len(models); N = len(labels)
    H = [(r - r.mean(0, keepdim=True)) / (r.std(0, keepdim=True) + EPS) for r in raws]
    w = [float(np.sqrt(c)) for c in counts]
    etf_n = F.normalize(generate_etf(N_CLASSES, fd, 42), dim=1)

    R_before = torch.zeros(N); S_uns = torch.zeros(N, fd); S_seen = torch.zeros(N, fd)
    for i in range(NC):
        saw = set(c for c, v in ccc[centers[i]].items() if v > 0)
        U = torch.as_tensor(np.array([0.0 if lab in saw else 1.0 for lab in labels]))
        R_before += U * w[i] * H[i].norm(dim=1)
        S_uns += (U * w[i]).unsqueeze(1) * H[i]
        S_seen += ((1.0 - U) * w[i]).unsqueeze(1) * H[i]
    R_after = S_uns.norm(dim=1)
    m = R_before > EPS
    reduction = 1.0 - R_after[m] / R_before[m]
    signal = (S_seen * etf_n[torch.as_tensor(labels).long()]).sum(1)
    snr = signal[m] / (R_after[m] + EPS)
    return {"reduction_mean": float(reduction.mean()), "reduction_median": float(reduction.median()),
            "R_before": float(R_before[m].mean()), "R_after": float(R_after[m].mean()),
            "snr_mean": float(snr.mean()), "frac_with_unseen": float(m.float().mean()), "n_centers": NC}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_parquet", type=Path, required=True)
    ap.add_argument("--test_parquet", type=Path, required=True)
    ap.add_argument("--ckpt_root", type=Path, required=True)
    ap.add_argument("--seeds", default="42,123,0")
    ap.add_argument("--feature_dim", type=int, default=256)
    ap.add_argument("--image_size", type=int, default=144)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--out", default="results/fedisic_cancel.json")
    a = ap.parse_args()
    train_df = pd.read_parquet(a.train_parquet, columns=["center", "label"])
    test_df = pd.read_parquet(a.test_parquet)
    centers = sorted(int(c) for c in train_df["center"].unique())
    ccc = {}
    for c in centers:
        vc = train_df[train_df["center"] == c]["label"].value_counts()
        ccc[c] = {int(k): int(v) for k, v in vc.items()}
    ds = IsicParquetDataset(test_df, build_transform(a.image_size, False))
    tl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    rows = []
    for s in [int(x) for x in a.seeds.split(",")]:
        ck = a.ckpt_root / f"fedsra_s{s}"
        if not ck.exists(): print(f"[skip] {ck}"); continue
        r = run_seed(ck, centers, ccc, tl, a.feature_dim); r["seed"] = s
        print(f"[s{s}] centers={r['n_centers']} frac_unseen={r['frac_with_unseen']:.2f} | "
              f"reduction={r['reduction_mean']*100:.1f}% (med {r['reduction_median']*100:.1f}) | "
              f"R_before={r['R_before']:.1f} R_after={r['R_after']:.1f} SNR={r['snr_mean']:.3f}", flush=True)
        rows.append(r)
    if rows:
        red = np.array([r["reduction_mean"] for r in rows])
        print(f"  reduction: {red.mean()*100:.1f} +/- {red.std()*100:.1f} %")
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(a.out, "w"), indent=2); print(f"saved {a.out}")


if __name__ == "__main__":
    main()
