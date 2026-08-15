"""
Direct residual-cancellation on PathMNIST (MedMNIST colorectal histology).
Inference-only: reuses the trained FedSRA (ETF) client backbones and the exact
deterministic Dirichlet partition to know which classes each client saw. Same
metric as src/eval_resid_cancel.py.
"""
import argparse, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from medmnist_fedsra import (ResNet18Backbone, MedArrayDataset, build_transform,
                             make_loader, dirichlet_partition, load_npz, generate_etf)
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
        if i == 0: labels = torch.cat([torch.as_tensor(t) for t in ys]).numpy()
        m.cpu(); torch.cuda.empty_cache()
    return raws, labels


def run_seed(ckpt_dir, counts, loader, NL, fd):
    NC = counts.shape[0]
    models = []
    for k in range(NC):
        d = torch.load(ckpt_dir / f"client_{k}.pt", map_location="cpu", weights_only=False)
        m = ResNet18Backbone(fd); m.load_state_dict(d["model"]); models.append(m)
    raws, labels = extract(models, loader)
    N = len(labels)
    H = [(r - r.mean(0, keepdim=True)) / (r.std(0, keepdim=True) + EPS) for r in raws]
    w = [float(np.sqrt(counts[k].sum())) for k in range(NC)]
    etf_n = F.normalize(generate_etf(NL, fd, 42), dim=1)

    R_before = torch.zeros(N); S_uns = torch.zeros(N, fd); S_seen = torch.zeros(N, fd)
    for k in range(NC):
        saw = set(int(c) for c in range(NL) if counts[k, c] > 0)
        U = torch.as_tensor(np.array([0.0 if int(lab) in saw else 1.0 for lab in labels]))
        R_before += U * w[k] * H[k].norm(dim=1)
        S_uns += (U * w[k]).unsqueeze(1) * H[k]
        S_seen += ((1.0 - U) * w[k]).unsqueeze(1) * H[k]
    R_after = S_uns.norm(dim=1)
    m = R_before > EPS
    reduction = 1.0 - R_after[m] / R_before[m]
    signal = (S_seen * etf_n[torch.as_tensor(labels).long()]).sum(1)
    snr = signal[m] / (R_after[m] + EPS)
    return {"reduction_mean": float(reduction.mean()), "reduction_median": float(reduction.median()),
            "R_before": float(R_before[m].mean()), "R_after": float(R_after[m].mean()),
            "snr_mean": float(snr.mean()), "frac_with_unseen": float(m.float().mean()), "n_clients": NC}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--dataset", default="pathmnist")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--NL", type=int, default=9)
    ap.add_argument("--feature_dim", type=int, default=256)
    ap.add_argument("--ckpt_root", type=Path, required=True)
    ap.add_argument("--seeds", default="42,123,0")
    ap.add_argument("--out", default="results/pathmnist_cancel.json")
    a = ap.parse_args()
    tr_imgs, tr_lbls, te_imgs, te_lbls = load_npz(a.data, 0, 0)
    test_ds = MedArrayDataset(te_imgs, te_lbls, None, build_transform(a.dataset, train=False))
    loader = make_loader(test_ds, 256, False, 4)
    astr = str(a.alpha).replace(".", "p")
    rows = []
    for s in [int(x) for x in a.seeds.split(",")]:
        mcs = min(32, max(2, len(tr_lbls) // (20 * a.K)))
        clients, counts = dirichlet_partition(tr_lbls, a.K, a.alpha, a.NL, s, min_client_size=mcs)
        ck = a.ckpt_root / f"{a.dataset}_a{astr}_k{a.K}_noise0p0_s{s}"
        if not ck.exists(): print(f"[skip] {ck}"); continue
        r = run_seed(ck, counts, loader, a.NL, a.feature_dim); r["seed"] = s
        print(f"[s{s}] K={r['n_clients']} frac_unseen={r['frac_with_unseen']:.2f} | "
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
