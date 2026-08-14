#!/usr/bin/env python3
"""RGA noise-cancellation diagnostics on MedMNIST.

Trains K ETF clients on an incomplete-coverage Dirichlet partition, then runs the
same Proposition-1 residual diagnostics as ETF-pesuade/eval_residual_diag.py so the
numbers are comparable to CIFAR and Fed-ISIC.
"""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from medmnist_fedsra import (
    DATASET_INFO, load_npz, dirichlet_partition, MedArrayDataset, build_transform,
    generate_etf, joint_etf_loss, ResNet18Backbone, make_loader, seed_everything,
)

EPS = 1e-8


def per_class_diag(H, labels, etf_n, ccc, w, NL, NC):
    W = sum(w.values()); per = []
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


def train_etf_client(images, labels, idx, dataset, etf, device, epochs, lr, bs, workers):
    model = ResNet18Backbone(256).to(device)
    loader = make_loader(MedArrayDataset(images, labels, idx, build_transform(dataset, True)),
                         bs, True, workers, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    etf_dev = etf.to(device); amp = device.type == "cuda"
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            if len(y) < 2:
                continue
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                loss = joint_etf_loss(model.forward_raw(x), y, etf_dev, 0.1)
            loss.backward(); opt.step()
        sch.step()
    return model.cpu().eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASET_INFO))
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--n_clients", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NL = DATASET_INFO[args.dataset]["classes"]; NC = args.n_clients
    seed_everything(args.seed)
    tr_imgs, tr_lbls, te_imgs, te_lbls = load_npz(args.data, 0, 0)
    clients, counts = dirichlet_partition(tr_lbls, NC, args.alpha, NL, args.seed,
                                          min_client_size=min(32, max(2, len(tr_lbls) // (20 * NC))))
    ccc = {k: {c: int(counts[k, c]) for c in range(NL) if counts[k, c] > 0} for k in range(NC)}
    etf = generate_etf(NL, 256, 42); etf_n = F.normalize(etf, dim=1)

    models = []
    for k, idx in enumerate(clients):
        seed_everything(args.seed * 1000 + k + 17)
        models.append(train_etf_client(tr_imgs, tr_lbls, idx, args.dataset, etf, device,
                                       args.epochs, 1e-3, 256, 4))
        print(f"  client {k} trained ({len(idx)} imgs, {len(ccc[k])} classes)", flush=True)

    test_loader = make_loader(MedArrayDataset(te_imgs, te_lbls, None, build_transform(args.dataset, False)),
                              256, False, 4)
    all_raw = [None] * NC; labels = None
    for k in range(NC):
        m = models[k].to(device); feats, ys = [], []
        with torch.no_grad():
            for x, y in test_loader:
                feats.append(m.forward_raw(x.to(device)).float().cpu())
                if k == 0: ys.append(y)
        all_raw[k] = torch.cat(feats)
        if k == 0: labels = torch.cat(ys)
        m.cpu()
    H = [(f - f.mean(0, keepdim=True)) / (f.std(0, keepdim=True) + EPS) for f in all_raw]
    w = {k: float(np.sqrt(sum(ccc[k].values()))) for k in range(NC)}
    summary, per = per_class_diag(H, labels, etf_n, ccc, w, NL, NC)
    json.dump({"dataset": args.dataset, "alpha": args.alpha, "n_clients": NC,
               "summary": summary, "per_class": per}, open(args.out, "w"), indent=2)
    def s(x): return f"{x:.3f}" if x is not None else "  -  "
    print(f"[{args.dataset} a{args.alpha}] mu={s(summary['signal_mu'])} "
          f"bias_ratio={s(summary['bias_ratio'])} bias_off={s(summary['bias_offtarget'])} "
          f"rho_cent={s(summary['rho_centered'])} rho_raw={s(summary['rho_raw'])} "
          f"seen_frac={s(summary['seen_frac'])}")
    print("saved", args.out)


if __name__ == "__main__":
    main()
