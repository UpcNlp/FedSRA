#!/usr/bin/env python3
"""Incomplete-NC degradation K-sweep on MedMNIST.

Trains K ETF clients, GPA-aggregates (per-client z-score + sqrt(n) weighted sum +
post-L2), and computes the aggregated NC1/NC2 exactly as
ETF-pesuade/measure_fednc.py, to locate the NC1>1 degradation threshold on a medical
dataset and compare it to CIFAR-100's K~20.
"""
from __future__ import annotations
import argparse, json
import numpy as np, torch, torch.nn.functional as F
from medmnist_fedsra import (
    DATASET_INFO, load_npz, dirichlet_partition, MedArrayDataset, build_transform,
    generate_etf, joint_etf_loss, ResNet18Backbone, make_loader, seed_everything,
)


def l2(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-9)


def compute_nc1(features, labels, NL):
    N = features.shape[0]; mu_g = features.mean(0); sw_tr = 0.0
    mu_c = np.zeros((NL, features.shape[1]), np.float32)
    for c in range(NL):
        m = labels == c
        if not m.any():
            continue
        f_c = features[m]; mu = f_c.mean(0); mu_c[c] = mu
        sw_tr += ((f_c - mu) ** 2).sum()
    sw_tr /= float(N)
    sb_tr = ((mu_c - mu_g) ** 2).sum() / float(NL)
    return float(sw_tr / (sb_tr + 1e-12))


def compute_nc2(features, labels, etf_n, NL):
    mu_c = np.zeros((NL, features.shape[1]), np.float32); seen = np.zeros(NL, bool)
    for c in range(NL):
        m = labels == c
        if m.any():
            mu_c[c] = features[m].mean(0); seen[c] = True
    cos_vec = (l2(mu_c) * etf_n).sum(1)
    return float(cos_vec[seen].mean()) if seen.any() else float("nan")


def gpa_aggregate(all_raw, NC, N, FD, weights):
    feat = np.zeros((N, FD), np.float32); wsum = 0.0
    for k in range(NC):
        if all_raw[k] is None:
            continue
        f = all_raw[k].numpy().astype(np.float32)
        fz = (f - f.mean(0, keepdims=True)) / (f.std(0, keepdims=True) + 1e-8)
        feat += fz * weights[k]; wsum += weights[k]
    return l2(feat / max(wsum, 1e-12))


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
    ap.add_argument("--data", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--min_size", type=int, default=0, help="override min client size (0=auto)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NL = DATASET_INFO[args.dataset]["classes"]; NC = args.K
    seed_everything(args.seed)
    tr_imgs, tr_lbls, te_imgs, te_lbls = load_npz(args.data, 0, 0)
    msize = args.min_size if args.min_size > 0 else min(32, max(2, len(tr_lbls) // (20 * NC)))
    clients, counts = dirichlet_partition(tr_lbls, NC, args.alpha, NL, args.seed, min_client_size=msize)
    etf = generate_etf(NL, 256, 42); etf_n = l2(etf.numpy().astype(np.float32))

    models = []
    for k, idx in enumerate(clients):
        seed_everything(args.seed * 1000 + k + 17)
        models.append(train_etf_client(tr_imgs, tr_lbls, idx, args.dataset, etf, device,
                                       args.epochs, 1e-3, 256, 4))
    print(f"  trained {NC} clients", flush=True)

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
        if k == 0: labels = torch.cat(ys).numpy()
        m.cpu()
    N = len(labels)
    nk = np.array([float(counts[k].sum()) for k in range(NC)], np.float32)
    w = np.sqrt(np.maximum(nk, 0.0)) + 1e-9
    pc_nc1 = [compute_nc1(all_raw[k].numpy().astype(np.float32), labels, NL) for k in range(NC)]
    gpa = gpa_aggregate(all_raw, NC, N, 256, w)
    nc1 = compute_nc1(gpa, labels, NL); nc2 = compute_nc2(gpa, labels, etf_n, NL)
    acc = float(((gpa @ etf_n.T).argmax(1) == labels).mean())
    out = {"dataset": args.dataset, "alpha": args.alpha, "K": NC, "seed": args.seed,
           "agg_nc1": nc1, "agg_nc2": nc2, "agg_acc": acc,
           "per_client_nc1_mean": float(np.mean(pc_nc1))}
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[{args.dataset} a{args.alpha} K{NC}] agg_NC1={nc1:.3f} NC2={nc2:.3f} "
          f"acc={acc*100:.2f}% perclient_NC1={np.mean(pc_nc1):.3f}", flush=True)
    print("saved", args.out)


if __name__ == "__main__":
    main()
