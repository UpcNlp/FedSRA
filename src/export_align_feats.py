"""
Export per-client test features + class centroids for representation figures
(cross-client alignment / NC / ETF geometry).  INFERENCE ONLY.

For a given (alpha, K) cell and a checkpoint dir holding client_*/backbone.pt,
runs every client's backbone over the SAME CIFAR-10 test set, then saves:
  - centroids[K, C, FD]   per-client per-class L2-normalized feature centroid
                          (computed over the FULL test set; NaN if unused)
  - seen[K, C]            whether client k observed class c in training
  - feats_sub[K, M, FD]   per-client features for a fixed stratified subset of
                          M test images (same images across clients) -- for t-SNE
  - labels_sub[M]         labels of that subset
  - etf[C, FD]            the shared simplex ETF prototypes
  - ccc (json)            per-client class counts

Usage:
  python export_align_feats.py --alpha 0.05 --K 10 \
      --ckpt_dir saved_models/a0.05_k10_s42 --tag ERL --out tmp/align_ERL_a0.05_k10.npz
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F

from rebuild8 import device, prepare_data, generate_etf
from resnet18_filter_merge import ResNet18Backbone

NL, FD = 10, 256
PER_CLASS_SUB = int(os.environ.get("PER_CLASS_SUB", "80"))  # test imgs/class for t-SNE subset


@torch.no_grad()
def extract(bb, loader):
    bb = bb.to(device).eval()
    feats, labs = [], []
    for x, y in loader:
        x = x.to(device)
        feats.append(bb(x).float().cpu())
        labs.append(y)
    return torch.cat(feats), torch.cat(labs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--K', type=int, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    etf = generate_etf(NL, FD)
    etf_np = etf.detach().cpu().numpy() if hasattr(etf, 'detach') else np.asarray(etf)
    cal, ccl, tl, ccc = prepare_data(args.K, args.alpha, NL)

    # fixed stratified subset of test indices (same across clients)
    labels_full = None
    centroids = np.full((args.K, NL, FD), np.nan, dtype=np.float32)
    seen = np.zeros((args.K, NL), dtype=bool)
    feats_sub = None
    sub_idx = None
    labels_sub = None

    for k in range(args.K):
        bb = ResNet18Backbone(FD)
        sd = torch.load(f"{args.ckpt_dir}/client_{k}/backbone.pt", map_location='cpu')
        bb.load_state_dict(sd)
        f, y = extract(bb, tl)
        fn = F.normalize(f, dim=1).numpy().astype(np.float32)
        y = y.numpy()
        if labels_full is None:
            labels_full = y
            rng = np.random.RandomState(0)
            idx = []
            for c in range(NL):
                cand = np.where(y == c)[0]
                idx.append(rng.choice(cand, size=min(PER_CLASS_SUB, len(cand)), replace=False))
            sub_idx = np.concatenate(idx)
            labels_sub = y[sub_idx]
            feats_sub = np.zeros((args.K, len(sub_idx), FD), dtype=np.float16)
        for c in range(NL):
            m = (labels_full == c)
            centroids[k, c] = fn[m].mean(0)
            seen[k, c] = (c in ccc[k])
        feats_sub[k] = fn[sub_idx].astype(np.float16)
        del bb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  client {k}: extracted {fn.shape}, seen {int(seen[k].sum())} classes", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        tag=args.tag, alpha=args.alpha, K=args.K,
        centroids=centroids, seen=seen,
        feats_sub=feats_sub, labels_sub=labels_sub,
        etf=etf_np.astype(np.float32),
        ccc=json.dumps([{int(c): int(n) for c, n in ccc[k].items()} for k in range(args.K)]),
    )
    print(f"saved {args.out}  centroids{centroids.shape} feats_sub{feats_sub.shape}", flush=True)


if __name__ == '__main__':
    main()
