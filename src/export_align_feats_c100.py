"""
Export per-client test features + class centroids for the CIFAR-100
Challenge-1 portrait figure (V106).  INFERENCE ONLY.

CIFAR-100 counterpart of export_align_feats.py: same outputs (centroids,
seen, feats_sub, labels_sub, etf, ccc/counts), but NL=100 and the test
loader / Dirichlet partition come from
eval_ablation_RIJ.get_test_loader_and_ccc('cifar100', ...), i.e. the same
helpers misalign_dump.py used for misalign_woetf_cifar100_*.npz, so the
partition matches the w/o-ETF backbones in
saved_models/ablation_woetf_cifar100/a{alpha}_k{K}_s42 (the provenance of
the paper's Fig.2(b) amplification data; backbone.pt only, W.pt ignored).

Sanity check: the printed per-client seen-class counts for a0.05 k10 must
match misalign_woetf_cifar100_a0.05_k10.npz, i.e.
[36, 36, 21, 34, 34, 32, 34, 33, 39, 100].

Usage:
  PER_CLASS_SUB=40 python export_align_feats_c100.py --alpha 0.05 --K 10 \
      --ckpt_dir saved_models/ablation_woetf_cifar100/a0.05_k10_s42 \
      --tag woetf --out tmp/align_woetf_cifar100_a0.05_k10.npz
"""
import os, json, argparse
import numpy as np
import torch
import torch.nn.functional as F

from rebuild8 import device, generate_etf
from eval_ablation_RIJ import get_test_loader_and_ccc, load_backbones

NL, FD = 100, 256
PER_CLASS_SUB = int(os.environ.get("PER_CLASS_SUB", "40"))  # test imgs/class in subset


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
    ap.add_argument('--tag', default='CE')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # same call order as misalign_dump.py: seed -> generate_etf -> partition
    etf = generate_etf(NL, FD)
    etf_np = etf.detach().cpu().numpy() if hasattr(etf, 'detach') else np.asarray(etf)
    tl, ccc = get_test_loader_and_ccc('cifar100', args.K, args.alpha, NL)
    bbs = load_backbones(args.ckpt_dir, args.K, FD)

    counts = np.array([[ccc.get(k, {}).get(c, 0) for c in range(NL)]
                       for k in range(args.K)], dtype=np.int32)
    seen = counts > 0
    print(f"partition seen-classes/client: {seen.sum(1).tolist()}", flush=True)

    labels_full = None
    centroids = np.full((args.K, NL, FD), np.nan, dtype=np.float32)
    feats_sub = sub_idx = labels_sub = None

    for k in range(args.K):
        if bbs[k] is None:
            print(f"  client {k}: MISSING backbone, skipped", flush=True)
            continue
        f, y = extract(bbs[k], tl)
        fn = F.normalize(f, dim=1).numpy().astype(np.float32)
        y = y.numpy()
        if labels_full is None:
            labels_full = y
            rng = np.random.RandomState(0)
            idx = [rng.choice(np.where(y == c)[0],
                              size=min(PER_CLASS_SUB, int((y == c).sum())),
                              replace=False) for c in range(NL)]
            sub_idx = np.concatenate(idx)
            labels_sub = y[sub_idx]
            feats_sub = np.zeros((args.K, len(sub_idx), FD), dtype=np.float16)
        for c in range(NL):
            m = (labels_full == c)
            centroids[k, c] = fn[m].mean(0)
        feats_sub[k] = fn[sub_idx].astype(np.float16)
        bbs[k] = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  client {k}: extracted {fn.shape}, seen {int(seen[k].sum())} classes", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        tag=args.tag, alpha=args.alpha, K=args.K,
        centroids=centroids, seen=seen, counts=counts,
        feats_sub=feats_sub, labels_sub=labels_sub,
        etf=etf_np.astype(np.float32),
        ccc=json.dumps([{int(c): int(n) for c, n in ccc.get(k, {}).items()}
                        for k in range(args.K)]),
    )
    print(f"saved {args.out}  centroids{centroids.shape} feats_sub{feats_sub.shape}", flush=True)


if __name__ == '__main__':
    main()
