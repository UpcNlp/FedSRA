"""
Cross-client representation-misalignment dump (INFERENCE ONLY).

For a (dataset, alpha, K) cell whose checkpoint dir holds client_*/backbone.pt
(WITHOUT shared ETF -> independently-trained one-shot clients), runs every
client over the FULL test set and stores, per (client, class) for classes the
client TRAINED on (seen):
  centroids[K, NL, FD]  L2-normalized feature centroid mu_{k,c}
  wvar[K, NL]           within-client within-class variance  E||fn - mu_{k,c}||^2
  seen[K, NL]           trained-on mask (Dirichlet counts > 0)
  counts[K, NL]         n_{k,c}

misalignment% (computed later) =
  mean_c CrossVar(c) / ( mean_{k,c} wvar + mean_c CrossVar(c) ) * 100,
  CrossVar(c) = E_k|| mu_{k,c} - mean_k mu_{k,c} ||^2  over clients that saw c.
"""
import os, argparse
import numpy as np, torch, torch.nn.functional as F
from rebuild8 import generate_etf, device
from eval_ablation_RIJ import load_backbones, get_test_loader_and_ccc


@torch.no_grad()
def extract_capped(bb, tl, FD, max_imgs):
    """Manual backbone forward (matches forward_features path), capped to
    max_imgs test images for speed -- centroids/within-var are stable on a
    stratified subset."""
    bb = bb.to(device).eval()
    feats, labs, n = [], [], 0
    for x, y in tl:
        x = x.to(device)
        xx = F.relu(bb.bn1(bb.conv1(x)))
        xx = bb.layer1(xx); xx = bb.layer2(xx); xx = bb.layer3(xx); xx = bb.layer4(xx)
        xx = bb.pool(xx).flatten(1)
        feats.append(bb.fc(xx).float().cpu()); labs.append(y)
        n += x.size(0)
        if n >= max_imgs:
            break
    return torch.cat(feats), torch.cat(labs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--K', type=int, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100'])
    ap.add_argument('--NL', type=int, default=100)
    ap.add_argument('--FD', type=int, default=256)
    ap.add_argument('--max_imgs', type=int, default=4000)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    NC, NL, FD = args.K, args.NL, args.FD
    etf = generate_etf(NL, FD)
    tl, ccc = get_test_loader_and_ccc(args.dataset, NC, args.alpha, NL)
    bbs = load_backbones(args.save_dir, NC, FD)
    counts = np.array([[ccc.get(k, {}).get(c, 0) for c in range(NL)]
                       for k in range(NC)], dtype=np.int32)
    seen = counts > 0

    centroids = np.full((NC, NL, FD), np.nan, dtype=np.float32)
    wvar = np.full((NC, NL), np.nan, dtype=np.float32)
    N = 0
    for k in range(NC):
        if bbs[k] is None:
            seen[k, :] = False; continue
        f, y = extract_capped(bbs[k], tl, FD, args.max_imgs)
        fn = F.normalize(f, dim=1).numpy().astype(np.float32)
        y = np.asarray(y); N = len(y)
        for c in range(NL):
            if not seen[k, c]:
                continue
            idx = y == c
            if idx.sum() < 2:
                continue
            mu = fn[idx].mean(0)
            centroids[k, c] = mu
            wvar[k, c] = ((fn[idx] - mu) ** 2).sum(1).mean()

    # quick on-node metric for sanity
    win, crs = [], []
    for c in range(NL):
        ks = [k for k in range(NC) if seen[k, c] and np.isfinite(wvar[k, c])]
        if len(ks) < 2:
            continue
        mus = centroids[ks, c]
        crs.append(((mus - mus.mean(0)) ** 2).sum(1).mean())
        win.extend(wvar[ks, c].tolist())
    W, Xc = float(np.mean(win)), float(np.mean(crs))
    pct = Xc / (W + Xc) * 100

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez_compressed(args.out, centroids=centroids.astype(np.float16),
                        wvar=wvar, seen=seen, counts=counts,
                        alpha=args.alpha, K=NC, NL=NL)
    cov = seen.sum(1).mean()
    print(f"saved {args.out}  K={NC} alpha={args.alpha} cov={cov:.1f}/{NL} "
          f"within={W:.3f} cross={Xc:.3f} misalign%={pct:.1f}", flush=True)


if __name__ == '__main__':
    main()
