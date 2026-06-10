"""Dump per-sample features for the Fig.1 UMAP on CIFAR-100 (woetf, a0.05, K=10).
Picks NSEL example classes (default 0..9), NPC test imgs each (same imgs across
clients), forwards every client's backbone -> feats_sub[K, M, FD], labels_sub[M].
INFERENCE ONLY.  Mirrors the cifar10 align_*.npz format used by V72/V78.
"""
import argparse
import numpy as np, torch, torch.nn.functional as F
from rebuild8 import generate_etf, device
from eval_ablation_RIJ import load_backbones, get_test_loader_and_ccc


@torch.no_grad()
def extract(bb, tl):
    bb = bb.to(device).eval()
    feats, labs = [], []
    for x, y in tl:
        x = x.to(device)
        xx = F.relu(bb.bn1(bb.conv1(x)))
        xx = bb.layer1(xx); xx = bb.layer2(xx); xx = bb.layer3(xx); xx = bb.layer4(xx)
        xx = bb.pool(xx).flatten(1)
        feats.append(bb.fc(xx).float().cpu()); labs.append(y)
    return torch.cat(feats).numpy(), torch.cat(labs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dataset', default='cifar100')
    ap.add_argument('--NL', type=int, default=100)
    ap.add_argument('--FD', type=int, default=256)
    ap.add_argument('--nsel', type=int, default=10)   # example classes
    ap.add_argument('--npc', type=int, default=80)    # imgs per class
    args = ap.parse_args()
    torch.manual_seed(42); np.random.seed(42)

    NC, NL, FD = args.K, args.NL, args.FD
    tl, ccc = get_test_loader_and_ccc(args.dataset, NC, args.alpha, NL)
    bbs = load_backbones(args.save_dir, NC, FD)

    feats_full, ref_y = [], None
    for k in range(NC):
        f, y = extract(bbs[k], tl)
        feats_full.append(F.normalize(torch.from_numpy(f), dim=1).numpy().astype(np.float32))
        if ref_y is None:
            ref_y = y
    ref_y = np.asarray(ref_y)

    sel_classes = list(range(args.nsel))
    sub_idx = np.concatenate([np.where(ref_y == c)[0][:args.npc] for c in sel_classes])
    feats_sub = np.stack([feats_full[k][sub_idx] for k in range(NC)])   # [K, M, FD]
    labels_sub = ref_y[sub_idx]
    counts = np.array([[ccc.get(k, {}).get(c, 0) for c in range(NL)]
                       for k in range(NC)], dtype=np.int32)
    seen = counts > 0

    np.savez_compressed(args.out, feats_sub=feats_sub.astype(np.float16),
                        labels_sub=labels_sub.astype(np.int64),
                        seen=seen, sel_classes=np.array(sel_classes),
                        alpha=args.alpha, K=NC, NL=NL)
    print(f"saved {args.out}  feats_sub{feats_sub.shape} classes={sel_classes}", flush=True)


if __name__ == '__main__':
    main()
