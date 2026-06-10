"""
Per-client feature saliency maps (FAFI-style inter-model inconsistency).
Same image -> each client's backbone -> last-conv activation map.  Different
clients attend to different regions => inconsistent features.  INFERENCE ONLY.
Saves a few candidate images + their per-client saliency maps; pick best locally.
"""
import argparse
import numpy as np, torch, torch.nn.functional as F
from rebuild8 import device
from eval_ablation_RIJ import load_backbones, get_test_loader_and_ccc


@torch.no_grad()
def maps(bb, x):
    bb = bb.to(device).eval(); x = x.to(device)
    h = F.relu(bb.bn1(bb.conv1(x)))
    h = bb.layer1(h); h = bb.layer2(h)
    l3 = bb.layer3(h); l4 = bb.layer4(l3)
    def sal(t):
        s = t.pow(2).sum(1, keepdim=True).sqrt()        # [B,1,h,w] activation magnitude
        s = F.interpolate(s, size=(32, 32), mode="bilinear", align_corners=False)
        s = s[:, 0].cpu().numpy()
        s = (s - s.min((1, 2), keepdims=True)) / (s.max((1, 2), keepdims=True)
                                                  - s.min((1, 2), keepdims=True) + 1e-9)
        return s
    return sal(l3), sal(l4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--nclients', type=int, default=6)
    ap.add_argument('--ncand', type=int, default=8)
    ap.add_argument('--dataset', default='cifar100'); ap.add_argument('--NL', type=int, default=100)
    ap.add_argument('--alpha', type=float, default=0.05); ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--FD', type=int, default=256)
    args = ap.parse_args()
    torch.manual_seed(42); np.random.seed(42)

    tl, _ = get_test_loader_and_ccc(args.dataset, args.K, args.alpha, args.NL)
    xb, yb = next(iter(tl))
    xb = xb[:args.ncand]; yb = yb[:args.ncand].numpy()
    bbs = load_backbones(args.save_dir, args.K, args.FD)
    use = [k for k in range(args.K) if bbs[k] is not None][:args.nclients]

    l3_all, l4_all = [], []
    for k in use:
        s3, s4 = maps(bbs[k], xb)
        l3_all.append(s3); l4_all.append(s4)
    l3_all = np.stack(l3_all)   # [nclients, ncand, 32, 32]
    l4_all = np.stack(l4_all)
    np.savez_compressed(args.out, imgs=xb.numpy().astype(np.float32),
                        labels=yb, sal_l3=l3_all.astype(np.float32),
                        sal_l4=l4_all.astype(np.float32), clients=np.array(use))
    print(f"saved {args.out}  imgs{xb.shape} sal_l4{l4_all.shape} clients={use} labels={yb.tolist()}")


if __name__ == '__main__':
    main()
