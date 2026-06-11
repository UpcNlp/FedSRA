"""
Filter-Merge serving: collapse the K ETF-anchored client backbones into ONE
model, so inference cost is O(1) in K instead of O(K).

This directly targets the serving-cost limitation. Instead of forwarding an input
through all K backbones and aggregating (RGA), we merge the K backbones offline by
clustering similar convolutional filters across clients (cosine threshold thr) into
a single MergedResNet18 whose per-layer width grows sub-linearly with K. At serving
time there is one forward pass and one merged BatchNorm -- which also removes the
per-client / per-batch standardization that RGA needs.

Inference-only on already-trained ETF-anchored backbones (NO retraining): merge,
then classify by nearest ETF prototype (post-L2). Reports, per thr:
  acc        : merged single-model accuracy (ETF classifier)
  params     : merged model parameter count (vs K x single-backbone for the ensemble)
  width      : merged stage channel widths C0..C3 (shows sub-linear growth)
  lat_ms/img : merged forward latency, vs the K-ensemble forward latency
against the full all-client RGA ensemble accuracy on the same test split.

Usage:
  python eval_filter_merge.py --dataset cifar100 --NL 100 --alpha 0.05 --K 50 \
      --save_dir saved_models/cifar100_a0.05_k50_s42 --thrs 0.9,0.95,0.98
"""
import os, argparse, json, time
import numpy as np, torch, torch.nn.functional as F
from rebuild8 import generate_etf, device
from resnet18_filter_merge import ResNet18Backbone, union_aggregate_resnet18
from eval_ablation_RIJ import (load_backbones, forward_features,
                               get_test_loader_and_ccc, znorm_sqrt_aggregate)


def count_params(m):
    return int(sum(p.numel() for p in m.parameters()))


@torch.no_grad()
def eval_merged(ubb, tl, etf):
    ed = etf.to(device); ubb = ubb.to(device).eval()
    correct = total = 0
    for x, y in tl:
        f = ubb(x.to(device))
        preds = (F.normalize(f, dim=1) @ ed.T).argmax(1).cpu()
        correct += (preds == y).sum().item(); total += y.size(0)
    return correct / total


@torch.no_grad()
def latency_ms_per_img(model_or_bbs, tl, ensemble=False, warmup=2, iters=6):
    """Median forward latency per image. ensemble=True times K backbones summed."""
    xb = next(iter(tl))[0].to(device)
    n = xb.size(0)
    def run():
        if ensemble:
            for bb in model_or_bbs:
                _ = bb(xb)
        else:
            _ = model_or_bbs(xb)
    for _ in range(warmup):
        run()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.time(); run()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        ts.append((time.time() - t0) / n * 1000.0)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--K', type=int, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100'])
    ap.add_argument('--NL', type=int, default=100)
    ap.add_argument('--FD', type=int, default=256)
    ap.add_argument('--thrs', default='0.9,0.95,0.98',
                    help='comma list of filter-merge cosine thresholds')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    NC, NL, FD = args.K, args.NL, args.FD
    thrs = [float(x) for x in args.thrs.split(',')]
    etf = generate_etf(NL, FD)
    tl, ccc = get_test_loader_and_ccc(args.dataset, NC, args.alpha, NL)
    bbs = load_backbones(args.save_dir, NC, FD)
    valid = [k for k in range(NC) if bbs[k] is not None]
    if not valid:
        raise SystemExit(f"No backbones found under {args.save_dir}")
    vbbs = [bbs[k] for k in valid]
    Kv = len(vbbs)

    # full all-client RGA ensemble (the baseline we want to match at O(1) cost)
    all_raw, labels, _ = forward_features(bbs, [{} for _ in range(NC)], tl, etf,
                                          NC, FD, NL, use_experts=False)
    labels = np.asarray(labels)
    feat = znorm_sqrt_aggregate(all_raw, ccc, NC, len(labels), FD)
    full_acc = float(((feat @ etf.T).argmax(1).numpy() == labels).mean())

    one_params = count_params(ResNet18Backbone(FD))
    ens_params = one_params * Kv
    for k in valid:                                  # move backbones to device for latency
        bbs[k] = bbs[k].to(device).eval()
    ens_lat = latency_ms_per_img(vbbs, tl, ensemble=True)
    for k in valid:
        bbs[k] = bbs[k].cpu()

    rows = []
    for thr in thrs:
        ubb = union_aggregate_resnet18([b.cpu() for b in vbbs], FD, thr, device)
        ubb.eval()
        acc = eval_merged(ubb, tl, etf)
        p = count_params(ubb)
        widths = [ubb.conv1.out_channels, ubb.layer2[0].conv2.out_channels,
                  ubb.layer3[0].conv2.out_channels, ubb.fc.in_features]
        lat = latency_ms_per_img(ubb, tl, ensemble=False)
        rows.append({'thr': thr, 'acc': acc, 'params': p,
                     'param_ratio': p / ens_params, 'widths': widths,
                     'lat_ms': lat, 'lat_speedup': ens_lat / lat})
        ubb.cpu(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\n[{args.dataset}] alpha={args.alpha} K={NC} | clients {Kv}/{NC}")
    print(f"  full all-client RGA ensemble : acc={full_acc * 100:6.2f}%  "
          f"params={ens_params / 1e6:7.2f}M ({Kv}x)  lat={ens_lat:6.3f} ms/img")
    print(f"  filter-merged single model:")
    print(f"    {'thr':>5s} {'acc':>8s} {'params':>10s} {'vs ens':>7s} "
          f"{'lat ms':>8s} {'speedup':>8s}  widths")
    for r in rows:
        print(f"    {r['thr']:5.2f} {r['acc'] * 100:7.2f}% {r['params'] / 1e6:8.2f}M "
              f"{r['param_ratio'] * 100:6.1f}% {r['lat_ms']:8.3f} {r['lat_speedup']:7.1f}x  {r['widths']}")

    out = args.out or f"results/filtermerge_{args.dataset}_a{args.alpha}_k{NC}_s{args.seed}.json"
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    json.dump({'dataset': args.dataset, 'alpha': args.alpha, 'K': NC, 'seed': args.seed,
               'n_clients': Kv, 'full_acc': full_acc,
               'ensemble_params': ens_params, 'ensemble_lat_ms': ens_lat,
               'single_backbone_params': one_params, 'merged': rows},
              open(out, 'w'), indent=2)
    print(f"  saved {out}")


if __name__ == '__main__':
    main()
