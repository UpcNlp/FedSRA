"""
Grouped Filter-Merge serving: collapse K client backbones into G merged models
(1 <= G <= K), not a single giant one. At inference, forward through the G
group-models and aggregate them with the usual RGA (per-model z-score, sqrt-size
weight, post-L2, nearest-ETF). G is a serving-cost knob:
  G = K : no merge -- the full all-client RGA ensemble (top accuracy, K forwards)
  G = 1 : merge everything into one (cheapest #models, but widest / least accurate)
  1<G<K : a few moderate-width models -- the intended sweet spot.

Within a group, similar filters across the (few, more alike) members cluster at
cosine threshold thr; keeping diversity ACROSS groups avoids the accuracy collapse
of a global merge. Cost has two axes we report: n_models (= G forward passes) and
total params summed over the G merged models (the "model too big" concern), since
independently trained backbones whose filters do not cluster yield wide models.

Inference-only on the SAME trained ETF-anchored backbones (NO retraining). Sweeps
G x thr. Reports acc, total params (vs K x single-backbone), and n_models, against
full all-client RGA on the same split.

Usage:
  python eval_grouped_merge.py --dataset cifar100 --NL 100 --alpha 0.05 --K 50 \
      --save_dir saved_models/cifar100_a0.05_k50_s42 --groups 1,5,10,25,50 --thrs 0.7,0.85,0.95
"""
import os, argparse, json
import numpy as np, torch, torch.nn.functional as F
from rebuild8 import generate_etf, device
from resnet18_filter_merge import ResNet18Backbone, union_aggregate_resnet18
from eval_ablation_RIJ import load_backbones, get_test_loader_and_ccc

EPS = 1e-8


def count_params(m):
    return int(sum(p.numel() for p in m.parameters()))


@torch.no_grad()
def raw_features(model, tl):
    """Raw fc features (pre-normalization) from a ResNet18Backbone OR MergedResNet18."""
    model = model.to(device).eval()
    feats, labels = [], []
    for x, y in tl:
        x = x.to(device)
        h = F.relu(model.bn1(model.conv1(x)))
        h = model.layer1(h); h = model.layer2(h); h = model.layer3(h); h = model.layer4(h)
        h = model.pool(h).flatten(1)
        feats.append(model.fc(h).cpu()); labels.append(y)
    model.cpu()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return torch.cat(feats, 0), torch.cat(labels).numpy()


def make_groups(valid, G, seed):
    """Partition valid client indices into G near-equal random groups."""
    order = np.random.RandomState(seed).permutation(valid)
    return [list(order[i::G]) for i in range(G)]   # round-robin -> balanced sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--K', type=int, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100'])
    ap.add_argument('--NL', type=int, default=100)
    ap.add_argument('--FD', type=int, default=256)
    ap.add_argument('--groups', default='1,5,10,25,50', help='comma list of G values')
    ap.add_argument('--thrs', default='0.7,0.85,0.95', help='comma list of merge thresholds')
    ap.add_argument('--calib_n', type=int, default=1024)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    NC, NL, FD = args.K, args.NL, args.FD
    G_list = [g for g in (int(x) for x in args.groups.split(',')) if g <= NC]
    thrs = [float(x) for x in args.thrs.split(',')]
    etf = generate_etf(NL, FD)
    tl, ccc = get_test_loader_and_ccc(args.dataset, NC, args.alpha, NL)
    bbs = load_backbones(args.save_dir, NC, FD)
    valid = [k for k in range(NC) if bbs[k] is not None]
    if not valid:
        raise SystemExit(f"No backbones found under {args.save_dir}")
    Kv = len(valid)
    one_params = count_params(ResNet18Backbone(FD))
    ens_params = one_params * Kv
    nk = {k: float(sum(ccc.get(k, {}).values())) for k in valid}

    def rga_acc(model_feats_weights, calib_idx, eval_idx, le):
        """RGA over a list of (raw_feat[N,FD], weight); frozen calib z-score."""
        ct, et = torch.as_tensor(calib_idx), torch.as_tensor(eval_idx)
        agg = None; wsum = 0.0
        for f, w in model_feats_weights:
            mu = f[ct].mean(0, keepdim=True); sd = f[ct].std(0, keepdim=True)
            fz = torch.nan_to_num((f[et] - mu) / (sd + EPS), nan=0.0, posinf=0.0, neginf=0.0)
            agg = fz * w if agg is None else agg + fz * w; wsum += w
        z = F.normalize(agg / max(wsum, 1e-12), dim=1)
        return float(((z @ etf.T).argmax(1).numpy() == le).mean())

    # shared eval split (computed once labels are known)
    rows = []
    labels = None; calib_idx = eval_idx = le = None
    for thr in thrs:
        for G in G_list:
            groups = make_groups(valid, G, args.seed)
            feats_weights, tot_params = [], 0
            for grp in groups:
                if len(grp) == 1:
                    model = bbs[grp[0]]
                else:
                    model = union_aggregate_resnet18([bbs[k] for k in grp], FD, thr, device)
                tot_params += count_params(model)
                f, lab = raw_features(model, tl)
                if labels is None:
                    labels = lab; N = len(labels)
                    perm = np.random.RandomState(args.seed).permutation(N)
                    calib_idx, eval_idx = perm[:args.calib_n], perm[args.calib_n:]
                    le = labels[eval_idx]
                w = float(np.sqrt(sum(nk[k] for k in grp)))
                feats_weights.append((f, w))
            acc = rga_acc(feats_weights, calib_idx, eval_idx, le)
            rows.append({'thr': thr, 'G': G, 'acc': acc, 'tot_params': tot_params,
                         'param_ratio': tot_params / ens_params, 'n_models': G})
            print(f"  thr={thr:.2f} G={G:2d}: acc={acc * 100:6.2f}%  "
                  f"params={tot_params / 1e6:7.1f}M ({tot_params / ens_params * 100:5.0f}% of ens)  "
                  f"n_models={G}")

    full = max((r for r in rows if r['G'] == Kv), key=lambda r: r['acc'], default=None)
    print(f"\n[{args.dataset}] alpha={args.alpha} K={NC} clients {Kv}/{NC}")
    if full:
        print(f"  full RGA (G={Kv}, no merge): acc={full['acc'] * 100:.2f}%  "
              f"params={ens_params / 1e6:.1f}M  n_models={Kv}")

    out = args.out or f"results/groupmerge_{args.dataset}_a{args.alpha}_k{NC}_s{args.seed}.json"
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    json.dump({'dataset': args.dataset, 'alpha': args.alpha, 'K': NC, 'seed': args.seed,
               'n_clients': Kv, 'ensemble_params': ens_params,
               'single_backbone_params': one_params, 'results': rows},
              open(out, 'w'), indent=2)
    print(f"  saved {out}")


if __name__ == '__main__':
    main()
