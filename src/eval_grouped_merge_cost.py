"""
A2 -- grouped filter-merge with the FULL serving-cost vector R2-D2 asks for:
accuracy, parameter count, merge/preprocessing time, peak GPU memory, and per-image
latency at batch B=1 (online) and B=256 (throughput). Sweeps merge threshold thr and
number of served models G on the SAME trained ETF backbones (inference only).

Serving cost is charged over ALL G merged models (every test sample is forwarded
through each of the G group-models, then RGA-aggregated), so latency is summed over
the G models. thr controls how aggressively similar filters are averaged: a LOW thr
merges more (compact model, maybe lower acc), a HIGH thr merges little (near-union,
wide model). This script measures whether a low thr genuinely compresses.

Output: results/groupmergecost_{dataset}_a{alpha}_k{K}_s{seed}.json
Usage:
  python eval_grouped_merge_cost.py --dataset cifar100 --NL 100 --alpha 0.05 --K 50 \
      --save_dir saved_models/cifar100_a0.05_k50_s42 --groups 1,5,10,25,50 \
      --thrs 0.5,0.7,0.85,0.95
"""
import os, argparse, json, time
import numpy as np, torch, torch.nn.functional as F
from rebuild8 import generate_etf, device
from resnet18_filter_merge import ResNet18Backbone, union_aggregate_resnet18
from eval_ablation_RIJ import load_backbones, get_test_loader_and_ccc

EPS = 1e-8


def count_params(m):
    return int(sum(p.numel() for p in m.parameters()))


def _forward(m, x):
    h = F.relu(m.bn1(m.conv1(x)))
    h = m.layer1(h); h = m.layer2(h); h = m.layer3(h); h = m.layer4(h)
    h = m.pool(h).flatten(1)
    return m.fc(h)


@torch.no_grad()
def raw_features(model, tl):
    model = model.to(device).eval()
    feats, labels = [], []
    for x, y in tl:
        feats.append(_forward(model, x.to(device)).cpu()); labels.append(y)
    model.cpu()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return torch.cat(feats, 0), torch.cat(labels).numpy()


@torch.no_grad()
def measure_latency(models, in_shape, B, reps=30, warmup=5):
    """Total per-image latency (ms) to forward ALL models on one batch of size B."""
    for m in models: m.to(device).eval()
    x = torch.randn(B, *in_shape, device=device)
    def one():
        for m in models: _forward(m, x)
    cuda = torch.cuda.is_available()
    for _ in range(warmup): one()
    if cuda: torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps): one()
    if cuda: torch.cuda.synchronize()
    dt = (time.time() - t0) / reps
    for m in models: m.cpu()
    if cuda: torch.cuda.empty_cache()
    return dt / B * 1000.0


@torch.no_grad()
def peak_mem_mb(models, in_shape, B=256):
    if not torch.cuda.is_available(): return None
    for m in models: m.to(device).eval()
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    x = torch.randn(B, *in_shape, device=device)
    for m in models: _forward(m, x)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e6
    for m in models: m.cpu()
    torch.cuda.empty_cache()
    return peak


def make_groups(valid, G, seed):
    order = np.random.RandomState(seed).permutation(valid)
    return [list(order[i::G]) for i in range(G)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--K', type=int, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100'])
    ap.add_argument('--NL', type=int, default=100)
    ap.add_argument('--FD', type=int, default=256)
    ap.add_argument('--groups', default='1,5,10,25,50')
    ap.add_argument('--thrs', default='0.5,0.7,0.85,0.95')
    ap.add_argument('--calib_n', type=int, default=1024)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    NC, NL, FD = args.K, args.NL, args.FD
    in_shape = (3, 32, 32)
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
        ct, et = torch.as_tensor(calib_idx), torch.as_tensor(eval_idx)
        agg = None; wsum = 0.0
        for f, w in model_feats_weights:
            mu = f[ct].mean(0, keepdim=True); sd = f[ct].std(0, keepdim=True)
            fz = torch.nan_to_num((f[et] - mu) / (sd + EPS), nan=0.0, posinf=0.0, neginf=0.0)
            agg = fz * w if agg is None else agg + fz * w; wsum += w
        z = F.normalize(agg / max(wsum, 1e-12), dim=1)
        return float(((z @ etf.T).argmax(1).numpy() == le).mean())

    rows = []
    labels = None; calib_idx = eval_idx = le = None
    for thr in thrs:
        for G in G_list:
            groups = make_groups(valid, G, args.seed)
            # build G merged models + time the merge (preprocessing cost)
            t_merge0 = time.time()
            models, tot_params = [], 0
            for grp in groups:
                m = bbs[grp[0]] if len(grp) == 1 else \
                    union_aggregate_resnet18([bbs[k] for k in grp], FD, thr, device)
                models.append(m); tot_params += count_params(m)
            merge_time = time.time() - t_merge0

            lat_b1 = measure_latency(models, in_shape, 1)
            lat_b256 = measure_latency(models, in_shape, 256)
            pmem = peak_mem_mb(models, in_shape, 256)

            feats_weights = []
            for grp, m in zip(groups, models):
                f, lab = raw_features(m, tl)
                if labels is None:
                    labels = lab; N = len(labels)
                    perm = np.random.RandomState(args.seed).permutation(N)
                    calib_idx, eval_idx = perm[:args.calib_n], perm[args.calib_n:]
                    le = labels[eval_idx]
                feats_weights.append((f, float(np.sqrt(sum(nk[k] for k in grp)))))
            acc = rga_acc(feats_weights, calib_idx, eval_idx, le)

            rows.append({'thr': thr, 'G': G, 'acc': acc, 'n_models': G,
                         'tot_params': tot_params, 'param_ratio': tot_params / ens_params,
                         'merge_time_s': merge_time, 'lat_b1_ms': lat_b1,
                         'lat_b256_ms': lat_b256, 'peak_mem_mb': pmem})
            print(f"  thr={thr:.2f} G={G:2d}: acc={acc*100:6.2f}% "
                  f"params={tot_params/1e6:7.1f}M lat(B1)={lat_b1:7.2f}ms "
                  f"lat(B256)={lat_b256:6.3f}ms mem={pmem:.0f}MB merge={merge_time:.1f}s")

    out = args.out or f"results/groupmergecost_{args.dataset}_a{args.alpha}_k{NC}_s{args.seed}.json"
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    json.dump({'dataset': args.dataset, 'alpha': args.alpha, 'K': NC, 'seed': args.seed,
               'n_clients': Kv, 'ensemble_params': ens_params,
               'single_backbone_params': one_params, 'in_shape': list(in_shape),
               'results': rows}, open(out, 'w'), indent=2)
    print(f"  saved {out}")


if __name__ == '__main__':
    main()
