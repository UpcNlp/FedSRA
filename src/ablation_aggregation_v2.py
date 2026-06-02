"""
ablation_aggregation.py — 2D ablation on the RELATIONAL aggregation path.

  Weighting (5):   uniform, log(n), sqrt(n), linear(n), square(n²)
  Normalization (4):
    N1 raw_postL2     — no per-client norm; final L2 only (ETF compat)
    N2 preL2_noPost   — per-client L2; weighted sum; no post-norm
    N3 preL2_postL2   — per-client L2; weighted sum; post-L2 (proto-style)
    N4 znorm_postL2   — per-feature znorm; weighted sum; post-L2 (PROPOSED)

For each cell (α, K) with saved_models, forward each client's backbone once,
then sweep all 5×4=20 (weight, norm) combinations on the cached features.

Usage:
  python ablation_aggregation.py --alpha 0.05 --n_clients 10 --seed 42
  python ablation_aggregation.py --alpha 0.05 --n_clients 10 --seed 42 --use_experts  # also fuse intrinsic

Output:
  results/ablation_agg_a{α}_k{K}_s{seed}.json
  Schema: { 'cell': {α, K, seed},
            'union_only_per_combo': {w:{n:acc}},      # 20 numbers (alpha_f=0 case)
            'fused_per_combo':      {w:{n:acc}},      # 20 numbers (with experts fused, if --use_experts)
            'alpha_f': 0.3 }                          # fixed fusion weight used
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json, os, argparse
from rebuild8 import (prepare_data, generate_etf, ConditionalExpert, device,
                       cross_client_per_client_logits, expert_original)
from resnet18_filter_merge import ResNet18Backbone


WEIGHTS = ['uniform', 'log', 'sqrt', 'linear', 'square']
NORMS = ['raw_postL2', 'preL2_noPost', 'preL2_postL2', 'znorm_postL2', 'znorm_noPost', 'raw_noPost']


def load_models(save_dir, NC, NL, FD=256):
    bbs = []; client_exps = []
    for k in range(NC):
        cd = os.path.join(save_dir, f"client_{k}")
        if not os.path.exists(cd):
            bbs.append(None); client_exps.append({}); continue
        bb = ResNet18Backbone(FD)
        bb.load_state_dict(torch.load(f"{cd}/backbone.pt", map_location='cpu', weights_only=True))
        bbs.append(bb)
        exps = {}
        for c in range(NL):
            ep = f"{cd}/expert_{c}.pt"
            if os.path.exists(ep):
                exp = ConditionalExpert(FD, FD, 128, 32)
                exp.load_state_dict(torch.load(ep, map_location='cpu', weights_only=True))
                exps[c] = exp
        client_exps.append(exps)
    ccc = torch.load(os.path.join(save_dir, "ccc.pt"), weights_only=False)
    return bbs, client_exps, ccc


def compute_weight(n_k, scheme):
    if scheme == 'uniform': return 1.0
    if scheme == 'log':     return float(np.log(n_k + 1))
    if scheme == 'sqrt':    return float(np.sqrt(n_k))
    if scheme == 'linear':  return float(n_k) / 100.0    # divide for numerical stability
    if scheme == 'square':  return (float(n_k) / 100.0) ** 2
    raise ValueError(scheme)


def aggregate(all_raw, n_per_client, weight_scheme, norm_scheme):
    """Returns (N, FD) aggregated feature for argmax via ETF.
    all_raw: list of K tensors (N, FD) or None
    n_per_client: list of K ints (total samples)
    """
    K = len(all_raw)
    # Build per-client transformed features f_k (N, FD)
    transformed = []
    for k in range(K):
        if all_raw[k] is None:
            transformed.append(None); continue
        f = all_raw[k]
        if norm_scheme == 'raw_postL2':
            f_t = f                                                  # no per-client transform
        elif norm_scheme in ('preL2_noPost', 'preL2_postL2'):
            f_t = F.normalize(f, dim=1)                              # per-sample L2 per client
        elif norm_scheme == 'znorm_postL2':
            f_t = (f - f.mean(0, keepdim=True)) / (f.std(0, keepdim=True) + 1e-8)
        elif norm_scheme == 'znorm_noPost':
            f_t = (f - f.mean(0, keepdim=True)) / (f.std(0, keepdim=True) + 1e-8)
        elif norm_scheme == 'raw_noPost':
            f_t = f
        else:
            raise ValueError(norm_scheme)
        transformed.append(f_t)

    # Weighted sum
    feat = torch.zeros_like([t for t in transformed if t is not None][0])
    w_sum = 0.0
    for k in range(K):
        if transformed[k] is None: continue
        w = compute_weight(n_per_client[k], weight_scheme)
        feat = feat + transformed[k] * w
        w_sum += w
    feat = feat / max(w_sum, 1e-12)

    # Post-normalization
    if norm_scheme in ('raw_postL2', 'preL2_postL2', 'znorm_postL2'):
        feat = F.normalize(feat, dim=1)
    # else preL2_noPost: leave as-is (sum-of-unit-vectors / w_sum)

    return feat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--n_clients', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--use_experts', action='store_true',
                        help='Also compute fused (relational+intrinsic) acc with fixed alpha_f.')
    parser.add_argument('--alpha_f', type=float, default=0.3,
                        help='Fixed fusion weight used in --use_experts mode.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Explicit backbone dir override; bypasses path auto-detection.')
    args = parser.parse_args()

    ALPHA = args.alpha; NC = args.n_clients; SEED = args.seed
    NL = 10 if args.dataset == 'cifar10' else 100; FD = 256

    # saved_models layout differs by dataset AND by K (K=5 ablation dirs use a
    # different prefix than the K=10/20/50 main-table dirs). Try candidates in order.
    if args.save_dir:
        candidates = [args.save_dir]
    elif args.dataset == 'cifar10':
        candidates = [f"saved_models/a{ALPHA}_k{NC}_s{SEED}",                 # K=10/20 legacy
                      f"saved_models/ablation_cifar10_a{ALPHA}_k{NC}_s{SEED}",  # K=5 ablation
                      f"saved_models/ablation_cifar10/J_a{ALPHA}_k{NC}_s{SEED}"]  # canonical symlink
    else:
        candidates = [f"saved_models/cifar100_a{ALPHA}_k{NC}_s{SEED}",
                      f"saved_models/ablation_cifar100_a{ALPHA}_k{NC}_s{SEED}",
                      f"saved_models/ablation_cifar100/J_a{ALPHA}_k{NC}_s{SEED}"]
    save_dir = next((d for d in candidates if os.path.exists(d)), None)
    if save_dir is None:
        print(f"saved_models 不存在,尝试过: {candidates}")
        return

    print(f"\nAblation aggregation: dataset={args.dataset}, α={ALPHA}, K={NC}, seed={SEED}")
    print(f"  Loading from {save_dir}...")
    torch.manual_seed(SEED); np.random.seed(SEED)
    etf = generate_etf(NL, FD)
    bbs, client_exps, ccc = load_models(save_dir, NC, NL, FD)
    n_loaded = sum(1 for b in bbs if b is not None)
    print(f"  Loaded {n_loaded}/{NC} clients")

    if args.dataset == 'cifar10':
        _, _, tl, _ = prepare_data(NC, ALPHA, NL)
    else:
        from rebuild8_cifar100 import prepare_data_cifar100
        _, _, tl, _ = prepare_data_cifar100(NC, ALPHA, NL)
    ed = etf.to(device)
    N = 10000

    # ── Forward once to cache raw features (and errors if experts used)
    all_raw = []; all_labels = []
    errors = torch.full((NC, N, NL), float('inf')) if args.use_experts else None

    for k in range(NC):
        if bbs[k] is None: all_raw.append(None); continue
        bbs[k] = bbs[k].to(device); bbs[k].eval()
        if args.use_experts:
            for c in client_exps[k]:
                client_exps[k][c] = client_exps[k][c].to(device)
        feats = []; offset = 0
        with torch.no_grad():
            for x, y in tl:
                x = x.to(device); bs = x.size(0)
                xx = F.relu(bbs[k].bn1(bbs[k].conv1(x)))
                xx = bbs[k].layer1(xx); xx = bbs[k].layer2(xx)
                xx = bbs[k].layer3(xx); xx = bbs[k].layer4(xx)
                xx = bbs[k].pool(xx).flatten(1)
                feat = bbs[k].fc(xx)
                feats.append(feat.cpu())
                if args.use_experts:
                    f_norm = F.normalize(feat, dim=1)
                    for c, exp in client_exps[k].items():
                        fr, _ = exp(f_norm, ed[c].unsqueeze(0).expand(bs, -1))
                        errors[k, offset:offset+bs, c] = ((f_norm - fr)**2).mean(1).cpu()
                if k == 0: all_labels.append(y)
                offset += bs
        all_raw.append(torch.cat(feats, 0))
        bbs[k] = bbs[k].cpu()
        if args.use_experts:
            for c in client_exps[k]: client_exps[k][c] = client_exps[k][c].cpu()
        torch.cuda.empty_cache()
    labels = torch.cat(all_labels).numpy()

    n_per_client = [sum(ccc.get(k, {}).values()) for k in range(NC)]
    sample_count = {}
    for k in range(NC):
        for c in client_exps[k]:
            sample_count[(k, c)] = ccc.get(k, {}).get(c, 0)

    # ── Sweep 5×4 = 20 combos
    print(f"\n  {'norm':>14} | " + " ".join([f"{w:>8}" for w in WEIGHTS]))
    print(f"  {'-'*14}-+-" + "-".join(["-"*8 for _ in WEIGHTS]))

    union_table = {}  # norm -> weight -> acc
    fused_table = {} if args.use_experts else None

    for nm in NORMS:
        union_row = {}; fused_row = {}
        line = f"  {nm:>14} | "
        for wt in WEIGHTS:
            feat_agg = aggregate(all_raw, n_per_client, wt, nm)
            union_logits = feat_agg @ etf.T
            preds = union_logits.argmax(1).numpy()
            acc_u = float((preds == labels).mean())
            union_row[wt] = acc_u

            if args.use_experts:
                sl, _ = union_logits.sort(dim=1, descending=True)
                data = {
                    'errors': errors, 'union_logits': union_logits,
                    'union_preds': preds, 'union_margin': (sl[:,0]-sl[:,1]).numpy(),
                    'labels': labels, 'sample_count': sample_count,
                    'K': NC, 'N': N, 'nc': NL,
                }
                preds_f = cross_client_per_client_logits(data, alpha=args.alpha_f, min_n=10)
                acc_f = float((preds_f == labels).mean())
                fused_row[wt] = acc_f
                line += f"{acc_u*100:5.2f}/{acc_f*100:5.2f} "
            else:
                line += f"  {acc_u*100:6.2f} "
        union_table[nm] = union_row
        if args.use_experts: fused_table[nm] = fused_row
        print(line)

    out = {
        'cell': {'dataset': args.dataset, 'alpha': ALPHA, 'K': NC, 'seed': SEED},
        'union_only_per_combo': union_table,
        'alpha_f': args.alpha_f,
    }
    if args.use_experts:
        out['fused_per_combo'] = fused_table

    os.makedirs('results', exist_ok=True)
    tag = '' if args.dataset == 'cifar10' else f'{args.dataset}_'
    path = f"results/ablation_agg_{tag}a{ALPHA}_k{NC}_s{SEED}.json"
    json.dump(out, open(path, 'w'), indent=2)
    print(f"\n  Saved: {path}")


if __name__ == '__main__':
    main()
