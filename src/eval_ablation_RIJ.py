"""
eval_ablation_RIJ.py — evaluate R/I/J ablation cell.

For one (dataset, α, K) cell, loads saved_models for all available loss_types
(R, I, J) and produces the 4 ablation rows:
  row R       : R backbone aggregated via znorm+postL2+sqrt
  row I       : I backbone aggregated via znorm+postL2+sqrt
  row J       : J backbone aggregated via znorm+postL2+sqrt
  row J+Expert: J backbone aggregated + per-class experts fused (α_f=0.3)

All four rows use the same server-side aggregation algorithm; only the
backbone training loss and the optional expert path differ.

Output: results/ablation_RIJ_eval_{dataset}_a{α}_k{K}_s{seed}.json
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json, os, argparse
from resnet18_filter_merge import ResNet18Backbone
from rebuild8 import (
    generate_etf, ConditionalExpert, device,
    cross_client_per_client_logits, prepare_data,
)


def get_test_loader_and_ccc(name, NC, ALPHA, NL):
    if name == 'cifar10':
        _, _, tl, ccc = prepare_data(NC, ALPHA, NL)
    else:
        from rebuild8_cifar100 import prepare_data_cifar100
        _, _, tl, ccc = prepare_data_cifar100(NC, ALPHA, NL)
    return tl, ccc


def load_backbones(save_dir, NC, FD):
    bbs = []
    for k in range(NC):
        bp = os.path.join(save_dir, f"client_{k}", "backbone.pt")
        if not os.path.exists(bp):
            bbs.append(None); continue
        bb = ResNet18Backbone(FD)
        bb.load_state_dict(torch.load(bp, map_location='cpu', weights_only=True))
        bbs.append(bb)
    return bbs


def load_experts(save_dir, NC, NL, FD):
    client_exps = []
    for k in range(NC):
        cd = os.path.join(save_dir, f"client_{k}")
        exps = {}
        if os.path.exists(cd):
            for c in range(NL):
                ep = os.path.join(cd, f"expert_{c}.pt")
                if os.path.exists(ep):
                    exp = ConditionalExpert(FD, FD, 128, 32)
                    exp.load_state_dict(torch.load(ep, map_location='cpu',
                                                    weights_only=True))
                    exps[c] = exp
        client_exps.append(exps)
    return client_exps


def forward_features(bbs, client_exps, tl, etf, NC, FD, NL, N=10000,
                      use_experts=False):
    """Forward each client's backbone on test set; cache raw features.
    If use_experts, also compute per-client expert errors for all classes."""
    ed = etf.to(device)
    all_raw = []; all_labels = []
    errors = torch.full((NC, N, NL), float('inf')) if use_experts else None
    for k in range(NC):
        if bbs[k] is None: all_raw.append(None); continue
        bbs[k] = bbs[k].to(device); bbs[k].eval()
        if use_experts:
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
                if use_experts and client_exps[k]:
                    f_norm = F.normalize(feat, dim=1)
                    for c, exp in client_exps[k].items():
                        fr, _ = exp(f_norm, ed[c].unsqueeze(0).expand(bs, -1))
                        errors[k, offset:offset+bs, c] = ((f_norm - fr)**2).mean(1).cpu()
                if k == 0: all_labels.append(y)
                offset += bs
        all_raw.append(torch.cat(feats, 0))
        bbs[k] = bbs[k].cpu()
        if use_experts:
            for c in client_exps[k]:
                client_exps[k][c] = client_exps[k][c].cpu()
        torch.cuda.empty_cache()
    labels = torch.cat(all_labels).numpy()
    return all_raw, labels, errors


def znorm_sqrt_aggregate(all_raw, ccc, NC, N, FD):
    """znorm per-feature + sqrt(n_k) weighted sum + post-L2."""
    feat = torch.zeros(N, FD); w_sum = 0.0
    for k in range(NC):
        if all_raw[k] is None: continue
        f = all_raw[k]
        f_z = (f - f.mean(0, keepdim=True)) / (f.std(0, keepdim=True) + 1e-8)
        w = np.sqrt(sum(ccc.get(k, {}).values()))
        feat = feat + f_z * w; w_sum += w
    return F.normalize(feat / max(w_sum, 1e-12), dim=1)


def eval_row(name, save_dir, tl, ccc, etf, NC, FD, NL, N=10000,
             alpha_fusion=0.3, min_n=10, use_experts=False):
    """Returns (acc, n_clients_loaded) for one ablation row, or (None, 0) if
    saved_models not present."""
    if not os.path.isdir(save_dir):
        return None, 0
    bbs = load_backbones(save_dir, NC, FD)
    n_loaded = sum(1 for b in bbs if b is not None)
    if n_loaded == 0: return None, 0

    client_exps = load_experts(save_dir, NC, NL, FD) if use_experts else [{} for _ in range(NC)]
    all_raw, labels, errors = forward_features(
        bbs, client_exps, tl, etf, NC, FD, NL, N,
        use_experts=use_experts and any(client_exps))

    feat_agg = znorm_sqrt_aggregate(all_raw, ccc, NC, N, FD)
    union_logits = feat_agg @ etf.T
    preds = union_logits.argmax(1).numpy()
    acc_union = float((preds == labels).mean())

    if not use_experts or errors is None:
        return acc_union, n_loaded

    # Fused: relational logits + intrinsic logits via cross_client_per_client_logits
    sl, _ = union_logits.sort(dim=1, descending=True)
    sample_count = {}
    for k in range(NC):
        for c in client_exps[k]:
            sample_count[(k, c)] = ccc.get(k, {}).get(c, 0)
    data = {
        'errors': errors,
        'union_logits': union_logits,
        'union_preds': preds,
        'union_margin': (sl[:, 0] - sl[:, 1]).numpy(),
        'labels': labels,
        'sample_count': sample_count,
        'K': NC, 'N': N, 'nc': NL,
    }
    preds_fused = cross_client_per_client_logits(data, alpha=alpha_fusion, min_n=min_n)
    acc_fused = float((preds_fused == labels).mean())
    return acc_fused, n_loaded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--n_clients', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alpha_fusion', type=float, default=0.3,
                        help='Fixed fusion weight α_f for J+Expert row.')
    parser.add_argument('--min_n', type=int, default=10,
                        help='Per-client per-class min sample threshold for expert path.')
    args = parser.parse_args()

    ALPHA = args.alpha; NC = args.n_clients; SEED = args.seed
    NL = 10 if args.dataset == 'cifar10' else 100
    FD = 256
    torch.manual_seed(SEED); np.random.seed(SEED)

    print(f"\nEval RIJ: dataset={args.dataset}, α={ALPHA}, K={NC}, seed={SEED}")
    etf = generate_etf(NL, FD)
    tl, ccc = get_test_loader_and_ccc(args.dataset, NC, ALPHA, NL)
    N = 10000

    base = f"saved_models/ablation_{args.dataset}"
    rows = {}
    for tag in ['R', 'I', 'J']:
        sd = f"{base}/{tag}_a{ALPHA}_k{NC}_s{SEED}"
        acc, nload = eval_row(tag, sd, tl, ccc, etf, NC, FD, NL, N,
                              use_experts=False)
        rows[tag] = {'acc': acc, 'n_clients_loaded': nload}
        if acc is not None:
            print(f"  row {tag:<8}: {acc*100:6.2f}%  (loaded {nload}/{NC})")
        else:
            print(f"  row {tag:<8}: ─── (saved_models missing at {sd})")

    sd_J = f"{base}/J_a{ALPHA}_k{NC}_s{SEED}"
    acc_je, nload = eval_row('J+E', sd_J, tl, ccc, etf, NC, FD, NL, N,
                              alpha_fusion=args.alpha_fusion,
                              min_n=args.min_n, use_experts=True)
    rows['J+Expert'] = {'acc': acc_je, 'n_clients_loaded': nload,
                         'alpha_fusion': args.alpha_fusion, 'min_n': args.min_n}
    if acc_je is not None:
        print(f"  row J+Expert: {acc_je*100:6.2f}%  (α_f={args.alpha_fusion}, min_n={args.min_n})")
    else:
        print(f"  row J+Expert: ─── (J saved_models or experts missing)")

    sd_I = f"{base}/I_a{ALPHA}_k{NC}_s{SEED}"
    acc_ie, nload = eval_row('I+E', sd_I, tl, ccc, etf, NC, FD, NL, N,
                              alpha_fusion=args.alpha_fusion,
                              min_n=args.min_n, use_experts=True)
    rows['I+Expert'] = {'acc': acc_ie, 'n_clients_loaded': nload,
                         'alpha_fusion': args.alpha_fusion, 'min_n': args.min_n}
    if acc_ie is not None:
        print(f"  row I+Expert: {acc_ie*100:6.2f}%  (α_f={args.alpha_fusion}, min_n={args.min_n})")
    else:
        print(f"  row I+Expert: ─── (I saved_models or I-experts missing)")

    out = {
        'cell': {'dataset': args.dataset, 'alpha': ALPHA, 'K': NC, 'seed': SEED},
        'aggregation': 'znorm+postL2+sqrt',
        'rows': rows,
    }
    os.makedirs('results', exist_ok=True)
    path = f"results/ablation_RIJ_eval_{args.dataset}_a{ALPHA}_k{NC}_s{SEED}.json"
    json.dump(out, open(path, 'w'), indent=2)
    print(f"\n  Saved: {path}")


if __name__ == '__main__':
    main()
