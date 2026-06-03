"""
eval_bb_woetf.py — evaluate the "w/o ETF" ablation.

Loads per-client backbones and learnable anchor matrices W_k trained by
train_bb_woetf.py. Aggregates features via GPA's znorm + sqrt(n) + post-L2
recipe, classifies by projecting onto W_avg = mean(W_k) (unit-normalized).

Output: results/woetf_eval_{dataset}_a{α}_k{K}_s{seed}.json
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json, os, argparse
from resnet18_filter_merge import ResNet18Backbone
from rebuild8 import device, prepare_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--n_clients', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    NL = 10 if args.dataset == 'cifar10' else 100
    FD = 256
    NC = args.n_clients; SEED = args.seed
    torch.manual_seed(SEED); np.random.seed(SEED)

    save_dir = (f"saved_models/ablation_woetf_{args.dataset}/"
                f"a{args.alpha}_k{NC}_s{SEED}")
    print(f"\nEval w/o ETF: dataset={args.dataset}, α={args.alpha}, K={NC}, seed={SEED}")
    print(f"  save_dir: {save_dir}")

    if not os.path.isdir(save_dir):
        print(f"  ERROR: save_dir not found")
        return

    # Use SAME data prep call order as training so partition matches
    if args.dataset == 'cifar10':
        _, _, tl, ccc = prepare_data(NC, args.alpha, NL)
    else:
        from rebuild8_cifar100 import prepare_data_cifar100
        _, _, tl, ccc = prepare_data_cifar100(NC, args.alpha, NL)

    # Load all clients' backbones and W
    bbs = []; Ws = []
    for k in range(NC):
        cd = os.path.join(save_dir, f"client_{k}")
        bb_path = os.path.join(cd, "backbone.pt")
        W_path = os.path.join(cd, "W.pt")
        if not (os.path.exists(bb_path) and os.path.exists(W_path)):
            bbs.append(None); Ws.append(None); continue
        bb = ResNet18Backbone(FD)
        bb.load_state_dict(torch.load(bb_path, map_location='cpu', weights_only=True))
        W = torch.load(W_path, map_location='cpu', weights_only=True)
        bbs.append(bb); Ws.append(W)

    n_loaded = sum(1 for b in bbs if b is not None)
    print(f"  Loaded {n_loaded}/{NC} clients")
    if n_loaded == 0:
        print(f"  ERROR: no client backbones loaded")
        return

    # Forward each client on test set; collect raw features
    all_feats = []; all_labels = None
    for k in range(NC):
        if bbs[k] is None: all_feats.append(None); continue
        bbs[k] = bbs[k].to(device); bbs[k].eval()
        feats = []; labels = []
        with torch.no_grad():
            for x, y in tl:
                f = bbs[k](x.to(device))
                feats.append(f.cpu())
                if all_labels is None: labels.append(y)
        all_feats.append(torch.cat(feats, 0))
        if all_labels is None: all_labels = torch.cat(labels).numpy()
        bbs[k] = bbs[k].cpu()
        torch.cuda.empty_cache()
    N = len(all_labels)

    # GPA aggregation: znorm per client + sqrt(n_k) weighted sum + post-L2
    feat_agg = torch.zeros(N, FD)
    w_sum = 0.0
    for k in range(NC):
        if all_feats[k] is None: continue
        f = all_feats[k]
        f_z = (f - f.mean(0, keepdim=True)) / (f.std(0, keepdim=True) + 1e-8)
        w = float(np.sqrt(sum(ccc.get(k, {}).values())))
        feat_agg = feat_agg + f_z * w
        w_sum += w
    feat_agg = F.normalize(feat_agg / max(w_sum, 1e-12), dim=1)

    # W_avg = mean of learnable anchor matrices across clients (unit-normalized rows)
    W_avg = torch.zeros(NL, FD)
    nW = 0
    for k in range(NC):
        if Ws[k] is None: continue
        W_avg = W_avg + F.normalize(Ws[k], dim=1)  # normalize each client's W rows first
        nW += 1
    W_avg = W_avg / max(nW, 1)
    W_avg = F.normalize(W_avg, dim=1)  # post-average normalization

    # Classify
    preds = (feat_agg @ W_avg.T).argmax(1).numpy()
    acc = float((preds == all_labels).mean())
    print(f"  w/o ETF acc: {acc*100:.2f}%")

    os.makedirs('results', exist_ok=True)
    out = {
        'method': 'w/o ETF (learnable W, W_avg eval)',
        'dataset': args.dataset, 'alpha': args.alpha, 'n_clients': NC, 'seed': SEED,
        'acc': acc, 'n_clients_loaded': n_loaded,
        'aggregation': 'znorm+postL2+sqrt; W_avg = mean(L2-normalized W_k)',
    }
    path = f"results/woetf_eval_{args.dataset}_a{args.alpha}_k{NC}_s{SEED}.json"
    json.dump(out, open(path, 'w'), indent=2)
    print(f"  Saved: {path}")


if __name__ == '__main__':
    main()
