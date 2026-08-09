"""
R1-D4 / R3-W7 -- quantitative multi-round vs one-shot comparison.

Implements a multi-round fixed-ETF federated method (the essence of FedETF/FedGELA:
a frozen simplex-ETF classifier + FedAvg over the client backbones each round) and
evaluates the single global backbone at rounds {1,3,5,10} by nearest-ETF prediction.
This is the apples-to-apples multi-round ETF/NC baseline the reviewers ask for: under
the one-shot constraint (round 1) it has had no aggregation, and we show how many
communication rounds it needs to approach FedSRA's one-shot accuracy.

Inference/aggregation of FedSRA (RGA over K one-shot backbones) is reported
separately; here we produce the multi-round curve.

Usage:
  python train_multiround_etf.py --dataset cifar100 --alpha 0.05 --K 5 \
      --rounds 10 --local_epochs 5 --eval_at 1,3,5,10
"""
import os, json, argparse, copy
import numpy as np, torch, torch.nn.functional as F
from rebuild8 import generate_etf, train_bb, device
from resnet18_filter_merge import ResNet18Backbone


def get_data(dataset, K, alpha, NL):
    if dataset == 'cifar10':
        from rebuild8 import prepare_data
        return prepare_data(K, alpha, NL)
    from rebuild8_cifar100 import prepare_data_cifar100
    return prepare_data_cifar100(K, alpha, NL)


def fedavg(states, weights):
    total = float(sum(weights))
    avg = copy.deepcopy(states[0])
    for key in avg:
        if torch.is_floating_point(avg[key]):
            avg[key] = sum(states[i][key] * (weights[i] / total) for i in range(len(states)))
        else:
            avg[key] = states[0][key]  # integer buffers (num_batches_tracked)
    return avg


@torch.no_grad()
def evaluate(state, tl, etf, FD):
    bb = ResNet18Backbone(FD); bb.load_state_dict(state); bb = bb.to(device).eval()
    ed = etf.to(device); correct = tot = 0
    for x, y in tl:
        h = F.normalize(bb(x.to(device)), dim=1)
        pred = (h @ ed.T).argmax(1).cpu()
        correct += (pred == y).sum().item(); tot += y.numel()
    bb.cpu()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return correct / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100'])
    ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--rounds', type=int, default=10)
    ap.add_argument('--local_epochs', type=int, default=5)
    ap.add_argument('--eval_at', default='1,3,5,10')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    NL = 10 if args.dataset == 'cifar10' else 100; FD = 256
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    etf = generate_etf(NL, FD)
    cal, ccl, tl, ccc = get_data(args.dataset, args.K, args.alpha, NL)
    eval_at = set(int(x) for x in args.eval_at.split(','))
    clients = [k for k in range(args.K) if k in cal]
    nk = {k: float(sum(ccc.get(k, {}).values())) for k in clients}

    global_state = {kk: v.detach().cpu() for kk, v in ResNet18Backbone(FD).state_dict().items()}
    curve = {}
    for r in range(1, args.rounds + 1):
        states, weights = [], []
        for k in clients:
            bb = ResNet18Backbone(FD); bb.load_state_dict(global_state)
            train_bb(bb, cal[k], list(ccc[k].keys()), etf,
                     epochs=args.local_epochs, loss_type='J', save_dir=None)
            states.append({kk: v.detach().cpu() for kk, v in bb.state_dict().items()})
            weights.append(nk[k]); bb.cpu()
        global_state = fedavg(states, weights)
        if r in eval_at:
            acc = evaluate(global_state, tl, etf, FD)
            curve[str(r)] = acc
            print(f"[{args.dataset} a{args.alpha} K{args.K}] round {r:2d}: acc={acc*100:6.2f}%", flush=True)

    out = args.out or f"results/multiround_{args.dataset}_a{args.alpha}_k{args.K}_s{args.seed}.json"
    os.makedirs('results', exist_ok=True)
    json.dump({'dataset': args.dataset, 'alpha': args.alpha, 'K': args.K, 'seed': args.seed,
               'rounds': args.rounds, 'local_epochs': args.local_epochs,
               'method': 'FedAvg over fixed-ETF backbones (FedETF-style multi-round)',
               'curve': curve}, open(out, 'w'), indent=2)
    print('saved', out)


if __name__ == '__main__':
    main()
