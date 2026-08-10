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
import os, json, argparse, copy, glob
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


def atomic_torch_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json_dump(obj, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


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
    ap.add_argument('--checkpoint_dir', default=None)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--save_clients', action='store_true',
                    help='retain every round/client state in addition to global states')
    args = ap.parse_args()

    NL = 10 if args.dataset == 'cifar10' else 100; FD = 256
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    etf = generate_etf(NL, FD)
    cal, ccl, tl, ccc = get_data(args.dataset, args.K, args.alpha, NL)
    eval_at = set(int(x) for x in args.eval_at.split(','))
    clients = [k for k in range(args.K) if k in cal]
    nk = {k: float(sum(ccc.get(k, {}).values())) for k in clients}

    out = args.out or f"results/multiround_{args.dataset}_a{args.alpha}_k{args.K}_s{args.seed}.json"
    ckpt_dir = args.checkpoint_dir or os.path.join(
        'saved_models', f"multiround_{args.dataset}_a{args.alpha}_k{args.K}_s{args.seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    config = {'dataset': args.dataset, 'alpha': args.alpha, 'K': args.K,
              'seed': args.seed, 'rounds': args.rounds,
              'local_epochs': args.local_epochs, 'eval_at': sorted(eval_at),
              'save_clients': args.save_clients}
    config_path = os.path.join(ckpt_dir, 'config.json')
    if os.path.exists(config_path):
        old = json.load(open(config_path))
        if old != config:
            raise RuntimeError(f"checkpoint config mismatch: {old} != {config}")
    else:
        atomic_json_dump(config, config_path)

    global_state = {kk: v.detach().cpu() for kk, v in ResNet18Backbone(FD).state_dict().items()}
    curve = {}
    start_round = 1
    if args.resume:
        completed = []
        for marker in glob.glob(os.path.join(ckpt_dir, 'round_*', 'COMPLETE.json')):
            try:
                completed.append(int(os.path.basename(os.path.dirname(marker)).split('_')[1]))
            except (ValueError, IndexError):
                pass
        if completed:
            last = max(completed)
            global_state = torch.load(os.path.join(ckpt_dir, f'round_{last:02d}', 'global.pt'),
                                      map_location='cpu', weights_only=True)
            progress = os.path.join(ckpt_dir, 'progress.json')
            if os.path.exists(progress):
                curve = json.load(open(progress)).get('curve', {})
            start_round = last + 1
            print(f"[resume] completed through round {last}", flush=True)

    def write_result():
        payload = {'dataset': args.dataset, 'alpha': args.alpha, 'K': args.K,
                   'seed': args.seed, 'rounds': args.rounds,
                   'local_epochs': args.local_epochs,
                   'method': 'FedAvg over fixed-ETF backbones (FedETF-style multi-round)',
                   'checkpoint_dir': os.path.abspath(ckpt_dir), 'curve': curve}
        atomic_json_dump(payload, out)
        atomic_json_dump({'last_round': max([0] + [int(x) for x in curve.keys()]),
                          'curve': curve}, os.path.join(ckpt_dir, 'progress.json'))

    for r in range(start_round, args.rounds + 1):
        round_dir = os.path.join(ckpt_dir, f'round_{r:02d}')
        os.makedirs(round_dir, exist_ok=True)
        states, weights = [], []
        for k in clients:
            client_path = os.path.join(round_dir, f'client_{k}.pt')
            if args.resume and args.save_clients and os.path.exists(client_path):
                state = torch.load(client_path, map_location='cpu', weights_only=True)
                print(f"[resume] round {r} client {k}", flush=True)
            else:
                bb = ResNet18Backbone(FD); bb.load_state_dict(global_state)
                train_bb(bb, cal[k], list(ccc[k].keys()), etf,
                         epochs=args.local_epochs, loss_type='J', save_dir=None)
                state = {kk: v.detach().cpu() for kk, v in bb.state_dict().items()}
                if args.save_clients:
                    atomic_torch_save(state, client_path)
                bb.cpu()
            states.append(state)
            weights.append(nk[k])
        global_state = fedavg(states, weights)
        atomic_torch_save(global_state, os.path.join(round_dir, 'global.pt'))
        if r in eval_at:
            acc = evaluate(global_state, tl, etf, FD)
            curve[str(r)] = acc
            print(f"[{args.dataset} a{args.alpha} K{args.K}] round {r:2d}: acc={acc*100:6.2f}%", flush=True)
        atomic_json_dump({'round': r, 'evaluated': r in eval_at},
                         os.path.join(round_dir, 'COMPLETE.json'))
        write_result()

    write_result()
    print('saved', out)


if __name__ == '__main__':
    main()
