"""
train_bb_woetf.py — train backbone with LEARNABLE per-client anchors W_k
(instead of the fixed ETF matrix). Used for the "w/o ETF" ablation row.

This isolates the contribution of ETF's fixed equiangular structure from the
rest of the loss machinery: same etf_cl + 0.5*etf_al losses, same GPA-style
aggregation pipeline, but W_k is learnable per-client (initialized random,
re-normalized to unit norm at each forward pass for fair comparison with ETF
which is unit-norm by construction).

Per-client saved files:
  saved_models/ablation_woetf_{dataset}/a{α}_k{K}_s{seed}/client_{k}/backbone.pt
  saved_models/ablation_woetf_{dataset}/a{α}_k{K}_s{seed}/client_{k}/W.pt
  saved_models/ablation_woetf_{dataset}/a{α}_k{K}_s{seed}/ccc.pt

Usage:
  python train_bb_woetf.py --dataset cifar100 --alpha 0.05 --n_clients 5 --seed 42
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json, time, os, argparse, sys
from tqdm import tqdm
from resnet18_filter_merge import ResNet18Backbone
from rebuild8 import device, USE_BF16, prepare_data, etf_cl, etf_al


def train_bb_with_W(bb, W_k, loader, classes, epochs=300, lr=1e-3, lambda_al=0.5,
                    client_id=None):
    """Train backbone + learnable W_k jointly with etf_cl + lambda_al * etf_al."""
    bb = bb.to(device); bb.train()
    # Re-wrap W_k as a leaf Parameter on the target device (avoids non-leaf error)
    W_k = nn.Parameter(W_k.detach().to(device))
    opt = torch.optim.Adam(list(bb.parameters()) + [W_k], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    amp = (torch.amp.autocast('cuda', dtype=torch.bfloat16) if USE_BF16
           else torch.amp.autocast('cuda', enabled=False))
    ncl = len(classes)
    pbar = tqdm(range(epochs), desc=f"  BB c{client_id}[wo-ETF]",
                ncols=100, mininterval=10.0, file=sys.stdout)
    for ep in pbar:
        el = 0; nb = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with amp:
                f = bb(x)
                W_norm = F.normalize(W_k, dim=1)  # ensure unit-norm at use time
                if ncl >= 2:
                    loss = etf_cl(f, y, W_norm) + lambda_al * etf_al(f, y, W_norm)
                else:
                    loss = etf_al(f, y, W_norm)
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step()
            el += loss.item(); nb += 1
        sch.step()
        pbar.set_postfix(loss=f"{el/max(nb,1):.3f}")
    pbar.close()
    bb.eval()
    return bb, W_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--n_clients', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lambda_al', type=float, default=0.5)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    NL = 10 if args.dataset == 'cifar10' else 100
    FD = 256
    EPB = args.epochs or (600 if args.dataset == 'cifar10' else 300)
    NC = args.n_clients; SEED = args.seed
    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    save_dir = (f"saved_models/ablation_woetf_{args.dataset}/"
                f"a{args.alpha}_k{NC}_s{SEED}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  w/o ETF training (learnable W_k)")
    print(f"  dataset={args.dataset}, α={args.alpha}, K={NC}, seed={SEED}")
    print(f"  epochs={EPB}, lambda_al={args.lambda_al}")
    print(f"  save_dir: {save_dir}")
    print(f"{'='*64}")

    if args.dataset == 'cifar10':
        cal, ccl, tl, ccc = prepare_data(NC, args.alpha, NL)
    else:
        from rebuild8_cifar100 import prepare_data_cifar100
        cal, ccl, tl, ccc = prepare_data_cifar100(NC, args.alpha, NL)

    t0 = time.time()
    for k in range(NC):
        cls = sorted(ccc.get(k, {}).keys())
        if not cls:
            print(f"  Client {k}: EMPTY (skip)"); continue
        cd = os.path.join(save_dir, f"client_{k}")
        bb_path = os.path.join(cd, "backbone.pt")
        W_path = os.path.join(cd, "W.pt")

        if args.resume and os.path.exists(bb_path) and os.path.exists(W_path):
            print(f"  Client {k}: resume (skip)")
            continue

        os.makedirs(cd, exist_ok=True)
        print(f"  Client {k}: {len(cls)} cls, {sum(ccc[k].values())} samp")

        bb = ResNet18Backbone(FD)
        # Init W_k as random unit vectors
        W_init = torch.randn(NL, FD) * 0.1
        W_init = F.normalize(W_init, dim=1)
        W_k = nn.Parameter(W_init)

        bb, W_k = train_bb_with_W(bb, W_k, cal[k], cls, EPB,
                                  lambda_al=args.lambda_al, client_id=k)

        torch.save(bb.cpu().state_dict(), bb_path)
        torch.save(W_k.detach().cpu(), W_path)
        torch.cuda.empty_cache()

    torch.save(dict(ccc), os.path.join(save_dir, "ccc.pt"))
    train_time = time.time() - t0
    print(f"\n  All clients done. Total: {train_time:.0f}s")

    meta = {
        'method': 'w/o ETF (learnable W)',
        'dataset': args.dataset, 'alpha': args.alpha, 'n_clients': NC, 'seed': SEED,
        'epochs': EPB, 'lambda_al': args.lambda_al,
        'train_time_sec': round(train_time, 1),
        'save_dir': save_dir,
    }
    meta_path = f"results/woetf_meta_{args.dataset}_a{args.alpha}_k{NC}_s{SEED}.json"
    os.makedirs('results', exist_ok=True)
    json.dump(meta, open(meta_path, 'w'), indent=2)
    print(f"  Meta -> {meta_path}")


if __name__ == '__main__':
    main()
