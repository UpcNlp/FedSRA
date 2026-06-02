"""
train_federated_ce_cifar10.py — train ONE federated CE client backbone
(no ETF anchor). Used to build the "no ETF" column of the fednc_overview
2x4 figure.

We re-use rebuild8.prepare_data(K=10, alpha=0.05) to get the SAME Dirichlet
partition that the J-backbone fed runs used; this makes the comparison
controlled to the loss alone (CE vs ETF-aligned).

Each client trains ResNet18Backbone(fd=256) + Linear(256, 10) classification
head with cross-entropy. After training we drop the head and save only the
backbone state_dict, matching the layout the J backbones use:

    saved_models/ce_cifar10_a{α}_k{K}_s{seed}/client_{i}/backbone.pt

So that the existing load_backbones() reads it without modification.

Usage on cluster:
  source /opt/dtk/env.sh
  HIP_VISIBLE_DEVICES=$GPU python -u train_federated_ce_cifar10.py \
        --client_id $i --alpha 0.05 --K 10 --seed 42
"""
import argparse, os, time, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from rebuild8 import prepare_data, device, USE_BF16
from resnet18_filter_merge import ResNet18Backbone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--client_id', type=int, required=True,
                    help='which client (0..K-1) to train')
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=600)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--save_root',
                    default='saved_models/ce_cifar10_a{alpha}_k{K}_s{seed}')
    args = ap.parse_args()

    NL = 10
    FD = 256

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # SAME Dirichlet split as the J-backbone runs (rebuild8.prepare_data uses
    # the same seed-dependent split when called with the same args).
    cal, ccl, tl, ccc = prepare_data(args.K, args.alpha, NL)
    if args.client_id not in cal:
        print(f"[fatal] client {args.client_id} has no data in this split")
        return
    loader = cal[args.client_id]
    classes_here = sorted(ccc.get(args.client_id, {}).keys())
    print(f"[client {args.client_id}] {len(classes_here)} classes, "
          f"{sum(ccc[args.client_id].values())} samples", flush=True)

    bb = ResNet18Backbone(FD).to(device)
    head = nn.Linear(FD, NL).to(device)
    opt = torch.optim.AdamW(list(bb.parameters()) + list(head.parameters()),
                            lr=args.lr, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss()

    bb.train(); head.train()
    t0 = time.time()
    for ep in range(args.epochs):
        total, n = 0.0, 0
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            f = bb(x)
            logits = head(f)
            loss = ce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss) * y.size(0); n += y.size(0)
        sch.step()
        if (ep + 1) % 50 == 0 or ep == 0 or ep == args.epochs - 1:
            print(f"[ep {ep+1:3d}/{args.epochs}] loss={total/max(n,1):.4f} "
                  f"lr={sch.get_last_lr()[0]:.2e} elapsed={time.time()-t0:.0f}s",
                  flush=True)

    save_root = args.save_root.format(alpha=args.alpha, K=args.K, seed=args.seed)
    save_dir = os.path.join(save_root, f'client_{args.client_id}')
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, 'backbone.pt')
    bb.eval().cpu()
    torch.save(bb.state_dict(), out)
    print(f"[save] {out}", flush=True)
    print(f"[done] total time {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
