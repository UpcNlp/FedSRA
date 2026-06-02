"""
train_centralized_cifar10.py — train ONE ETF-anchored ResNet18 on the full
CIFAR-10 train set (no federation). This is the "centralized reference" for
the fednc_overview figure: shows what NC structure looks like when a single
model sees the entire dataset with the same ETF anchor used in FedDSI.

Uses the same loss as the J variant (etf_cl + 0.5 * etf_al) and the same
ResNet18 backbone, so the comparison to per-client J backbones is controlled
to data partitioning + aggregation alone.

Saves to: saved_models/centralized_cifar10_s42/client_0/backbone.pt
(client_0 naming chosen so the existing load_backbones API can read it.)

Usage on cluster:
  source /opt/dtk/env.sh
  HIP_VISIBLE_DEVICES=1 python -u train_centralized_cifar10.py --seed 42
"""
import argparse, os, time
import torch, torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from rebuild8 import etf_cl, etf_al, generate_etf, device, DL_KWARGS
from resnet18_filter_merge import ResNet18Backbone


def make_full_loader(root='./data', bs=128):
    tt = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])
    train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=tt)
    return DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True,
                      **DL_KWARGS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=600)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--save_dir',
                    default='saved_models/centralized_cifar10_s42/client_0')
    args = ap.parse_args()

    NL = 10
    FD = 256

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    etf = generate_etf(NL, FD).to(device)
    loader = make_full_loader()
    print(f"[data] {len(loader.dataset)} samples (full CIFAR-10 train), "
          f"{len(loader)} batches/epoch", flush=True)

    bb = ResNet18Backbone(FD).to(device)
    opt = torch.optim.AdamW(bb.parameters(), lr=args.lr, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    classes = list(range(NL))  # all classes present
    bb.train()
    t0 = time.time()
    for ep in range(args.epochs):
        total = 0.0; n = 0
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            f = bb(x)
            loss = etf_cl(f, y, etf) + 0.5 * etf_al(f, y, etf)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * y.size(0); n += y.size(0)
        sch.step()
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"[ep {ep+1:3d}/{args.epochs}] loss={total/n:.4f} "
                  f"lr={sch.get_last_lr()[0]:.2e} elapsed={time.time()-t0:.0f}s",
                  flush=True)

    os.makedirs(args.save_dir, exist_ok=True)
    out = os.path.join(args.save_dir, 'backbone.pt')
    bb.eval().cpu()
    torch.save(bb.state_dict(), out)
    print(f"[save] {out}", flush=True)
    print(f"[done] total time {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
