"""
Train CE (cross-entropy, NO ETF) ResNet-18 backbones -- the "without ETF"
contrast for the cross-client alignment figures. Mirrors variant H of
ablation_components_v2.py, but PERSISTS each client's backbone.

Same call order as rebuild8_resnet18.main (seed -> generate_etf -> prepare_data)
so the client partition matches the ERL backbones exactly.

Per-client resume: skips client_k if backbone.pt already exists.
"""
import os, time, argparse
import numpy as np
import torch
import torch.nn as nn

from rebuild8 import device, prepare_data, generate_etf
from resnet18_filter_merge import ResNet18Backbone

NL, FD = 10, 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--K', type=int, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    _ = generate_etf(NL, FD)  # advance RNG identically to training run
    cal, ccl, tl, ccc = prepare_data(args.K, args.alpha, NL)
    os.makedirs(args.out, exist_ok=True)

    for k in range(args.K):
        cd = os.path.join(args.out, f"client_{k}")
        ckpt = os.path.join(cd, "backbone.pt")
        if os.path.exists(ckpt):
            print(f"client {k}: skip (resume)", flush=True)
            continue
        os.makedirs(cd, exist_ok=True)
        bb = ResNet18Backbone(FD).to(device); bb.train()
        head = nn.Linear(FD, NL).to(device)
        opt = torch.optim.Adam(list(bb.parameters()) + list(head.parameters()), lr=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        lossfn = nn.CrossEntropyLoss()
        t0 = time.time()
        for ep in range(args.epochs):
            el = nb = 0
            for x, y in cal[k]:
                x, y = x.to(device), y.to(device)
                loss = lossfn(head(bb(x)), y)
                opt.zero_grad(); loss.backward(); opt.step()
                el += loss.item(); nb += 1
            sch.step()
            if ep % 50 == 0 or ep == args.epochs - 1:
                print(f"client {k} ep{ep} loss{el/max(nb,1):.3f} t{time.time()-t0:.0f}s", flush=True)
        torch.save(bb.cpu().state_dict(), ckpt)
        print(f"client {k}: saved {ckpt}  ({time.time()-t0:.0f}s)", flush=True)
        del bb, head
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("CE training done", flush=True)


if __name__ == '__main__':
    main()
