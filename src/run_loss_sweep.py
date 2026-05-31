"""
run_loss_sweep.py - Loss-weight sensitivity (lambda_al / tau) for FedDSI.
====================================================================
Trains ResNet-18 backbones + per-class experts under a chosen backbone-loss
weight and evaluates FedDSI accuracy at the *adaptive* fusion weight
alpha_f = gamma * coverage.mean() (the paper's parameter-free rule).

This is the training-time analogue of the inference-time alpha_f sweep:
each (lambda_al, tau) value needs its own backbone, so cells are expensive
(~4h at K=5). Defaults reproduce the main-table config (lambda_al=0.5, tau=0.1).

Usage:
  python run_loss_sweep.py --alpha 0.05 --lambda_al 0.25 --seed 42
  python run_loss_sweep.py --alpha 0.05 --tau 0.2 --seed 42
"""
import sys
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json, time, os, argparse
from tqdm import tqdm

import rebuild8
rebuild8.DL_KWARGS = dict(num_workers=4, pin_memory=True, persistent_workers=False)

from rebuild8 import (prepare_data, generate_etf, ConditionalExpert,
                       device, USE_BF16,
                       cross_client_per_client_logits, expert_original,
                       etf_cl, etf_al, train_exp, preextract)
from resnet18_filter_merge import ResNet18Backbone

GAMMA = 0.2     # adaptive alpha_f scale (paper Eq. alpha_f)
NMIN = 10       # min per-class samples for a valid expert / coverage


def train_bb_backbone(bb, loader, classes, etf, epochs=600, lr=1e-3,
                      lambda_al=0.5, tau=0.1):
    bb = bb.to(device); ed = etf.to(device); ncl = len(classes); bb.train()
    opt = torch.optim.Adam(bb.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    amp = (torch.amp.autocast('cuda', dtype=torch.bfloat16) if USE_BF16
           else torch.amp.autocast('cuda', enabled=False))
    pbar = tqdm(range(epochs), desc="  BB", ncols=100, mininterval=10.0, file=sys.stdout)
    for ep in pbar:
        el = 0; nb = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            with amp:
                f = bb(x)
                # --- backbone loss: discriminative (etf_cl, temp=tau) + lambda_al * alignment ---
                if ncl >= 2:
                    loss = etf_cl(f, y, ed, temp=tau) + lambda_al * etf_al(f, y, ed)
                else:
                    loss = etf_al(f, y, ed)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            el += loss.item(); nb += 1
        sch.step()
        pbar.set_postfix(loss=f"{el/max(nb,1):.3f}")
    pbar.close()
    return bb


def train_experts_backbone(bb, cls_loaders, classes, etf, nc, fdim, ldim, epochs, lr=1e-3):
    bb.eval(); ed = etf.to(device); exps = {}
    for cls in classes:
        cached = preextract(bb, cls_loaders[cls])
        exp = ConditionalExpert(fdim, fdim, 128, ldim).to(device)
        om = torch.tensor([k for k in range(nc) if k != cls], device=device)
        exp = train_exp(exp, cached, ed[cls], ed, om, fdim, epochs, lr)
        exps[cls] = exp.cpu()
    return exps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--lambda_al', type=float, default=0.5)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--n_clients', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=600)       # backbone epochs
    parser.add_argument('--expert_epochs', type=int, default=600)
    args = parser.parse_args()

    ALPHA = args.alpha; SEED = args.seed
    LAMBDA_AL = args.lambda_al; TAU = args.tau
    NC = args.n_clients; NL = 10; FD = 256; LD = 32
    EPB = args.epochs; EPE = args.expert_epochs
    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    print(f"\n{'='*64}")
    print(f"  LOSS SWEEP | lambda_al={LAMBDA_AL} tau={TAU} | a={ALPHA} K={NC} s={SEED}")
    print(f"{'='*64}")

    etf = generate_etf(NL, FD)
    cal, ccl, tl, ccc = prepare_data(NC, ALPHA, NL)

    bbs = []; client_exps = []
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for k in range(NC):
        cls = sorted(ccc.get(k, {}).keys())
        if not cls:
            bbs.append(None); client_exps.append({}); continue
        print(f"  Client {k}: {len(cls)} cls, {sum(ccc[k].values())} samp", flush=True)
        bb = ResNet18Backbone(FD)
        bb = train_bb_backbone(bb, cal[k], cls, etf, EPB, lambda_al=LAMBDA_AL, tau=TAU)
        exps = train_experts_backbone(bb, ccl[k], cls, etf, NL, FD, LD, EPE)
        bbs.append(bb.cpu()); client_exps.append(exps)
        torch.cuda.empty_cache()
    train_time = time.time() - t0

    # ---- inference: build relational + intrinsic signals ----
    ed = etf.to(device); N = 10000
    all_raw = []; all_labels = []
    errors = torch.full((NC, N, NL), float('inf'))
    for k in range(NC):
        if bbs[k] is None: all_raw.append(None); continue
        bbs[k] = bbs[k].to(device); bbs[k].eval()
        for c in client_exps[k]: client_exps[k][c] = client_exps[k][c].to(device)
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
                f_norm = F.normalize(feat, dim=1)
                for c, exp in client_exps[k].items():
                    fr, _ = exp(f_norm, ed[c].unsqueeze(0).expand(bs, -1))
                    errors[k, offset:offset+bs, c] = ((f_norm - fr)**2).mean(1).cpu()
                if k == 0: all_labels.append(y)
                offset += bs
        all_raw.append(torch.cat(feats, 0))
        bbs[k] = bbs[k].cpu()
        for c in client_exps[k]: client_exps[k][c] = client_exps[k][c].cpu()
        torch.cuda.empty_cache()
    labels = torch.cat(all_labels).numpy()

    sample_count = {}
    for k in range(NC):
        for c in client_exps[k]:
            sample_count[(k, c)] = ccc.get(k, {}).get(c, 0)

    # relational (znorm + sqrt(n) post-norm aggregation)
    feat = torch.zeros(N, FD); w_sum = 0
    for k in range(NC):
        if all_raw[k] is None: continue
        f = all_raw[k]
        f_z = (f - f.mean(0, keepdim=True)) / (f.std(0, keepdim=True) + 1e-8)
        w = np.sqrt(sum(ccc.get(k, {}).values()))
        feat += f_z * w; w_sum += w
    feat_n = F.normalize(feat / w_sum, dim=1)
    union_logits = feat_n @ etf.T
    acc_union = (union_logits.argmax(1).numpy() == labels).mean()

    sl, _ = union_logits.sort(dim=1, descending=True)
    data = {
        'errors': errors, 'union_logits': union_logits,
        'union_preds': union_logits.argmax(1).numpy(),
        'union_margin': (sl[:, 0] - sl[:, 1]).numpy(),
        'labels': labels, 'sample_count': sample_count,
        'K': NC, 'N': N, 'nc': NL,
    }
    acc_expert = (expert_original(data) == labels).mean()

    # ---- adaptive alpha_f = gamma * coverage.mean() ----
    cov = [sum(1 for k in range(NC) if ccc.get(k, {}).get(c, 0) >= NMIN) / NC
           for c in range(NL)]
    adaptive_af = GAMMA * float(np.mean(cov))
    acc_adaptive = (cross_client_per_client_logits(data, alpha=adaptive_af, min_n=NMIN) == labels).mean()

    # reference grid (for sanity / robustness band)
    acc_full = {}
    for a in [0.1, 0.3, 0.5, 1.0]:
        acc_full[a] = float((cross_client_per_client_logits(data, alpha=a, min_n=NMIN) == labels).mean())
    best_a = max(acc_full, key=acc_full.get)

    print(f"\n  union(rel)={acc_union*100:.2f}  expert(int)={acc_expert*100:.2f}  "
          f"adaptive(af={adaptive_af:.3f})={acc_adaptive*100:.2f}  best={acc_full[best_a]*100:.2f}")

    os.makedirs('results', exist_ok=True)
    out = {
        'method': 'loss_sweep', 'backbone': 'resnet18',
        'lambda_al': LAMBDA_AL, 'tau': TAU,
        'alpha': ALPHA, 'n_clients': NC, 'seed': SEED,
        'epochs': EPB, 'expert_epochs': EPE, 'train_time': train_time,
        'acc_union': float(acc_union), 'acc_expert': float(acc_expert),
        'adaptive_af': adaptive_af, 'acc_adaptive': float(acc_adaptive),
        'acc_full': {str(k): v for k, v in acc_full.items()},
        'best_alpha': float(best_a), 'best_acc': float(acc_full[best_a]),
    }
    path = (f"results/loss_sweep_lal{LAMBDA_AL}_tau{TAU}"
            f"_a{ALPHA}_k{NC}_s{SEED}.json")
    json.dump(out, open(path, 'w'), indent=2)
    print(f"  Saved: {path}")


if __name__ == '__main__':
    main()
