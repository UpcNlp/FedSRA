"""
run_ablation_RIJ.py — R/I/J backbone ablation training for two-knowledge analysis.

Trains a single (dataset, loss_type, α, K) cell:
  --loss_type R  : etf_cl only (relational/NC2; cross-entropy push-apart)
  --loss_type I  : etf_al only (intrinsic/NC1;  alignment pull-to-prototype)
  --loss_type J  : etf_cl + 0.5*etf_al (joint = current default)

For loss_type=J, also trains per-class conditional experts (used by the
J+Expert ablation row). For R and I, experts are not trained (those rows
are pure-backbone-aggregation evaluations).

All three backbone variants share identical optimizer, scheduler, augmentation,
init, and seed → row-to-row differences are isolated to the loss function.

Saved weights:
  saved_models/ablation_{dataset}/{R,I,J}_a{α}_k{K}_s{seed}/
    client_{k}/backbone.pt
    client_{k}/expert_{c}.pt    (J only)
    ccc.pt

Meta (timings, R-fallback statistics, GPU peak):
  results/ablation_RIJ_meta_{dataset}_{R,I,J}_a{α}_k{K}_s{seed}.json

Usage:
  python run_ablation_RIJ.py --dataset cifar100 --loss_type R \
      --alpha 0.05 --n_clients 10 --seed 42
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json, time, os, argparse
from resnet18_filter_merge import ResNet18Backbone
from rebuild8_tinyimagenet import ResNet18Backbone64, prepare_data_tinyimagenet, train_experts_tiny
from rebuild8 import (
    generate_etf, train_bb, ConditionalExpert,
    device, USE_BF16,
    prepare_data,
)


def get_dataset(name, NC, ALPHA, NL):
    """Returns (cal, ccl, tl, ccc) tuple for the requested dataset."""
    if name == 'cifar10':
        return prepare_data(NC, ALPHA, NL)
    elif name == 'cifar100':
        from rebuild8_cifar100 import prepare_data_cifar100
        return prepare_data_cifar100(NC, ALPHA, NL)
    elif name == 'tinyimagenet':
        return prepare_data_tinyimagenet(NC, ALPHA, NL)
    raise ValueError(name)


def train_experts_wrap(name, bb, ccl_k, classes, etf, ccc_k, NL, FD, LD, EPE):
    """Dispatch to dataset-specific train_experts."""
    if name == 'cifar10':
        from rebuild8 import train_experts
        return train_experts(bb, ccl_k, classes, etf, NL, FD, LD, EPE)
    elif name == 'cifar100':
        from rebuild8_cifar100 import train_experts_cifar100
        return train_experts_cifar100(bb, ccl_k, classes, etf, ccc_k,
                                       NL, FD, LD, EPE)
    else:
        return train_experts_tiny(bb, ccl_k, classes, etf, ccc_k,
                                   NL, FD, LD, EPE)


def save_backbones(bbs, ccc, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for k in range(len(bbs)):
        if bbs[k] is None: continue
        cd = os.path.join(save_dir, f"client_{k}")
        os.makedirs(cd, exist_ok=True)
        torch.save(bbs[k].state_dict(), os.path.join(cd, "backbone.pt"))
    torch.save(dict(ccc), os.path.join(save_dir, "ccc.pt"))
    print(f"  Backbones saved -> {save_dir}")


def save_experts(client_exps, save_dir):
    for k in range(len(client_exps)):
        cd = os.path.join(save_dir, f"client_{k}")
        if not os.path.exists(cd): continue
        for c, exp in client_exps[k].items():
            torch.save(exp.state_dict(), os.path.join(cd, f"expert_{c}.pt"))
    print(f"  Experts saved   -> {save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'cifar100', 'tinyimagenet'])
    parser.add_argument('--loss_type', type=str, required=True,
                        choices=['R', 'I', 'J', 'PR'])
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--n_clients', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs_bb', type=int, default=None,
                        help='Backbone epochs. Default 600 (cifar10) / 300 (cifar100).')
    parser.add_argument('--epochs_exp', type=int, default=None,
                        help='Expert epochs. Default 600 (cifar10) / 200 (cifar100).')
    parser.add_argument('--skip_experts', action='store_true',
                        help='Skip expert training even when loss_type=J.')
    parser.add_argument('--resume', action='store_true',
                        help='If saved backbones already exist, skip backbone training.')
    args = parser.parse_args()

    ALPHA = args.alpha; NC = args.n_clients; SEED = args.seed
    NL = {'cifar10': 10, 'cifar100': 100, 'tinyimagenet': 200}[args.dataset]
    FD = 256; LD = 32
    EPB = args.epochs_bb if args.epochs_bb else (600 if args.dataset == 'cifar10' else 300)
    EPE = args.epochs_exp if args.epochs_exp else (600 if args.dataset == 'cifar10' else 200)
    torch.manual_seed(SEED); np.random.seed(SEED)

    save_dir = (f"saved_models/ablation_{args.dataset}/"
                f"{args.loss_type}_a{ALPHA}_k{NC}_s{SEED}")

    print(f"\n{'='*68}")
    print(f"  Ablation RIJ training")
    print(f"  dataset={args.dataset}, loss={args.loss_type}, "
          f"α={ALPHA}, K={NC}, seed={SEED}")
    print(f"  epochs: bb={EPB}  exp={EPE}")
    print(f"  save_dir: {save_dir}")
    print(f"{'='*68}")

    etf = generate_etf(NL, FD)
    cal, ccl, tl, ccc = get_dataset(args.dataset, NC, ALPHA, NL)

    # ── 1) Backbones ────────────────────────────────────────────
    bbs = [None] * NC
    r_fallback_clients = []

    skip_backbone = False
    if args.resume:
        existing = sum(1 for k in range(NC)
                       if os.path.exists(os.path.join(save_dir, f"client_{k}", "backbone.pt")))
        non_empty = sum(1 for k in range(NC) if ccc.get(k, {}))
        if existing == non_empty and existing > 0:
            print(f"  [resume] all {existing} backbones already saved; skipping training.")
            skip_backbone = True

    if not skip_backbone:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for k in range(NC):
            cls = sorted(ccc.get(k, {}).keys())
            if not cls:
                print(f"  Client {k}: EMPTY (skip)"); continue
            print(f"  Client {k}: {len(cls)} cls, {sum(ccc[k].values())} samp")
            if args.loss_type == 'R' and len(cls) < 2:
                r_fallback_clients.append(k)
                print(f"    [R-fallback] ncl<2 -> using etf_al instead of etf_cl")
            BBCls = ResNet18Backbone64 if args.dataset == 'tinyimagenet' else ResNet18Backbone
            bb = BBCls(FD)
            bb = train_bb(bb, cal[k], cls, etf, EPB,
                          save_dir=None, client_id=k,
                          save_every=10**9,         # disable mid-train snapshot
                          loss_type=args.loss_type)
            bbs[k] = bb.cpu()
            torch.cuda.empty_cache()
        train_bb_time = time.time() - t0
        gpu_peak_mb = (torch.cuda.max_memory_allocated() / 1024**2
                       if torch.cuda.is_available() else 0.0)
        save_backbones(bbs, ccc, save_dir)
    else:
        # Load from disk for downstream expert training
        train_bb_time = 0.0
        gpu_peak_mb = 0.0
        for k in range(NC):
            cd = os.path.join(save_dir, f"client_{k}")
            bp = os.path.join(cd, "backbone.pt")
            if os.path.exists(bp):
                BBCls = ResNet18Backbone64 if args.dataset == 'tinyimagenet' else ResNet18Backbone
                bb = BBCls(FD)
                bb.load_state_dict(torch.load(bp, map_location='cpu', weights_only=True))
                bbs[k] = bb

    # ── 2) Experts (only for J) ────────────────────────────────
    client_exps = [{} for _ in range(NC)]
    train_exp_time = 0.0
    if args.loss_type == 'J' and not args.skip_experts:
        # Skip if all experts already exist
        all_exp_exist = True
        for k in range(NC):
            if bbs[k] is None: continue
            cls = sorted(ccc.get(k, {}).keys())
            cd = os.path.join(save_dir, f"client_{k}")
            for c in cls:
                if args.dataset == 'cifar100' and ccc.get(k, {}).get(c, 0) < 5:
                    continue  # cifar100 skips small classes
                if not os.path.exists(os.path.join(cd, f"expert_{c}.pt")):
                    all_exp_exist = False; break
            if not all_exp_exist: break

        if args.resume and all_exp_exist:
            print(f"  [resume] all experts already saved; skipping.")
        else:
            t0 = time.time()
            for k in range(NC):
                if bbs[k] is None: continue
                cls = sorted(ccc.get(k, {}).keys())
                print(f"  Client {k} experts ({len(cls)} cls)")
                bbs[k] = bbs[k].to(device)
                exps = train_experts_wrap(args.dataset, bbs[k], ccl[k], cls,
                                          etf, ccc.get(k, {}),
                                          NL, FD, LD, EPE)
                client_exps[k] = {c: exp.cpu() for c, exp in exps.items()}
                bbs[k] = bbs[k].cpu()
                torch.cuda.empty_cache()
            train_exp_time = time.time() - t0
            save_experts(client_exps, save_dir)

    # ── 3) Meta ──────────────────────────────────────────────
    os.makedirs('results', exist_ok=True)
    meta = {
        'dataset': args.dataset,
        'loss_type': args.loss_type,
        'alpha': ALPHA, 'n_clients': NC, 'seed': SEED,
        'epochs_bb': EPB, 'epochs_exp': EPE,
        'train_bb_time_sec': round(train_bb_time, 1),
        'train_exp_time_sec': round(train_exp_time, 1),
        'gpu_peak_mb': round(gpu_peak_mb, 1),
        'r_fallback_clients': r_fallback_clients,
        'r_fallback_count': len(r_fallback_clients),
        'has_experts': (args.loss_type == 'J' and not args.skip_experts),
        'save_dir': save_dir,
    }
    meta_path = (f"results/ablation_RIJ_meta_"
                 f"{args.dataset}_{args.loss_type}_a{ALPHA}_k{NC}_s{SEED}.json")
    json.dump(meta, open(meta_path, 'w'), indent=2)
    print(f"\n  Meta -> {meta_path}")
    print(f"  bb_time={train_bb_time:.1f}s  exp_time={train_exp_time:.1f}s  "
          f"peak={gpu_peak_mb:.0f}MB  R_fallback={len(r_fallback_clients)}/{NC}")


if __name__ == '__main__':
    main()
