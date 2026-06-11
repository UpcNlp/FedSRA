"""
Confidence-ordered early-exit (cascade) inference for FedSRA / RGA.

Addresses the serving-cost limitation: instead of evaluating ALL K client
backbones for every input, evaluate them in descending sqrt(n_k) order -- the
reliability prior the server already holds, computed without any label -- and
STOP as soon as the running RGA aggregate is confident (top-1 minus top-2 ETF
cosine margin >= tau). Easy inputs exit after a few clients; hard inputs draw
on more. This trades a confidence threshold for average serving cost while the
RGA cancellation still operates within whatever prefix is evaluated.

Inference-only. Features are extracted ONCE per client (the only GPU step);
every curve below is pure re-aggregation on the cached features. Standardization
uses FROZEN calibration stats (the paper's deployment mode), so each sample's
prediction is independent of the batch -- required for a per-sample cascade.

Reports, on one held-out eval split:
  - all-K full RGA accuracy (baseline; cost = K backbones / input)
  - fixed top-m by weight (non-adaptive truncation): accuracy vs m
  - confidence early-exit: accuracy vs AVERAGE clients evaluated, sweeping tau
  - label-aware seen-only oracle (upper bound; uses the true label, NOT deployable)

Usage:
  python eval_early_exit.py --dataset cifar100 --NL 100 --alpha 0.05 --K 50 \
      --save_dir saved_models/cifar100_a0.05_k50_s42
"""
import os, argparse, json
import numpy as np, torch, torch.nn.functional as F
from rebuild8 import generate_etf
from eval_ablation_RIJ import load_backbones, forward_features, get_test_loader_and_ccc

EPS = 1e-8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--K', type=int, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100'])
    ap.add_argument('--NL', type=int, default=100, help='num classes (10 / 100)')
    ap.add_argument('--FD', type=int, default=256)
    ap.add_argument('--calib_n', type=int, default=1024,
                    help='size of the unlabeled calibration pool for frozen stats')
    ap.add_argument('--order', default='weight', choices=['weight', 'random'],
                    help='label-free client visit order for the cascade')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    NC, NL, FD = args.K, args.NL, args.FD
    etf = generate_etf(NL, FD)                       # [NL, FD], unit-norm rows
    tl, ccc = get_test_loader_and_ccc(args.dataset, NC, args.alpha, NL)
    bbs = load_backbones(args.save_dir, NC, FD)
    all_raw, labels, _ = forward_features(bbs, [{} for _ in range(NC)], tl, etf,
                                          NC, FD, NL, use_experts=False)
    labels = np.asarray(labels); N = len(labels)
    valid = [k for k in range(NC) if all_raw[k] is not None]
    if not valid:
        raise SystemExit(f"No backbones found under {args.save_dir}")
    w = {k: float(np.sqrt(sum(ccc.get(k, {}).values()))) for k in valid}

    # deterministic calibration / evaluation split
    perm = np.random.RandomState(args.seed).permutation(N)
    calib_idx = perm[:args.calib_n]; eval_idx = perm[args.calib_n:]
    ne = len(eval_idx); le = labels[eval_idx]
    ct = torch.as_tensor(calib_idx); et = torch.as_tensor(eval_idx)

    # frozen per-client z-score stats from the calib pool, applied per-sample to eval
    Z = {}
    for k in valid:
        mu = all_raw[k][ct].mean(0, keepdim=True)
        sd = all_raw[k][ct].std(0, keepdim=True)
        f = all_raw[k][et]
        Z[k] = torch.nan_to_num((f - mu) / (sd + EPS), nan=0.0, posinf=0.0, neginf=0.0)

    # label-free client visit order
    if args.order == 'weight':
        ordering = sorted(valid, key=lambda k: (-w[k], k))
    else:
        rng = np.random.RandomState(args.seed + 1)
        ordering = list(valid); rng.shuffle(ordering)
    M = len(ordering)

    # cumulative RGA aggregate over the ordered prefix; per-prefix pred + margin
    cum = torch.zeros(ne, FD); cw = 0.0
    margins = np.empty((M, ne), dtype=np.float32)
    preds = np.empty((M, ne), dtype=np.int64)
    accs_prefix = np.empty(M, dtype=np.float64)          # fixed top-m accuracy
    for j, k in enumerate(ordering):
        cum = cum + Z[k] * w[k]; cw += w[k]
        z = F.normalize(cum / max(cw, 1e-12), dim=1)
        logits = z @ etf.T                                # [ne, NL] cosine sims
        top2 = torch.topk(logits, 2, dim=1).values
        margins[j] = (top2[:, 0] - top2[:, 1]).numpy()
        p = logits.argmax(1).numpy(); preds[j] = p
        accs_prefix[j] = float((p == le).mean())
    full_acc = float(accs_prefix[-1])                     # all-K RGA baseline

    # confidence early-exit: first prefix whose margin >= tau, else all K
    taus = [round(float(x), 3) for x in np.linspace(0.0, 0.6, 13)]
    ee = []
    for tau in taus:
        hit = margins >= tau                              # [M, ne]
        exit_j = np.where(hit.any(0), hit.argmax(0), M - 1)
        pred = preds[exit_j, np.arange(ne)]
        acc = float((pred == le).mean())
        avg_cost = float(exit_j.mean() + 1)
        ee.append({'tau': tau, 'acc': acc, 'avg_clients': avg_cost,
                   'frac_clients': avg_cost / M})

    # label-aware seen-only oracle (upper bound; uses true label, not deployable)
    seen = {k: set(ccc.get(k, {}).keys()) for k in valid}
    of = torch.zeros(ne, FD); ow = np.zeros(ne)
    for k in valid:
        mask = np.isin(le, list(seen[k])) if seen[k] else np.zeros(ne, bool)
        if not mask.any(): continue
        mt = torch.as_tensor(mask)
        of[mt] += Z[k][mt] * w[k]; ow[mask] += w[k]
    zo = F.normalize(of / torch.as_tensor(np.maximum(ow, 1e-12), dtype=torch.float32)[:, None], dim=1)
    oracle_acc = float(((zo @ etf.T).argmax(1).numpy() == le).mean())

    # headline: cheapest tau within 0.5 pp of full accuracy
    within = [r for r in ee if r['acc'] >= full_acc - 0.005]
    best = min(within, key=lambda r: r['avg_clients']) if within else ee[0]
    m_match = int(np.ceil(best['avg_clients']))
    fixed_match_acc = float(accs_prefix[min(m_match, M) - 1])

    print(f"\n[{args.dataset}] alpha={args.alpha} K={NC} order={args.order} | "
          f"eval_n={ne} calib_n={len(calib_idx)} | clients {M}/{NC}")
    print(f"  all-K full RGA              : {full_acc * 100:6.2f}%   (cost = {M} clients/img)")
    print(f"  seen-only oracle (upper bnd): {oracle_acc * 100:6.2f}%")
    print("  confidence early-exit (sweep tau):")
    print(f"    {'tau':>6s} {'acc':>8s} {'avg_clients':>12s} {'frac':>7s}")
    for r in ee:
        print(f"    {r['tau']:6.3f} {r['acc'] * 100:7.2f}% {r['avg_clients']:12.2f} "
              f"{r['frac_clients'] * 100:6.1f}%")
    print(f"  >> within 0.5pp of full at tau={best['tau']}: "
          f"{best['acc'] * 100:.2f}% using {best['avg_clients']:.1f}/{M} clients "
          f"({best['frac_clients'] * 100:.0f}% of cost)")
    print(f"     vs fixed top-{m_match} truncation: {fixed_match_acc * 100:.2f}% "
          f"(adaptive advantage {(best['acc'] - fixed_match_acc) * 100:+.2f} pp)")

    out = args.out or (f"results/earlyexit_{args.dataset}_a{args.alpha}"
                       f"_k{NC}_{args.order}_s{args.seed}.json")
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    json.dump({
        'dataset': args.dataset, 'alpha': args.alpha, 'K': NC, 'seed': args.seed,
        'order': args.order, 'eval_n': int(ne), 'calib_n': int(len(calib_idx)),
        'n_clients': M, 'full_acc': full_acc, 'oracle_acc': oracle_acc,
        'fixed_topm_acc': [float(a) for a in accs_prefix],
        'early_exit': ee,
        'headline': {'tau': best['tau'], 'acc': best['acc'],
                     'avg_clients': best['avg_clients'],
                     'frac_clients': best['frac_clients'],
                     'fixed_match_m': m_match, 'fixed_match_acc': fixed_match_acc},
    }, open(out, 'w'), indent=2)
    print(f"  saved {out}")


if __name__ == '__main__':
    main()
