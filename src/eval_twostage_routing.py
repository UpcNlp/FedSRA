"""
Label-free TWO-STAGE client selection for FedSRA, targeting the serving-cost
limitation when K is large.

Motivation. The seen-only oracle (aggregate ONLY the clients that observed the
true class) is BOTH cheaper -- it touches a handful of clients -- AND more
accurate than full all-client RGA, but it needs the label. The naive confidence
cascade (eval_early_exit.py) fails because a small prefix is overconfident. Here
we approximate the oracle WITHOUT labels in two stages:

  Stage 1 (cheap, always-on anchor committee): pick A clients by greedy class
    coverage (server-side, from metadata only), run plain RGA over just those A
    backbones, and take the top-r predicted classes as a CANDIDATE set C-hat.
  Stage 2 (metadata-routed, restricted to C-hat): for each candidate class
    c in C-hat, aggregate the seen-c clients (sqrt(n_k) weight, label-free per-
    class mask from n_{k,c}) and score by cosine to the ETF prototype e_c;
    predict argmax over c in C-hat.

The clients actually forwarded for an input are anchor ∪ {k : k covers some
c in C-hat}; its size is the serving cost we report. Everything is inference-
only re-aggregation on features cached ONCE per client; the active-set size is
COUNTED (what deployment would forward), not enforced.

Sweeps anchor size A x candidate breadth r. Reports, per (A, r):
  recall    : fraction of inputs whose true class lands in C-hat (acc ceiling)
  acc       : final two-stage accuracy
  avg_active: average distinct clients forwarded / input  (the cost)
against full all-client RGA (cost K) and the seen-only oracle (upper bound, with
its own average covering-client count).

Usage:
  python eval_twostage_routing.py --dataset cifar100 --NL 100 --alpha 0.05 --K 50 \
      --save_dir saved_models/cifar100_a0.05_k50_s42
"""
import os, argparse, json
import numpy as np, torch, torch.nn.functional as F
from rebuild8 import generate_etf
from eval_ablation_RIJ import load_backbones, forward_features, get_test_loader_and_ccc

EPS = 1e-8
SEEN_THR = 1   # n_{k,c} >= SEEN_THR  =>  client k counts as having "seen" class c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--K', type=int, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100'])
    ap.add_argument('--NL', type=int, default=100)
    ap.add_argument('--FD', type=int, default=256)
    ap.add_argument('--calib_n', type=int, default=1024)
    ap.add_argument('--anchors', default='3,5,8', help='comma list of anchor sizes A')
    ap.add_argument('--rs', default='3,5,10,20,30', help='comma list of candidate breadths r')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    NC, NL, FD = args.K, args.NL, args.FD
    A_list = [int(x) for x in args.anchors.split(',')]
    r_list = [int(x) for x in args.rs.split(',')]
    R_max = max(r_list)

    etf = generate_etf(NL, FD)
    tl, ccc = get_test_loader_and_ccc(args.dataset, NC, args.alpha, NL)
    bbs = load_backbones(args.save_dir, NC, FD)
    all_raw, labels, _ = forward_features(bbs, [{} for _ in range(NC)], tl, etf,
                                          NC, FD, NL, use_experts=False)
    labels = np.asarray(labels); N = len(labels)
    valid = [k for k in range(NC) if all_raw[k] is not None]
    if not valid:
        raise SystemExit(f"No backbones found under {args.save_dir}")
    w = {k: float(np.sqrt(sum(ccc.get(k, {}).values()))) for k in valid}
    seen_sets = {k: {c for c, n in ccc.get(k, {}).items() if n >= SEEN_THR} for k in valid}

    # held-out split + frozen per-client z-score stats (per-sample, batch-independent)
    perm = np.random.RandomState(args.seed).permutation(N)
    calib_idx, eval_idx = perm[:args.calib_n], perm[args.calib_n:]
    ne = len(eval_idx); le = labels[eval_idx]
    ct, et = torch.as_tensor(calib_idx), torch.as_tensor(eval_idx)
    Z = {}
    for k in valid:
        mu = all_raw[k][ct].mean(0, keepdim=True); sd = all_raw[k][ct].std(0, keepdim=True)
        Z[k] = torch.nan_to_num((all_raw[k][et] - mu) / (sd + EPS), nan=0.0, posinf=0.0, neginf=0.0)

    le_t = torch.as_tensor(le)
    Kv = len(valid)
    # client coverage matrix cov[i, c] for i-th valid client
    cov = torch.zeros(Kv, NL, dtype=torch.bool)
    for i, k in enumerate(valid):
        for c in seen_sets[k]:
            cov[i, c] = True

    # ---- full all-client RGA (baseline; cost = Kv) ----
    cum = torch.zeros(ne, FD); cw = 0.0
    for k in valid:
        cum = cum + Z[k] * w[k]; cw += w[k]
    full_logits = F.normalize(cum / cw, dim=1) @ etf.T
    full_acc = float((full_logits.argmax(1).numpy() == le).mean())

    # ---- deployable per-class metadata routing scores: score_full[n, c] ----
    # score for class c = cos( L2( sum_{k seen c} sqrt(n_k) z_k ), e_c )
    score_full = torch.full((ne, NL), -1.0)
    for c in range(NL):
        feat = torch.zeros(ne, FD); ws = 0.0
        for i, k in enumerate(valid):
            if cov[i, c]:
                feat = feat + Z[k] * w[k]; ws += w[k]
        if ws == 0:
            continue
        score_full[:, c] = F.normalize(feat / ws, dim=1) @ etf[c]

    # ---- seen-only oracle (upper bound; uses true label) ----
    of = torch.zeros(ne, FD); ow = torch.zeros(ne)
    n_cover = torch.zeros(ne)                       # clients touched by the oracle
    for i, k in enumerate(valid):
        m = cov[i, le_t]                            # [ne] did client k see this sample's class
        if not m.any(): continue
        of[m] += Z[k][m] * w[k]; ow[m] += w[k]; n_cover[m] += 1
    zo = F.normalize(of / torch.clamp(ow, min=1e-12)[:, None], dim=1)
    oracle_acc = float(((zo @ etf.T).argmax(1).numpy() == le).mean())
    oracle_cost = float(n_cover.mean())

    # ---- greedy class-coverage anchor committee (label-free) ----
    def greedy_cover(A):
        chosen, covered, cand = [], set(), set(valid)
        for _ in range(min(A, Kv)):
            k = max(cand, key=lambda k: (len(seen_sets[k] - covered), w[k]))
            chosen.append(k); covered |= seen_sets[k]; cand.discard(k)
        return chosen

    results = []
    for A in A_list:
        anchor = greedy_cover(A)
        anchor_local = [valid.index(k) for k in anchor]
        # stage-1 plain RGA over anchor committee -> candidate classes
        acum = torch.zeros(ne, FD); aw = 0.0
        for k in anchor:
            acum = acum + Z[k] * w[k]; aw += w[k]
        anchor_logits = F.normalize(acum / max(aw, 1e-12), dim=1) @ etf.T
        topr_idx = anchor_logits.topk(R_max, dim=1).indices            # [ne, R_max]

        anchor_mask = torch.zeros(Kv, dtype=torch.bool); anchor_mask[anchor_local] = True
        for r in r_list:
            cand = topr_idx[:, :r]                                     # [ne, r]
            recall = float((cand == le_t[:, None]).any(1).float().mean())
            cand_scores = torch.gather(score_full, 1, cand)            # [ne, r]
            local = cand_scores.argmax(1)
            pred = cand[torch.arange(ne), local].numpy()
            acc = float((pred == le).mean())
            # active clients = anchor ∪ covers any candidate class
            cov_cand = cov[:, cand]                                    # [Kv, ne, r]
            active = cov_cand.any(2) | anchor_mask[:, None]            # [Kv, ne]
            avg_active = float(active.sum(0).float().mean())
            results.append({'A': A, 'r': r, 'recall': recall, 'acc': acc,
                            'avg_active': avg_active, 'frac_active': avg_active / Kv})

    print(f"\n[{args.dataset}] alpha={args.alpha} K={NC} | eval_n={ne} clients {Kv}/{NC}")
    print(f"  full all-client RGA : {full_acc * 100:6.2f}%   (cost = {Kv} clients/img)")
    print(f"  seen-only oracle    : {oracle_acc * 100:6.2f}%   (cost = {oracle_cost:.1f} clients/img, uses label)")
    print(f"  two-stage routing (label-free):")
    print(f"    {'A':>3s} {'r':>3s} {'recall':>7s} {'acc':>8s} {'avg_active':>11s} {'frac':>6s}")
    for x in results:
        print(f"    {x['A']:3d} {x['r']:3d} {x['recall'] * 100:6.1f}% {x['acc'] * 100:7.2f}% "
              f"{x['avg_active']:11.2f} {x['frac_active'] * 100:5.0f}%")
    # best: highest acc among configs cheaper than full (and report cheapest within 0.5pp of full)
    cheaper = [x for x in results if x['avg_active'] < Kv]
    near = [x for x in cheaper if x['acc'] >= full_acc - 0.005]
    if near:
        b = min(near, key=lambda x: x['avg_active'])
        print(f"  >> matches full (within 0.5pp) at A={b['A']} r={b['r']}: {b['acc'] * 100:.2f}% "
              f"using {b['avg_active']:.1f}/{Kv} clients ({b['frac_active'] * 100:.0f}% of cost)")
    if cheaper:
        bb = max(cheaper, key=lambda x: x['acc'])
        print(f"  >> best label-free acc: A={bb['A']} r={bb['r']}: {bb['acc'] * 100:.2f}% "
              f"({(bb['acc'] - full_acc) * 100:+.2f}pp vs full) at {bb['avg_active']:.1f}/{Kv} clients")

    out = args.out or f"results/twostage_{args.dataset}_a{args.alpha}_k{NC}_s{args.seed}.json"
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    json.dump({'dataset': args.dataset, 'alpha': args.alpha, 'K': NC, 'seed': args.seed,
               'eval_n': int(ne), 'n_clients': Kv, 'full_acc': full_acc,
               'oracle_acc': oracle_acc, 'oracle_cost': oracle_cost, 'results': results},
              open(out, 'w'), indent=2)
    print(f"  saved {out}")


if __name__ == '__main__':
    main()
