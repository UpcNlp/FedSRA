"""
Dump per-sample FedDSI (full=J) fused class probabilities, reusing the CORRECT
eval_ablation_RIJ pipeline (raw fc feature -> znorm+sqrt aggregate), which
reproduces the reported ~0.84 at cifar10 a0.05 k5. (My earlier export_decision_feats
double-normalized via bb(x)=F.normalize(fc) and was wrong.)

fused = z-norm(union_logits) + alpha_f * en ; probs = softmax(fused).
"""
import os, json, argparse
import numpy as np, torch, torch.nn.functional as F

import rebuild8
from rebuild8 import prepare_data, generate_etf, device
from eval_ablation_RIJ import load_backbones, load_experts, forward_features, znorm_sqrt_aggregate

NL, FD = 10, 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--K', type=int, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--alpha_f', type=float, default=0.3)   # J+Expert fusion weight (paper)
    ap.add_argument('--min_n', type=int, default=10)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    NC = args.K
    etf = generate_etf(NL, FD)
    _, _, tl, ccc = prepare_data(NC, args.alpha, NL)

    bbs = load_backbones(args.save_dir, NC, FD)
    client_exps = load_experts(args.save_dir, NC, NL, FD)
    all_raw, labels, errors = forward_features(bbs, client_exps, tl, etf, NC, FD, NL,
                                               use_experts=True)
    N = len(labels)

    feat_agg = znorm_sqrt_aggregate(all_raw, ccc, NC, N, FD)
    union_logits = feat_agg @ etf.T
    un = (union_logits - union_logits.mean(1, keepdim=True)) / (union_logits.std(1, keepdim=True) + 1e-8)

    sample_count = {(k, c): ccc.get(k, {}).get(c, 0) for k in range(NC) for c in ccc.get(k, {})}
    ensemble = torch.zeros(N, NL); weight_sum = torch.zeros(N, NL)
    for k in range(NC):
        ek = errors[k].clone(); ek[ek == float('inf')] = 1e6
        cl = torch.zeros(N, NL)
        for c in range(NL):
            n = sample_count.get((k, c), 0)
            if n < args.min_n: continue
            cl[:, c] = -ek[:, c]; weight_sum[:, c] += np.log(n + 1)
        valid = (cl != 0)
        if valid.any():
            cln = (cl - cl.mean(1, keepdim=True)) / (cl.std(1, keepdim=True) + 1e-8)
            cln[~valid] = 0
            ensemble += cln * weight_sum.clamp(min=0)
    has = (ensemble.abs().sum(1) > 0)
    en = (ensemble - ensemble.mean(1, keepdim=True)) / (ensemble.std(1, keepdim=True) + 1e-8)
    en[~has.unsqueeze(1).expand_as(en)] = 0

    fused = un + args.alpha_f * en
    rel_probs = F.softmax(un, dim=1).numpy()          # relational signal (ERL/ETF)
    int_probs = F.softmax(en, dim=1).numpy()          # intrinsic signal (experts)
    fused_probs = F.softmax(fused, dim=1).numpy()     # dual-signal fusion (DAF)
    acc_union = float((un.argmax(1).numpy() == labels).mean())
    acc_int = float((en.argmax(1).numpy() == labels).mean())
    acc_fused = float((fused.argmax(1).numpy() == labels).mean())
    print(f"FedDSI(J) a={args.alpha} K={NC} | rel={acc_union:.4f} int={acc_int:.4f} "
          f"fused(af={args.alpha_f})={acc_fused:.4f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, probs=fused_probs.astype(np.float16),
                        rel_probs=rel_probs.astype(np.float16),
                        int_probs=int_probs.astype(np.float16),
                        fused_probs=fused_probs.astype(np.float16),
                        labels=labels.astype(np.int16),
                        acc_union=acc_union, acc_int=acc_int, acc_fused=acc_fused, alpha_f=args.alpha_f)
    print("saved", args.out, flush=True)


if __name__ == '__main__':
    main()
