"""
Export per-sample DUAL-SIGNAL decision representations for the crossover t-SNE
grid.  INFERENCE ONLY (loads trained backbones + experts, no training).

For one (alpha, K) cell it reproduces FedDSI's inference exactly as in
run_loss_sweep.py and saves, per test sample (N=10000):
  - rel_feat[N, FD]   GPA-aggregated relational feature (z-norm + sqrt(n) pool)
  - rel_logits[N, C]  z-normed relational logits  (un)        -- "relational signal"
  - int_logits[N, C]  z-normed intrinsic ensemble logits (en) -- "intrinsic signal"
  - fused[N, C]       un + alpha_f * en at the adaptive alpha_f
  - labels[N], alpha_f, and sanity accs (union / expert / fused)

Usage:
  python export_decision_feats.py --alpha 0.05 --K 10 \
      --ckpt_dir saved_models/a0.05_k10_s42 --out tmp/decision_a0.05_k10.npz
"""
import os, json, argparse
import numpy as np
import torch
import torch.nn.functional as F

from rebuild8 import device, prepare_data, generate_etf, ConditionalExpert
from resnet18_filter_merge import ResNet18Backbone

NL, FD, LD = 10, 256, 32
GAMMA, NMIN = 0.2, 10          # adaptive alpha_f = GAMMA * coverage.mean()


def znorm(x):
    return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--K', type=int, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    etf = generate_etf(NL, FD); ed = etf.to(device)
    cal, ccl, tl, ccc = prepare_data(args.K, args.alpha, NL)
    N = 10000

    errors = torch.full((args.K, N, NL), float('inf'))
    all_raw = [None] * args.K
    all_labels = []
    for k in range(args.K):
        cls = sorted(ccc.get(k, {}).keys())
        if not cls:
            continue
        bb = ResNet18Backbone(FD)
        bb.load_state_dict(torch.load(f"{args.ckpt_dir}/client_{k}/backbone.pt", map_location='cpu'))
        bb = bb.to(device).eval()
        exps = {}
        for c in cls:
            ep = f"{args.ckpt_dir}/client_{k}/expert_{c}.pt"
            if not os.path.exists(ep):
                continue
            e = ConditionalExpert(FD, FD, 128, LD)
            e.load_state_dict(torch.load(ep, map_location='cpu'))
            exps[c] = e.to(device).eval()
        feats, offset = [], 0
        with torch.no_grad():
            for x, y in tl:
                x = x.to(device); bs = x.size(0)
                feat = bb(x)
                feats.append(feat.float().cpu())
                fn = F.normalize(feat, dim=1)
                for c, e in exps.items():
                    fr, _ = e(fn, ed[c].unsqueeze(0).expand(bs, -1))
                    errors[k, offset:offset+bs, c] = ((fn - fr) ** 2).mean(1).cpu()
                if k == 0:
                    all_labels.append(y)
                offset += bs
        all_raw[k] = torch.cat(feats, 0)
        bb = bb.cpu()
        print(f"  client {k}: {len(exps)} experts, feats {all_raw[k].shape}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    labels = torch.cat(all_labels).numpy()

    sample_count = {(k, c): ccc.get(k, {}).get(c, 0)
                    for k in range(args.K) for c in ccc.get(k, {})}

    # ---- relational signal: z-norm + sqrt(n) post-norm aggregation ----
    feat = torch.zeros(N, FD); w_sum = 0.0
    for k in range(args.K):
        if all_raw[k] is None:
            continue
        f = all_raw[k]
        f_z = (f - f.mean(0, keepdim=True)) / (f.std(0, keepdim=True) + 1e-8)
        w = float(np.sqrt(sum(ccc.get(k, {}).values())))
        feat += f_z * w; w_sum += w
    rel_feat = F.normalize(feat / w_sum, dim=1)
    union_logits = rel_feat @ etf.T
    un = znorm(union_logits)

    # ---- intrinsic signal: per-client expert-error ensemble (en) ----
    ensemble = torch.zeros(N, NL); weight_sum = torch.zeros(N, NL)
    for k in range(args.K):
        ek = errors[k].clone(); ek[ek == float('inf')] = 1e6
        cl = torch.zeros(N, NL)
        for c in range(NL):
            n = sample_count.get((k, c), 0)
            if n < NMIN:
                continue
            cl[:, c] = -ek[:, c]
            weight_sum[:, c] += np.log(n + 1)
        valid = (cl != 0)
        if valid.any():
            cln = (cl - cl.mean(1, keepdim=True)) / (cl.std(1, keepdim=True) + 1e-8)
            cln[~valid] = 0
            ensemble += cln * weight_sum.clamp(min=0)
    has = (ensemble.abs().sum(1) > 0)
    en = (ensemble - ensemble.mean(1, keepdim=True)) / (ensemble.std(1, keepdim=True) + 1e-8)
    en[~has.unsqueeze(1).expand_as(en)] = 0

    # ---- adaptive alpha_f + fused ----
    cov = [sum(1 for k in range(args.K) if ccc.get(k, {}).get(c, 0) >= NMIN) / args.K
           for c in range(NL)]
    alpha_f = GAMMA * float(np.mean(cov))
    fused = un + alpha_f * en

    acc_union = float((un.argmax(1).numpy() == labels).mean())
    err_min = errors.clone(); err_min[err_min == float('inf')] = 1e6
    acc_expert = float((err_min.min(0)[0].argmin(1).numpy() == labels).mean())
    acc_fused = float((fused.argmax(1).numpy() == labels).mean())
    print(f"  a={args.alpha} K={args.K} | union={acc_union:.3f} expert={acc_expert:.3f} "
          f"fused(af={alpha_f:.3f})={acc_fused:.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out, alpha=args.alpha, K=args.K, alpha_f=alpha_f,
        rel_feat=rel_feat.numpy().astype(np.float16),
        rel_logits=un.numpy().astype(np.float32),
        int_logits=en.numpy().astype(np.float32),
        fused=fused.numpy().astype(np.float32),
        labels=labels.astype(np.int16),
        acc_union=acc_union, acc_expert=acc_expert, acc_fused=acc_fused,
        coverage=float(np.mean(cov)),
    )
    print(f"saved {args.out}", flush=True)


if __name__ == '__main__':
    main()
