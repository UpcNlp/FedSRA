"""
Fig.2 (Challenge 2: aggregation under non-uniform client reliability) data dump.

Inference-only. Loads already-trained ETF-anchored backbones (no training) and,
for the FULL test set, dumps everything the reliability / noise-cancellation
panels need:

  cos_single[K, N, C] : each client's per-sample raw-fc feature cos to the C ETF
                        prototypes.  reliability of client k on sample i w.r.t.
                        its true class y_i is cos_single[k, i, y_i].
  cos_agg[N, C]       : GPA-aggregated (znorm + sqrt(n_k) weighted + post-L2)
                        feature cos to the ETF prototypes.
  labels[N]
  counts[K, C]        : n_{k,c}  (from the Dirichlet partition ccc)
  seen_mask[K, C]     : counts > 0   (which classes each client has observed)
  w[K]                : sqrt(sum_c n_{k,c}) GPA aggregation weight per client
  m_reliable[N]       : # clients that have seen the true class of sample i
  sig_cos[N]          : cos( L2( sum_{k: seen y_i} w_k z(h_k) ), e_{y_i} )
                        -> reliable clients accumulate toward the true vertex
  noise_cos[N]        : cos( L2( sum_{k: unseen y_i} w_k z(h_k) ), e_{y_i} )
                        -> unreliable clients' directional residual on true dir (~0)
  full_cos[N]         : cos( L2( all clients ), e_{y_i} )  (~= cos_agg[i, y_i]; sanity)
  pred_agg[N]         : argmax_c cos_agg  (GPA prediction)

z-normalization statistics are computed over the full test set, matching
eval_ablation_RIJ.znorm_sqrt_aggregate exactly.
"""
import os, argparse
import numpy as np, torch, torch.nn.functional as F
from rebuild8 import prepare_data, generate_etf
from eval_ablation_RIJ import load_backbones, forward_features, znorm_sqrt_aggregate


def l2_np(a, ax=-1):
    return a / (np.linalg.norm(a, axis=ax, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--K', type=int, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--NL', type=int, default=10, help='num classes (10 / 100)')
    ap.add_argument('--FD', type=int, default=256, help='feature dim')
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    NC, NL, FD = args.K, args.NL, args.FD
    etf = generate_etf(NL, FD)
    En = l2_np(etf.numpy().astype(np.float32))                 # [C, FD]

    _, _, tl, ccc = prepare_data(NC, args.alpha, NL)
    bbs = load_backbones(args.save_dir, NC, FD)
    all_raw, labels, _ = forward_features(bbs, [{} for _ in range(NC)], tl, etf,
                                          NC, FD, NL, use_experts=False)
    N = len(labels)
    valid = [k for k in range(NC) if all_raw[k] is not None]
    if len(valid) < NC:
        print(f"[warn] {NC - len(valid)} clients have no backbone: {set(range(NC)) - set(valid)}",
              flush=True)

    # --- coverage metadata ---
    counts = np.array([[ccc.get(k, {}).get(c, 0) for c in range(NL)] for k in range(NC)],
                      dtype=np.int32)                          # [K, C]
    seen_mask = counts > 0                                     # [K, C] bool
    w = np.sqrt(counts.sum(1).clip(min=0)).astype(np.float32)  # [K]  sqrt(n_k), GPA weight
    for k in range(NC):                                        # inert any missing backbone
        if all_raw[k] is None:
            w[k] = 0.0; seen_mask[k, :] = False

    # --- per-client raw-fc cosine to ETF (reliability signal) ---
    raw = np.stack([(all_raw[k].numpy().astype(np.float32) if all_raw[k] is not None
                     else np.zeros((N, FD), np.float32)) for k in range(NC)])  # [K, N, FD]
    cos_single = np.stack([l2_np(raw[k]) @ En.T for k in range(NC)]).astype(np.float16)  # [K, N, C]

    # --- GPA aggregate (exact same pipeline as the method) ---
    feat_agg = znorm_sqrt_aggregate(all_raw, ccc, NC, N, FD).numpy().astype(np.float32)
    cos_agg = (l2_np(feat_agg) @ En.T).astype(np.float32)      # [N, C]
    pred_agg = cos_agg.argmax(1).astype(np.int16)

    # --- signal / noise decomposition per sample ---
    # z-norm each client over the full test set (matches znorm_sqrt_aggregate)
    fz = np.zeros((NC, N, FD), dtype=np.float32)
    for k in range(NC):
        f = raw[k]
        fz[k] = (f - f.mean(0, keepdims=True)) / (f.std(0, keepdims=True) + 1e-8)
    e_true = En[labels]                                        # [N, FD]  true-class ETF vertex
    rel = seen_mask[:, labels].T.astype(np.float32)            # [N, K]  1 if client k saw y_i
    wk = w[None, :]                                            # [1, K]
    # weighted partial sums:  [N, FD]
    vR = np.einsum('nk,kn...->n...', rel * wk, fz)             # reliable clients
    vU = np.einsum('nk,kn...->n...', (1.0 - rel) * wk, fz)     # unreliable clients
    vAll = vR + vU
    sig_cos = (l2_np(vR) * e_true).sum(1).astype(np.float32)
    noise_cos = (l2_np(vU) * e_true).sum(1).astype(np.float32)
    full_cos = (l2_np(vAll) * e_true).sum(1).astype(np.float32)
    m_reliable = seen_mask[:, labels].sum(0).astype(np.int16)  # [N]  |R_i|

    acc_agg = float((pred_agg == labels.astype(np.int16)).mean())

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez_compressed(
        args.out,
        cos_single=cos_single, cos_agg=cos_agg.astype(np.float16),
        labels=labels.astype(np.int16),
        counts=counts, seen_mask=seen_mask, w=w,
        m_reliable=m_reliable, pred_agg=pred_agg,
        sig_cos=sig_cos.astype(np.float16),
        noise_cos=noise_cos.astype(np.float16),
        full_cos=full_cos.astype(np.float16),
        alpha=args.alpha, K=NC, NL=NL, acc_agg=acc_agg,
    )
    cov = seen_mask.sum(1).mean()
    print(f"saved {args.out}  N={N} cos_single{cos_single.shape} "
          f"acc_agg={acc_agg:.4f} mean_coverage={cov:.1f}/{NL} "
          f"seen_sig={sig_cos.mean():.3f} unseen_noise={np.abs(noise_cos).mean():.3f}",
          flush=True)


if __name__ == '__main__':
    main()
