# Extended experiments and analysis

Supplementary results for FedSRA beyond the main tables: the mechanism behind
Reliability-Guided Aggregation (RGA), how the method scales with the number of
clients, medical and natural federations, single-sample inference, serving cost,
and comparisons against multi-round training. Runs are from scratch and use
seeds 0, 42, 123 unless noted. Scripts for every result below live in `src/` and
`experiments/`.

## 1. Why RGA works: do unseen-class residuals cancel?

RGA sums z-scored client logits with sqrt(n) weights. A client that never saw a
class can still emit a biased logit on it, so we measure, per class, the residual
of unseen clients and its cross-client correlation on three data families
(CIFAR-10/100 synthetic skew, MedMNIST, and the natural Fed-ISIC federation).
Script: `src/eval_residual_diag.py`, `experiments/residdiag_*.py`.

- Individual unseen clients are biased (residual-to-RMS 0.32–0.64, clearly
  non-zero), but after removing each client's mean bias the remaining part is
  only weakly correlated across clients (centred rho = 0.04–0.11).
- The sqrt(n)-weighted sum cancels this random part while the true signal mu
  stays positive and tracks accuracy. Correlation is smallest exactly in the
  severe-skew, low seen-fraction regime where cancellation matters most (CIFAR
  rho rises 0.05 -> 0.21 as the seen fraction rises 0.3 -> 0.9).
- Un-standardized aggregation instead attenuates the signal to (w_S / W)·mu with
  w_S / W = 0.24–0.40 at alpha = 0.05, matching its 4.6 -> 36.6 point gap to RGA.

Cancellation needs only weak cross-client correlation, strictly weaker than
per-client zero mean. See `figures/residdiag_rho.png`.

## 2. Scaling with client count and the incomplete-collapse threshold

As clients are added, each sees fewer classes and per-client neural collapse
becomes incomplete. We aggregate the per-client features (GPA) and track the
aggregated NC1 ratio. Script: `src/measure_fednc.py`, `experiments/ncsweep_medmnist.py`.

On CIFAR-100 (100 classes) aggregated NC1 crosses 1.0 at K ~ 20 for every alpha,
coinciding with the accuracy elbow (the sharpest drop is always K = 20 -> 50,
8.5–10.2 points). The threshold is set by the class count, not universal. On
PathMNIST (9 classes) aggregated NC1 never crosses 1.0 through K = 50:

| K  | aggregated NC1 | NC2   | accuracy |
|----|----------------|-------|----------|
| 5  | 0.464          | 0.774 | 78.62%   |
| 10 | 0.385          | 0.710 | 77.17%   |
| 20 | 0.277          | 0.776 | 81.77%   |
| 50 | 0.300          | 0.786 | 84.43%   |

See `figures/nc_threshold.png`.

## 3. Medical and natural federations

Against independently trained baselines (a real shared-initialization one-shot
FedAvg, CE ensembles, and FAFI), not aggregation variants of a shared backbone.
Scripts: `experiments/medmnist_realbaseline.py`, `experiments/fedisic_fedsra.py`,
`experiments/fedisic_fafi.py`.

**MedMNIST, Dirichlet skew.** On PathMNIST / OrganAMNIST we sweep
alpha in {0.05, 0.1, 0.3, 0.5}. FedSRA is best in every incomplete-coverage cell,
and its Top-1 margin over the strongest baseline FAFI shrinks monotonically as
skew relaxes:

| alpha | 0.05  | 0.1   | 0.3  | 0.5  |
|-------|-------|-------|------|------|
| PathMNIST margin vs FAFI   | +15.6 | +14.7 | +1.3 | -0.9 |
| OrganAMNIST margin vs FAFI | +7.1  | +1.4  | +0.8 | -0.3 |

At alpha = 0.05 the absolute Top-1 is 82.2 / 84.5% (FedSRA) versus 66.5 / 77.4%
(FAFI), 54.6 / 64.2% (CE ensemble), and 12.7 / 16.4% (one-shot FedAvg). Under
20% / 40% label noise FedSRA stays best or tied (PathMNIST 82.2 -> 73.8 -> 71.1%),
and under MedMNIST-C corruption it holds 59.9 / 66.1%. See `figures/medmnist.png`.

The margin tracking skew is the method's operating characteristic: the ETF frame
helps most when class coverage is incomplete, and at mild skew (alpha = 0.5) the
advantage naturally approaches zero.

**Fed-ISIC2019, a natural federation** (six real FLamby acquisition centers,
8 classes, a non-synthetic partition with measured per-center label entropy
0.14–0.80), trained to convergence:

| Method        | Balanced accuracy | Worst-class accuracy |
|---------------|-------------------|----------------------|
| FedSRA        | **61.9**          | **50.0**             |
| FAFI          | 48.0              | <= 23.4              |
| CE ensemble   | 37.0              | <= 23.4              |
| one-shot FedAvg | 23.1            | <= 23.4              |

Baselines collapse toward the majority class (FAFI / CE lead only on macro-AUC, a
ranking metric a majority predictor scores well on). Per-epoch curves show a fair,
converged comparison: baselines saturate by epoch 20 while FedSRA converges near
epoch 160–200, so training all methods to 200 epochs widens FedSRA's lead over
FAFI from +11.9 to +13.9, and one-shot FedAvg degrades with more training. See
[`figures/convergence.pdf`](../figures/convergence.pdf).

## 4. Feature standardization and single-sample inference

The Eq. (4) statistics mu_k, sigma_k are per-feature over samples (eps = 1e-8).
Script: `src/eval_batch_zscore.py`.

- A frozen-calibration variant (statistics estimated once from 1,024 unlabeled
  samples, applied independently per sample) matches the transductive result
  within 0.5 point in all 12 settings (CIFAR-10, K = 10, alpha = 0.05: 78.81%
  vs 78.93%), and stays 78.81% at batch size 1 and under class-sorted streams,
  where naive per-batch statistics collapse to 10.1%.
- A fully calibration-free variant, in which each client uploads only its
  training-feature mean/std (two 256-D vectors) with its model and no extra
  round, reaches the 57.3% deployable row, above every baseline.

## 5. Serving cost and grouped merging

We report the full inference cost. Scripts: `src/measure_efficiency.py`,
`src/eval_grouped_merge_cost.py`.

At K = 50, FedSRA is 5.20 ms/image at batch 256 but 145 ms/image at batch 1
(2,525 MB), about an order of magnitude above a single model (0.07 / 2.8 ms,
366 MB), since inference scales with the retained backbones. Grouped merging is a
complete data-free algorithm (balanced grouping, cosine-threshold filter merging
within a group, then RGA across groups). Its cost/accuracy trade-off on CIFAR-100
(alpha = 0.05, K = 10, tau = 0.5):

| Groups G | Accuracy | Parameters | Latency @ B=1 | Memory  | Preprocessing |
|----------|----------|------------|---------------|---------|---------------|
| 10       | 63.7%    | 113 M      | 27.8 ms       | 800 MB  | 0 s           |
| 5        | 40.4%    | 169 M      | 13.6 ms       | 1243 MB | 323 s         |
| 2        | 32.9%    | 418 M      | 18.1 ms       | 2877 MB | 519 s         |

It cuts the forward count, but its union form widens models and can lose accuracy,
so it is an explicit trade-off, not a free reduction.

## 6. Reproducibility, ordering, and multi-round comparison

**Seeds.** All 12 CIFAR-100 alpha x K cells are repeated over three seeds
(std 0.12–0.83 point, mean 0.41); Fed-ISIC and MedMNIST also use three seeds. A
released IntactOFL run stays below FAFI and FedSRA in every alpha cell
(40.7 / 61.3% at alpha = 0.05 / 0.5).

**Ordering vs components.** Matched pre-L2 / logit-ensemble ablations (same
backbone and ETF frame) lose 5–20 points at alpha <= 0.1, so the
standardize -> weight -> aggregate -> L2 ordering, not the components alone,
drives the gain. Script: `src/eval_ablation_RIJ.py`.

**Multi-round.** A matched fixed-ETF FedAvg control (rounds 1/3/5/10) is near
random at one round and, after 10 rounds, still 13.3–41.4 points below one-round
FedSRA across eight settings (CIFAR-100 alpha = 0.05:
1.2 -> 6.1 -> 23.1 -> 41.7% vs 66.2%). Script: `src/train_multiround_etf.py`,
see `figures/multiround.png`.
