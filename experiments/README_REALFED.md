# RealFed fundus rebuttal experiments

This directory contains the exact entrypoints used for the three-source clinical
cross-silo experiment in the rebuttal. BRSET, mBRSET, and ODIR-5K are independent
acquisition sources, and each source is treated as one silo. The shared single-label
task is binary diabetes-related retinal-disease recognition: ICDR-positive DR for
BRSET/mBRSET and the released D category for ODIR-5K.

## Protocol

- Use the pre-specified approximately 60/20/20 patient-level
  train/validation/test assignments, fixed before method training, and verify that
  patient identifiers do not cross splits within a source. These assignments are
  experimental splits, not claimed to be official splits from all three releases.
- Verify that every selected image path is unique and that no image crosses splits.
  The audited counts are 10,809/10,809 unique BRSET paths, 4,884/4,884 mBRSET
  paths, and 6,392/6,392 ODIR-5K paths.
- Train at 224 x 224 with an ImageNet-initialized ResNet-18 for 30 local epochs.
- Use class-balanced local sampling for every method because BRSET has about 7%
  positives. Evaluation is always on the untouched natural test distribution.
- Report balanced accuracy, AUROC, AUPRC, macro-F1, sensitivity, specificity,
  source-specific performance, pooled performance, and the worst participating
  source. Every formal cell uses seeds 0, 42, and 123.
- The primary FedSRA deployment variant standardizes with each client's local
  training-feature mean and standard deviation. The client uploads two 256-D
  vectors (512 scalars) with its model in the same one-shot communication. Full-test statistics
  are retained only as a transductive diagnostic.

## Expected data layout

The launchers expect `/public/home/dongshou/fedETF/realfed_data/` with:

```text
BRSET/dataset_multilabel_split.csv
mBRSET/dataset_multilabel_split.csv
ODIR-5K/dataset_multilabel_split.csv
```

Each CSV contains `split` in `{train,val,test}` and a prepared
`final_icdr_binary` label. The source-specific image and patient columns are:

| Source | Image column | Patient column |
|---|---|---|
| BRSET | `view_0` | `patient_id` |
| mBRSET | `file` | `patient` |
| ODIR-5K | `file` | `ID` |

Run a read-only audit before training:

```bash
/public/home/dongshou/anaconda/envs/ct/bin/python \
  review_response/experiments/realfed_fundus.py \
  --method fedsra --data_root realfed_data --output /tmp/realfed_validate \
  --validate_only
```

## Formal launchers

All launchers are resumable. A completed JSON is skipped; an interrupted local
training resumes from the latest `*.last.pt` checkpoint.

On the 8-GPU pod:

```bash
nohup bash review_response/experiments/run_realfed_fedsra_only.sh \
  > realfed_out/logs/fedsra_master.log 2>&1 &
nohup bash review_response/experiments/run_realfed_ce_only.sh \
  > realfed_out/logs/ce_master.log 2>&1 &
nohup bash review_response/experiments/run_realfed_heldout.sh \
  fedsra 0:0 1:42 2:123 > realfed_out/logs/heldout_fedsra_master.log 2>&1 &
nohup bash review_response/experiments/run_realfed_heldout.sh \
  ce 3:0 4:42 5:123 > realfed_out/logs/heldout_ce_master.log 2>&1 &
nohup bash review_response/experiments/run_realfed_coboost.sh \
  6:0 7:123 > realfed_out/logs/coboost_cluster8_master.log 2>&1 &
```

On the shared-storage 4-GPU pod:

```bash
nohup bash review_response/experiments/run_realfed_fafi.sh \
  > realfed_out/logs/fafi_master.log 2>&1 &
nohup bash review_response/experiments/run_realfed_heldout.sh \
  fafi 0:0 1:42 2:123 > realfed_out/logs/heldout_fafi_master.log 2>&1 &
nohup bash review_response/experiments/run_realfed_coboost.sh \
  3:42 > realfed_out/logs/coboost_cluster4_master.log 2>&1 &
nohup bash review_response/experiments/run_realfed_summary_wait.sh \
  > realfed_out/logs/summary_wait.log 2>&1 &
```

The held-out launchers wait for the corresponding three-source result. The
Co-Boosting launchers wait for all three final CE teachers for their seed.

## Baselines and fairness

- `realfed_fundus.py --method ce` trains all CE clients from an identical initial
  state. It evaluates sample-size-weighted one-shot FedAvg, uniform logit ensemble,
  and square-root-size-weighted logit ensemble from the same checkpoints.
- `realfed_fafi.py` imports the official ICML 2025 four-term FAFI objective and
  retains its learned prototypes and prototype/feature aggregation.
- `realfed_coboost.py` uses the official ICLR 2024 synthesizer, adaptive teacher
  weights, hard-sample term, ODS, and KL distillation. Its teachers are exactly the
  matched CE checkpoints above, avoiding duplicate local training.
- FedSRA and FAFI use their method-specific local objectives. Every method shares
  the same source splits, ResNet-18 family, image resolution, local epoch budget,
  and seed set. See `../BASELINE_PROVENANCE.md` for the full audit.

## Outputs and verification

Outputs are written atomically under `realfed_out/`:

```text
checkpoints/realfed_binary_<method>_heldout-<none|mbrset>_s<seed>/
results/realfed_binary_<method>_heldout-<none|mbrset>_s<seed>.json
logs/
```

The formal matrix has 21 cells: 12 three-source cells (four methods x three
seeds) and 9 held-out-mBRSET cells (FedSRA/CE/FAFI x three seeds). Validate every
field and then aggregate mean +/- sample standard deviation:

```bash
/public/home/dongshou/anaconda/envs/ct/bin/python \
  review_response/experiments/validate_realfed_results.py \
  --results realfed_out/results
/public/home/dongshou/anaconda/envs/ct/bin/python \
  review_response/experiments/summarize_realfed.py \
  --results realfed_out/results --output review_response/realfed_summary.csv
```
