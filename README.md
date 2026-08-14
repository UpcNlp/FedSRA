# FedSRA

One-shot (single-round) federated learning. Each client shares a frozen simplex
Equiangular Tight Frame (ETF) classifier and trains only its backbone (ERL); the
server fuses the clients with Reliability-Guided Aggregation (RGA): per-client
z-score standardization, sqrt(n)-weighted sum, post-L2 normalization, then
nearest-ETF classification.

## Layout

```
src/            method + CIFAR/Tiny-ImageNet experiments (ETF, RGA, grouped-merge,
                scalability, R/I/J ablations, NC / residual diagnostics)
experiments/    medical + natural-federation experiments (MedMNIST, Fed-ISIC2019)
figures/        figure scripts and PDFs
docs/           DATA.md, BASELINE_PROVENANCE.md, EXPERIMENTS.md
```

## Setup

```bash
conda create -n fedsra python=3.10 && conda activate fedsra
pip install -r requirements.txt
```

DCU/ROCm: `source /opt/dtk/env.sh` first; pick a device with `HIP_VISIBLE_DEVICES`.
On CUDA use `CUDA_VISIBLE_DEVICES`.

## Data

See `docs/DATA.md`. CIFAR-10/100, Tiny-ImageNet, and MedMNIST auto-download.
Fed-ISIC2019 is obtained through the FLamby benchmark (we do not redistribute the
clinical data); place its two parquet files under `data/Fed-ISIC2019/`.

## Reproduce

Runs are deterministic given the seed; metrics print at the end. Three-seed
mean/std uses seeds 0, 42, 123.

### CIFAR / Tiny-ImageNet (main table, scalability, ablations) — `src/`

```bash
python src/rebuild8_cifar100.py --alpha 0.05 --backbone resnet18 --seed 42 --n_clients 10
python src/eval_ablation_RIJ.py    ...   # R/I/J loss ablation
python src/measure_fednc.py --dataset cifar100 --alpha 0.05 --K 10 --variant J --save_dir <ckpt> --out <json>
python src/eval_residual_diag.py --dataset cifar100 --alpha 0.05 --K 10 --save_dir <ckpt>
python src/eval_grouped_merge_cost.py ...  # serving-cost / accuracy Pareto
```

### Fed-ISIC2019 natural federation — `experiments/`

```bash
TR=data/Fed-ISIC2019/train-00000-of-00001.parquet
TE=data/Fed-ISIC2019/test-00000-of-00001.parquet

# FedSRA (RGA) + real CE (O-FedAvg + ensembles), from scratch, 3 seeds, 200 epochs
python experiments/fedisic_fedsra.py --method both --seed 42 --epochs 200 \
    --snapshot_every 40 --image_size 144 --train_parquet $TR --test_parquet $TE --output out/fedisic
# FAFI baseline (adapter over the official FAFI objective; see docs/BASELINE_PROVENANCE.md)
python experiments/fedisic_fafi.py --seed 42 --epochs 200 \
    --image_size 144 --train_parquet $TR --test_parquet $TE --output out/fedisic
# accuracy-vs-epoch convergence curve (from saved milestones)
python experiments/fedisic_eval_curve.py --method fedsra --seed 42 \
    --epochs 40,80,120,160,200 --final_epoch 200 --image_size 144 \
    --train_parquet $TR --test_parquet $TE --output out/fedisic
```

### MedMNIST margin-vs-skew law, label noise — `experiments/`

```bash
# sweep alpha in {0.05,0.1,0.3,0.5}, method in {fedsra,ce,fafi}, seeds {42,0,123}
python experiments/medmnist_realbaseline.py --method fedsra --dataset pathmnist \
    --data data/medmnist/pathmnist.npz --alpha 0.05 --seed 42 --epochs 100 --output out/medmnist
# label noise: add --noise_rate 0.2 or 0.4
```

### RGA mechanism / incomplete-NC diagnostics — `experiments/`

```bash
python experiments/residdiag_fedisic.py  --ckpt_dir out/fedisic/checkpoints/fedsra_s42 \
    --train_parquet $TR --test_parquet $TE --out out/residdiag_fedisic_s42.json
python experiments/residdiag_medmnist.py --dataset pathmnist --data data/medmnist/pathmnist.npz \
    --alpha 0.05 --seed 42 --out out/residdiag_pathmnist_s42.json
python experiments/ncsweep_medmnist.py   --dataset pathmnist --data data/medmnist/pathmnist.npz \
    --alpha 0.05 --K 20 --seed 42 --out out/ncsweep_pathmnist_k20.json
```

## Baselines

Baselines live in `baselines/` as git submodules pinned to the exact upstream
commit we ran (we reference the official code, we do not redistribute it). Fetch
them with:

```bash
git submodule update --init --recursive
```

Our adapters (in `experiments/`) change only the backbone and data handling for a
fair comparison; every method-specific objective and server procedure is
unchanged. See `baselines/README.md` and `docs/BASELINE_PROVENANCE.md`.
