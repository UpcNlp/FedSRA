# MedMNIST supplementary experiments

The bundle is self-contained and is designed to be copied to a fresh directory on each
RTX 5090 host. It reads the official MedMNIST `.npz` files directly and does not modify
the source datasets.

Smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python medmnist_fedsra.py \
  --dataset pathmnist --data /path/to/pathmnist.npz --output /tmp/fedsra_smoke \
  --alpha 0.05 --noise_rate 0 --seed 42 --epochs 1 \
  --limit_train 512 --limit_test 256 --workers 0
```

Data and partition validation without training:

```bash
python medmnist_fedsra.py --dataset pathmnist --data /path/to/pathmnist.npz \
  --output /tmp/unused --alpha 0.05 --n_clients 5 --seed 42 --validate_only
```

Pilot gate (run this first):

```bash
nohup env PILOT_ONLY=1 bash wait_for_idle_and_run.sh \
  bash run_medmnist_matrix.sh pathmnist /path/to/pathmnist.npz /path/to/output /path/to/python \
  > launcher_pathmnist_pilot.log 2>&1 &
echo $!
```

After inspecting the pilot, launch the full resumable matrix without `PILOT_ONLY=1`:

```bash
nohup bash wait_for_idle_and_run.sh \
  bash run_medmnist_matrix.sh pathmnist /path/to/pathmnist.npz /path/to/output /path/to/python \
  > launcher_pathmnist.log 2>&1 &
echo $!
```

Each completed cell has a result JSON, meta JSON, per-client checkpoint directory, and
cell log. Re-running the master skips completed result files and resumes incomplete
clients from their latest epoch checkpoint.
