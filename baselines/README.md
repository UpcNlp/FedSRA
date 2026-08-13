# Baselines

Each competing method is referenced as a **git submodule** pinned to the exact
upstream commit we ran, so a reviewer gets the official code (not a copy we
redistribute) plus our thin adapter against it. We never edit a submodule's
source: the adapter imports the upstream objective and only swaps the backbone,
the data pipeline, and the reporting so every method shares one protocol.

Fetch the upstream code after cloning:

```bash
git submodule update --init --recursive
```

| Directory | Method | Upstream | Pinned commit |
|---|---|---|---|
| `FAFI/`        | FAFI (ICML 2025), our strongest baseline | https://github.com/zenghui9977/FAFI_ICML25 | `26cf90e` |
| `Co-Boosting/` | Co-Boosting (ICLR 2024)                  | https://github.com/rong-dai/Co-Boosting    | `b95d715` |
| `FuseFL/`      | FuseFL (NeurIPS 2024)                    | https://github.com/wizard1203/FuseFL       | `8248f54` |
| `FedCGS/`      | FedCGS (AAAI 2025)                       | https://github.com/Yuqin-G/FedCGS          | `e02cc1f` |
| `IntactOFL/`   | IntactOFL (ACM MM 2024)                  | https://github.com/zenghui9977/IntactOFL   | `e8f86ef` |

## What we changed, per baseline

### FAFI
Adapters: `experiments/fedisic_fafi.py`, `experiments/realfed_fafi.py`, and the
FAFI branch of `experiments/medmnist_realbaseline.py`.

- **Imported unchanged** from `baselines/FAFI/oneshot_algorithms/ours/unsupervised_loss.py`:
  the official prototype/contrastive losses (`Contrastive_proto_feature_loss`,
  `Contrastive_proto_loss`, `SupConLoss`), together with the learnable
  prototypes, the unweighted global prototype average, and the size-weighted
  feature ensemble. We do not reimplement the objective.
- **Backbone:** replaced the upstream encoder with the same from-scratch
  ResNet-18 and training budget used by our method, so the comparison is
  matched.
- **Data:** aligned the client partition to the shared protocol used by every
  method (the upstream run uses an overlapping split); read images from the
  FLamby / MedMNIST pipelines instead of the original loaders.

### Co-Boosting
Adapter: `experiments/realfed_coboost.py`.

- **Imported unchanged** from `baselines/Co-Boosting/`: the `COBOOSTSynthesizer`
  (`datafree/synthesis/coboost.py`), the `WEnsemble` weighting (`utils_fl.py`),
  and the official generator (`datafree/models/generator.py`). The synthesizer,
  ensemble-weight adaptation, hard-sample loss, ODS perturbation, and data-free
  KL distillation are all upstream's.
- **Teachers:** fed it our own CE client checkpoints rather than retraining, so
  the distillation baseline and our method share the same clients.
- **Compatibility only:** we load the generator module directly instead of
  `import datafree.models`, because importing that package eagerly executes
  legacy classifier modules that reference the removed
  `torchvision.models.utils` namespace and would crash on modern torchvision.
  Synthetic images are generated at a lower resolution (64, valid because
  ResNet-18 is fully convolutional) while real validation/test images stay at
  the shared evaluation resolution.

### FuseFL, FedCGS, IntactOFL
Run directly from the submodule with **config-level changes only**: dataset,
backbone family, input resolution, device, and result logging are set to the
shared protocol; no objective or server code is modified.

### Not submodules
- **DENSE** (NeurIPS 2022) and **FedDF** (NeurIPS 2020) are run through the
  distributed implementations bundled inside `Co-Boosting/` and `IntactOFL/`
  respectively (dataset/device/reporting adapters only), so they need no
  separate checkout.
- **O-FedAvg** and the **CE ensemble** are standard and implemented by us
  (`src/`, `experiments/`): shared-initialization clients with, respectively, a
  sample-size-weighted parameter average and a uniform / sqrt(n) logit average.
- **FedOV** is not compared: it requires auxiliary open-set / abstention
  supervision the other methods do not use.

See `../docs/BASELINE_PROVENANCE.md` for the full fairness protocol (shared
partition, from-scratch training, seeds 0/42/123).
