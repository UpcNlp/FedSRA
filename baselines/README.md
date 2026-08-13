# Baselines

Each competing method is referenced as a **git submodule** pinned to the exact
upstream commit we ran, so a reviewer gets the official code (not a copy we
redistribute) and our adapters against it. We change only the backbone and data
handling; every method-specific objective and server procedure is upstream's.

Fetch them after cloning:

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

Not submodules:

- **DENSE** (NeurIPS 2022) and **FedDF** (NeurIPS 2020) are run through the
  distributed implementations inside `Co-Boosting/` and `IntactOFL/`
  respectively, so they need no separate checkout.
- **O-FedAvg** and the **CE ensemble** are standard and implemented by us
  (`src/`, `experiments/`).
- **FedOV** is not compared: it requires auxiliary open-set / abstention
  supervision the other methods do not use.

Our adapters (thin wrappers importing each upstream objective) live in
`experiments/` (e.g. `fedisic_fafi.py`, `realfed_coboost.py`). See
`../docs/BASELINE_PROVENANCE.md` for the full fairness protocol and per-method
adaptation notes.
