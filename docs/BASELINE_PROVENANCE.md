# Baseline provenance

We do not redistribute baseline code: none of the baseline repositories we ran ship a
license that permits redistribution. For each baseline we therefore link to its
official repository at a pinned commit and ship only (i) our thin adapter and (ii) a
`setup.sh` that clones the upstream code. This lets a reader verify the exact
official code we ran against our minimal diff.

Every adapter changes only the backbone and data handling so that all methods share
one experimental protocol; each method-specific objective and server procedure is
left unchanged.

| Method | Official source (pin a commit) | Venue | Our adapter | Status |
|---|---|---|---|---|
| O-FedAvg | ours (standard) | — | shared-init clients, sample-size-weighted parameter average | own code, included |
| CE ensemble | ours (standard) | — | same CE clients, uniform / sqrt(n) logit average | own code, included |
| FAFI | https://github.com/zenghui9977/FAFI_ICML25 | ICML 2025 | import the four loss terms + learned prototypes + prototype/feature aggregation; replace encoder with the shared ResNet-18; align the client data partition to the shared protocol | strongest baseline, included |
| Co-Boosting | https://github.com/rong-dai/Co-Boosting | ICLR 2024 | keep synthesizer, adaptive weights, hard-sample term, ODS, KL distillation; shared teachers | official, included |
| DENSE | via the Co-Boosting eval repo | NeurIPS 2022 | dataset/device/reporting adapters | distributed impl, included |
| FedDF | via the IntactOFL eval repo | NeurIPS 2020 | dataset/device/reporting adapters | distributed impl, included |
| FuseFL | https://github.com/wizard1203/FuseFL | NeurIPS 2024 | dataset/config/reporting adapters | official, included |
| FedCGS | https://github.com/Yuqin-G/FedCGS | AAAI 2025 | dataset/config/reporting adapters | official, included |
| IntactOFL | https://github.com/zenghui9977/IntactOFL | ACM MM 2024 | dataset/config/reporting adapters | official, included |
| FedOV | — | — | — | not compared: requires auxiliary open-set / abstention supervision the other methods do not use |

## Fairness controls

- All methods in a comparison use the same data partition, backbone family, input
  resolution, local-epoch budget, and the seeds 0/42/123.
- CIFAR/Tiny, MedMNIST, and Fed-ISIC are trained from scratch (no ImageNet
  initialization), matching the paper protocol.
- O-FedAvg, CE ensemble, and the distillation baselines share the same CE client
  checkpoints, isolating the server aggregation rather than retraining teachers.
- FAFI and FedSRA use their own local objectives from the same backbone and budget.

## Notes on adaptation

- Backbone/data adapters are the only changes; method-specific objectives and server
  procedures are preserved.
- FAFI's client data partition is aligned to the shared partition used by all methods
  (its original run uses an overlapping split).
