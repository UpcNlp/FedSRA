"""
dump_fafi_features.py — extract FAFI (LearnableProtoResNet) features and
package them in the same npz layout the fednc_overview figure expects,
so we can plug them in as a baseline column.

For each FAFI client (0..K-1) we load the final-epoch checkpoint, forward
the CIFAR-10 test set, get the encoder output (un-normalized), datasize-
weighted sum across clients, then L2-normalize at the end.  This matches
FAFI's training-time `eval_with_proto(WEnsembleFeature, ...)` pipeline.

CRITICAL gotcha: FAFI's `WEnsembleFeature` stores submodels in a plain
Python list (not `nn.ModuleList`), so `.eval()` does NOT propagate. The
training-time eval that yields the 55-60% reported acc actually runs with
**BatchNorm in train mode** (batch stats from the test batches). We
reproduce this regime here — do NOT call `model.eval()` on the loaded
clients. (Confirmed 2026-06-02: BN=eval+uniform reproduces 22.55% at
ep295, BN=train+datasize reproduces yaml's 59.52% exactly.)

Output:  results/fednc_overview_fafi_ens_a{α}_k{K}_s{seed}.npz

Notes:
  - FAFI's encoder is a vanilla ResNet18 (dim_in=512) so this npz's
    'feats_256' field is actually 512-D. We keep the field name for
    drop-in compatibility with the figure scripts (they don't care about
    width).
  - FAFI uses its own per-client learnable_proto and a fixed orthogonal
    proto_classifier (constructed inside the model).  We do NOT save a
    "fixed ETF" — FAFI has no shared anchor, so the figure for this
    column will skip the prototype-square overlay.
  - The NC matrix is computed on the post-ensemble class means, which
    is dimension-agnostic.
"""
import argparse, os, sys
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader

# Make FAFI's models_lib + dataset_helper importable. We reuse FAFI's OWN
# test loader (ToTensor-only, no Normalize) — its training-time eval uses
# this exact pipeline.
FAFI_ROOT = os.path.expanduser(
    '/public/home/dongshou/fedETF/FAFI_ICML25-master-orgin')
if not os.path.isdir(FAFI_ROOT):
    FAFI_ROOT = os.path.expanduser(
        '~/projects/fedETF/FAFI_ICML25-master-orgin')
sys.path.insert(0, FAFI_ROOT)

from models_lib import get_train_models
from common_libs import load_yaml_config, setup_seed
from dataset_helper import get_fl_dataset


def l2(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default=None,
                    help='path to FAFI yaml; if set, alpha/K/seed are read from it')
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epoch', type=int, default=695,
                    help='which FAFI checkpoint epoch to load per client')
    ap.add_argument('--model_name', default='resnet18')
    ap.add_argument('--ckpt_root',
                    default=f'{FAFI_ROOT}/checkpoints/CIFAR10_alpha{{α}}_K{{K}}/local_models')
    ap.add_argument('--n_sub', type=int, default=10000)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    # Load FAFI's training-time config to reproduce its exact data partition
    # (datasize weights depend on the per-client Dirichlet draw).
    if args.cfg is None:
        cfg_path = f'{FAFI_ROOT}/configs/CIFAR10_alpha{args.alpha}_K{args.K}.yaml'
    else:
        cfg_path = args.cfg
    cfg = load_yaml_config(cfg_path)
    setup_seed(cfg['seed'])    # MUST match training seed → same partition

    NL = cfg['dataset']['num_classes']; FD = 512   # FAFI ResNet18 encoder dim
    K = cfg['client']['num_clients']
    alpha = cfg['distribution']['alpha']
    ckpt_root = args.ckpt_root.format(**{'α': alpha, 'K': K})
    print(f"[fafi] cfg={cfg_path}  seed={cfg['seed']}  K={K}  α={alpha}", flush=True)
    print(f"[fafi] checkpoint root: {ckpt_root}", flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # FAFI's own test loader — ToTensor only, no Normalize. Required to
    # match the training-time eval pipeline.
    os.chdir(FAFI_ROOT)
    trainset, testset, client_idx_map = get_fl_dataset(
        cfg['dataset']['data_name'], cfg['dataset']['root_path'], K,
        cfg['distribution']['type'], cfg['distribution']['label_num_per_client'],
        alpha)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False,
                             num_workers=0, pin_memory=True)

    # datasize weights — exactly what FAFI uses when aggregated_by_datasize=True
    sizes = [len(client_idx_map[c]) for c in range(K)]
    use_ds = cfg['server'].get('aggregated_by_datasize', True)
    if use_ds:
        weights = [s / sum(sizes) for s in sizes]
    else:
        weights = [1.0 / K for _ in range(K)]
    print(f"[fafi] sizes  = {sizes}", flush=True)
    print(f"[fafi] weights= {[round(w,4) for w in weights]} "
          f"(aggregated_by_datasize={use_ds})", flush=True)

    # Capture labels in CIFAR-10 test order
    all_labels = []
    for _, y in test_loader:
        all_labels.append(y.numpy())
    all_labels = np.concatenate(all_labels)
    N = len(all_labels)
    print(f"[fafi] {N} test samples, NL={NL}", flush=True)

    # Forward each client's ENCODER (un-normalized), datasize-weighted sum.
    # CRITICAL: do NOT call model.eval() — see module docstring. BN stays in
    # train mode (uses batch stats) to match FAFI's training-time eval.
    feat_sum = np.zeros((N, FD), np.float32)
    proto_sum = np.zeros((NL, FD), np.float32)   # for the global learnable_proto avg
    for k in range(K):
        ckpt_path = os.path.join(ckpt_root, f'client_{k}', f'epoch_{args.epoch}.pth')
        if not os.path.isfile(ckpt_path):
            print(f"[warn] missing: {ckpt_path}", flush=True); continue
        model = get_train_models(args.model_name, num_classes=NL, mode='our')
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        elif not isinstance(sd, dict):
            sd = sd.state_dict()
        model.load_state_dict(sd, strict=True)
        model.to(device)
        # NO .eval() — BN must stay in train mode (matches training-time eval).
        # `with torch.no_grad()` still disables autograd; BN train mode uses
        # batch stats either way.

        feats = []
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device, non_blocking=True)
                feat = model.encoder(x)        # (B, 512), un-normalized
                feats.append(feat.cpu().numpy())
        feats = np.concatenate(feats, axis=0)  # (N, 512)
        feat_sum += weights[k] * feats         # datasize-weighted sum
        proto_sum += model.learnable_proto.detach().cpu().numpy() / K  # mean over K
        model.cpu()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"[fafi] client {k} encoder forwarded  (w={weights[k]:.4f})", flush=True)

    feat_ens = feat_sum                        # weighted sum already
    feat_ens_n = l2(feat_ens)                  # final L2 norm (eval_with_proto does this)

    # Optional sanity check: reproduce the yaml acc using the global proto
    global_proto = proto_sum                   # (NL, 512), mean over K of learnable_proto
    logits = feat_ens_n @ global_proto.T
    pred = logits.argmax(1)
    repro_acc = float((pred == all_labels).mean())
    print(f"[sanity] reproduced acc using avg(learnable_proto) = {repro_acc:.4f}", flush=True)

    # Class means + NC matrix (on the L2-normed ensemble feature, matching
    # what FAFI's classifier head actually consumes via eval_with_proto)
    mus = np.zeros((NL, FD), np.float32)
    for c in range(NL):
        mask = (all_labels == c)
        if mask.any():
            mus[c] = feat_ens_n[mask].mean(0)
    mus_n = l2(mus)
    nc_mat = mus_n @ mus_n.T

    # Stratified sub-sample for the scatter plot (1000 per class default)
    rng = np.random.default_rng(cfg['seed'])
    per_cls = args.n_sub // NL
    keep = []
    for c in range(NL):
        idxs = np.where(all_labels == c)[0]
        n = min(per_cls, len(idxs))
        keep.append(rng.choice(idxs, size=n, replace=False))
    keep = np.concatenate(keep)
    feats_sub = feat_ens_n[keep]
    labels_sub = all_labels[keep]

    # Save in the SAME schema as the other 4 settings (figure script-ready)
    # NOTE: feats_2d / class_means_2d are LEGACY 2D projections; we set
    # them to zeros because the figure script re-projects via t-SNE.
    # We do NOT include an ETF prototype matrix — FAFI has no shared anchor.
    out = args.out
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    np.savez_compressed(
        out,
        feats_2d=np.zeros((len(keep), 2), np.float32),     # placeholder
        feats_256=feats_sub.astype(np.float32),            # ACTUALLY 512-D
        labels=labels_sub.astype(np.int32),
        class_means_256=mus.astype(np.float32),            # 512-D
        class_means_2d=np.zeros((NL, 2), np.float32),      # placeholder
        nc_matrix=nc_mat.astype(np.float32),
        etf_2d=np.zeros((NL, 2), np.float32),              # FAFI has no fixed ETF
        projection_W=np.zeros((2, FD), np.float32),        # placeholder
        etf_256=np.zeros((NL, FD), np.float32),            # placeholder
        setting='fafi_ens',
        alpha=alpha, K=K, seed=cfg['seed'],
        feat_dim=FD,
        no_etf=True,                                       # flag for figure
        repro_acc=repro_acc,                               # sanity check
    )
    print(f"[save] {out}  (feats_256 shape {feats_sub.shape})", flush=True)
    print(f"   NC matrix diag mean = {np.diag(nc_mat).mean():.3f}, "
          f"off-diag mean = {nc_mat[~np.eye(NL, dtype=bool)].mean():.3f} "
          f"(ideal off-diag = {-1/(NL-1):.3f})", flush=True)


if __name__ == '__main__':
    main()
