"""
dump_fafi_features.py — extract FAFI (LearnableProtoResNet) features and
package them in the same npz layout the fednc_overview figure expects,
so we can plug them in as a baseline column.

For each FAFI client (0..K-1) we load the final-epoch checkpoint, forward
the CIFAR-10 test set, get the L2-normalized feature_norm (the same tensor
FAFI's IFFI fuses over).  The 10 clients' features are arithmetically
averaged per sample, then L2-renormalized — the cleanest feature-level
analogue of an ensemble.

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
from torchvision import datasets, transforms

# Make FAFI's models_lib importable
FAFI_ROOT = os.path.expanduser(
    '/public/home/dongshou/fedETF/FAFI_ICML25-master-orgin')
if not os.path.isdir(FAFI_ROOT):
    FAFI_ROOT = os.path.expanduser(
        '~/projects/fedETF/FAFI_ICML25-master-orgin')
sys.path.insert(0, FAFI_ROOT)

from models_lib import get_train_models


def l2(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-9)


def make_test_loader(bs=256, num_workers=2):
    te = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=te)
    return DataLoader(ds, batch_size=bs, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def main():
    ap = argparse.ArgumentParser()
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

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NL = 10; FD = 512   # FAFI ResNet18 encoder output dim
    K = args.K

    ckpt_root = args.ckpt_root.format(**{'α': args.alpha, 'K': K})
    print(f"[fafi] checkpoint root: {ckpt_root}", flush=True)

    test_loader = make_test_loader()
    # forward once to capture labels in CIFAR-10 test order
    all_labels = []
    for _, y in test_loader:
        all_labels.append(y.numpy())
    all_labels = np.concatenate(all_labels)
    N = len(all_labels)
    print(f"[fafi] {N} test samples, NL={NL}", flush=True)

    # Forward each client backbone, accumulate feature_norm sum.
    feat_sum = np.zeros((N, FD), np.float32)
    n_loaded = 0
    for k in range(K):
        ckpt_path = os.path.join(ckpt_root, f'client_{k}', f'epoch_{args.epoch}.pth')
        if not os.path.isfile(ckpt_path):
            print(f"[warn] missing: {ckpt_path}", flush=True); continue
        # Build model and load state_dict
        model = get_train_models(args.model_name, num_classes=NL, mode='our')
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        # FAFI sometimes wraps in a dict; if so unwrap.
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        elif not isinstance(sd, dict):
            sd = sd.state_dict()
        model.load_state_dict(sd, strict=True)
        model.to(device); model.eval()

        feats = []
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device, non_blocking=True)
                _, feat_norm = model(x)        # (B, 512), already L2-normed
                feats.append(feat_norm.cpu().numpy())
        feats = np.concatenate(feats, axis=0)  # (N, 512)
        feat_sum += feats
        n_loaded += 1
        model.cpu()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"[fafi] client {k} done", flush=True)

    if n_loaded == 0:
        raise RuntimeError("no FAFI clients loaded")
    feat_ens = feat_sum / n_loaded             # arithmetic mean across clients
    feat_ens_n = l2(feat_ens)                  # final L2 renormalize

    # Class means + NC matrix (in 512-D, then a 10x10 cosine matrix)
    mus = np.zeros((NL, FD), np.float32)
    for c in range(NL):
        mask = (all_labels == c)
        if mask.any():
            mus[c] = feat_ens[mask].mean(0)
    mus_n = l2(mus)
    nc_mat = mus_n @ mus_n.T

    # Stratified sub-sample for the scatter plot (1000 per class default)
    rng = np.random.default_rng(args.seed)
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
        alpha=args.alpha, K=K, seed=args.seed,
        feat_dim=FD,
        no_etf=True,                                       # flag for figure
    )
    print(f"[save] {out}  (feats_256 shape {feats_sub.shape})", flush=True)
    print(f"   NC matrix diag mean = {np.diag(nc_mat).mean():.3f}, "
          f"off-diag mean = {nc_mat[~np.eye(NL, dtype=bool)].mean():.3f} "
          f"(ideal off-diag = {-1/(NL-1):.3f})", flush=True)


if __name__ == '__main__':
    main()
