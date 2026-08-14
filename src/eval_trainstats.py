"""
Calibration-free standardization: each client uploads its TRAINING-feature
mean/std (computed on its own local training partition, no test/calibration
data, no extra round). Inference-only; reuses the same aggregation and the same
eval/calib split as eval_batch_zscore.py so the numbers are directly comparable.

Reports, per (dataset, alpha, K):
  trainstats : calibration-free (this method)
  frozen     : per-client stats from a 1,024 unlabeled TEST calib pool (sanity;
               must reproduce eval_batch_zscore's frozen number)
  global     : full-test-set stats (the paper's default)
"""
import os, argparse, json
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from rebuild8 import generate_etf, prepare_data, device
from eval_ablation_RIJ import load_backbones, forward_features
from eval_batch_zscore import _aggregate, acc_frozen, acc_batch


def client_train_stats(bb, loader):
    """Raw fc-feature mean/std over a client's own training data (same extraction
    path as forward_features: conv1->bn1->layer1..4->pool->fc, NO final L2)."""
    bb = bb.to(device).eval()
    feats = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            xx = F.relu(bb.bn1(bb.conv1(x)))
            xx = bb.layer1(xx); xx = bb.layer2(xx); xx = bb.layer3(xx); xx = bb.layer4(xx)
            xx = bb.pool(xx).flatten(1)
            feats.append(bb.fc(xx).cpu())
    bb.cpu(); torch.cuda.empty_cache()
    Ff = torch.cat(feats, 0)
    return Ff.mean(0, keepdim=True), Ff.std(0, keepdim=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['cifar10', 'cifar100'])
    ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--K', type=int, required=True)
    ap.add_argument('--NL', type=int, required=True)
    ap.add_argument('--FD', type=int, default=256)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--calib_n', type=int, default=1024)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    NC, NL, FD = a.K, a.NL, a.FD
    etf = generate_etf(NL, FD)

    if a.dataset == 'cifar10':
        cal, ccl, tl, ccc = prepare_data(NC, a.alpha, NL)
        clean_ds = datasets.CIFAR10(root='./data', train=True, download=True,
            transform=transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]))
    else:
        from rebuild8_cifar100 import prepare_data_cifar100, CIFAR100_MEAN, CIFAR100_STD
        cal, ccl, tl, ccc = prepare_data_cifar100(NC, a.alpha, NL)
        clean_ds = datasets.CIFAR100(root='./data', train=True, download=True,
            transform=transforms.Compose([transforms.ToTensor(),
                transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)]))

    bbs = load_backbones(a.save_dir, NC, FD)
    all_raw, labels, _ = forward_features(bbs, [{} for _ in range(NC)], tl, etf,
                                          NC, FD, NL, use_experts=False)
    labels = np.asarray(labels); N = len(labels)
    valid = [k for k in range(NC) if all_raw[k] is not None]
    if not valid:
        raise SystemExit(f"No backbones under {a.save_dir}")
    w = {k: float(np.sqrt(sum(ccc.get(k, {}).values()))) for k in valid}

    perm = np.random.RandomState(a.seed).permutation(N)
    calib_idx = perm[:a.calib_n]; eval_idx = perm[a.calib_n:]

    # calibration-free: each client's own training-feature mean/std
    train_mu_sd = {}
    for k in valid:
        if k not in cal:
            continue
        cidx_k = cal[k].dataset.indices
        ld = DataLoader(Subset(clean_ds, cidx_k), batch_size=256, shuffle=False,
                        num_workers=4, pin_memory=True)
        train_mu_sd[k] = client_train_stats(bbs[k], ld)
    valid_ts = [k for k in valid if k in train_mu_sd]
    preds = _aggregate(all_raw, valid_ts, w, etf, eval_idx, mu_sd=train_mu_sd)
    ts_acc = float((preds == labels[eval_idx]).mean())

    frozen = acc_frozen(all_raw, valid, w, etf, labels, eval_idx, calib_idx)
    ne = len(eval_idx)
    glob = acc_batch(all_raw, valid, w, etf, labels, eval_idx, ne)

    print(f"[{a.dataset}] a={a.alpha} K={NC} | eval_n={ne} clients={len(valid_ts)}/{NC} | "
          f"trainstats={ts_acc*100:.2f}%  frozen={frozen*100:.2f}%  global={glob*100:.2f}%")
    out = a.out or f"results/trainstats_{a.dataset}_a{a.alpha}_k{NC}_s{a.seed}.json"
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    json.dump({'dataset': a.dataset, 'alpha': a.alpha, 'K': NC, 'seed': a.seed,
               'eval_n': int(ne), 'clients': len(valid_ts),
               'trainstats_acc': ts_acc, 'frozen_acc': frozen, 'global_acc': glob},
              open(out, 'w'), indent=2)
    print(f"saved {out}")


if __name__ == '__main__':
    main()
