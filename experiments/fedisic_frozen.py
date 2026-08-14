"""
Fed-ISIC2019 frozen-calibration standardization for RGA on the real federation.
Inference-only: loads the trained center backbones, estimates each
center's feature mean/std ONCE from a 1,024-sample unlabeled calibration pool of
the pooled test stream, then applies them per sample (batch-size independent).

Reuses aggregate_logits: passing the frozen moments and reading its
'rga_client_local_moments' output IS the frozen-standardized aggregation.
Also recomputes client-local-moments and inference-batch as a sanity check
(reproduces the ~57.3 and ~61.9 balanced-accuracy rows of the Fed-ISIC table).
"""
import argparse, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from fedisic_fedsra import (N_CLASSES, Backbone, IsicParquetDataset, build_transform)
from medmnist_fedsra import aggregate_logits, generate_etf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def extract(models, loader):
    raws = []; labels = None
    for i, m in enumerate(models):
        m.to(device).eval(); feats = []; ys = []
        for x, y in loader:
            feats.append(m.forward_raw(x.to(device, non_blocking=True)).float().cpu())
            if i == 0:
                ys.append(y)
        raws.append(torch.cat(feats))
        if i == 0:
            labels = torch.cat(ys).numpy()
        m.cpu(); torch.cuda.empty_cache()
    return raws, labels


def ba(logits, labels):
    return balanced_accuracy_score(labels, logits.argmax(1).numpy()) * 100


def run_seed(ckpt_dir, test_loader, etf, fd, calib_n, seed):
    cps = sorted(ckpt_dir.glob("center_*.pt"))
    cps = [p for p in cps if ".ep" not in p.name]
    models, ck_moments, counts = [], [], []
    for p in cps:
        d = torch.load(p, map_location="cpu", weights_only=False)
        m = Backbone(fd, pretrained=False)
        m.load_state_dict({k: v.float() for k, v in d["model"].items()})
        models.append(m)
        ck_moments.append((d["mu"].float(), d["sd"].float()))
        counts.append(int(d["meta"]["n"]))
    w = np.sqrt(np.asarray(counts))
    raws, labels = extract(models, test_loader)
    N = len(labels)

    # frozen moments from a 1,024-sample unlabeled calibration pool of the test stream
    calib = np.random.RandomState(seed).permutation(N)[:min(calib_n, N // 2)]
    ct = torch.as_tensor(calib)
    frozen = [(raws[i][ct].mean(0), raws[i][ct].std(0)) for i in range(len(models))]

    out_ck = aggregate_logits(raws, ck_moments, w, etf)   # sanity: client-local + batch
    out_fz = aggregate_logits(raws, frozen, w, etf)        # frozen (read local key)
    return {
        "frozen": ba(out_fz["rga_client_local_moments"], labels),
        "client_local": ba(out_ck["rga_client_local_moments"], labels),
        "inference_batch": ba(out_ck["rga_full_batch_diagnostic"], labels),
        "n_centers": len(models), "eval_n": int(N),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_parquet", type=Path, required=True)
    ap.add_argument("--ckpt_root", type=Path, required=True,
                    help="dir containing fedsra_s{seed}/ checkpoint subdirs")
    ap.add_argument("--seeds", default="42,123,0")
    ap.add_argument("--feature_dim", type=int, default=256)
    ap.add_argument("--image_size", type=int, default=144)
    ap.add_argument("--calib_n", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--out", default="results/fedisic_frozen.json")
    a = ap.parse_args()

    test_df = pd.read_parquet(a.test_parquet)
    etf = generate_etf(N_CLASSES, a.feature_dim, 42)
    ds = IsicParquetDataset(test_df, build_transform(a.image_size, False))
    tl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    rows = []
    for s in [int(x) for x in a.seeds.split(",")]:
        ck = a.ckpt_root / f"fedsra_s{s}"
        if not ck.exists():
            print(f"[skip] {ck} missing"); continue
        r = run_seed(ck, tl, etf, a.feature_dim, a.calib_n, s)
        r["seed"] = s
        print(f"[s{s}] centers={r['n_centers']} eval_n={r['eval_n']} | "
              f"frozen={r['frozen']:.2f}  client_local={r['client_local']:.2f} "
              f"inference_batch={r['inference_batch']:.2f}", flush=True)
        rows.append(r)

    if rows:
        for key in ["frozen", "client_local", "inference_batch"]:
            v = np.array([r[key] for r in rows])
            print(f"  {key:16s}: {v.mean():.2f} +/- {v.std():.2f}")
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(a.out, "w"), indent=2)
        print(f"saved {a.out}")


if __name__ == "__main__":
    main()
