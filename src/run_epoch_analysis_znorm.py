"""
run_epoch_analysis_znorm.py - Epoch 影响分析 (znorm 版)
======================================================
对齐 Co-Boost pretrain Le = {10, 30, 50, 100, 150, 300}
用 znorm+sqrt(n) 聚合 (不用 FM)

用法:
  python run_epoch_analysis_znorm.py --alpha 0.05 --n_clients 10 --seed 42
  python run_epoch_analysis_znorm.py --alpha 0.3  --n_clients 10 --seed 42
"""
import sys
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json, time, os, copy, argparse
from tqdm import tqdm

# monkey-patch DL_KWARGS for AMD
import rebuild8
rebuild8.DL_KWARGS = dict(num_workers=4, pin_memory=True, persistent_workers=False)

from resnet18_filter_merge import ResNet18Backbone
from rebuild8 import (prepare_data, generate_etf, train_exp, preextract,
                       ConditionalExpert, device, USE_BF16,
                       cross_client_per_client_logits, expert_original,
                       etf_cl, etf_al)


def train_bb_with_snapshots(bb, loader, classes, etf, max_epochs=300, lr=1e-3,
                             snapshot_epochs=[10, 30, 50, 100, 150, 300]):
    bb = bb.to(device); ed = etf.to(device); bb.train()
    opt = torch.optim.Adam(bb.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    amp = (torch.amp.autocast('cuda', dtype=torch.bfloat16) if USE_BF16
           else torch.amp.autocast('cuda', enabled=False))
    snapshots = {}
    ncl = len(classes)
    pbar = tqdm(range(max_epochs), desc="  BB", ncols=100, mininterval=10.0, file=sys.stdout)
    for ep in pbar:
        el = 0; nb = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            with amp:
                f = bb(x)
                if ncl >= 2: loss = etf_cl(f, y, ed) + 0.5 * etf_al(f, y, ed)
                else: loss = etf_al(f, y, ed)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            el += loss.item(); nb += 1
        sch.step()
        pbar.set_postfix(loss=f"{el/max(nb,1):.3f}")
        if (ep + 1) in snapshot_epochs:
            snapshots[ep + 1] = copy.deepcopy(bb.cpu().state_dict())
            bb = bb.to(device)
            tqdm.write(f"      BB {ep+1}/{max_epochs} loss={el/max(nb,1):.4f} [snapshot saved]")
    pbar.close()
    return bb, snapshots


def train_experts_for_snapshot(bb_sd, ccl_k, classes, etf, NL=10, FD=256, LD=32, EPE=200):
    bb = ResNet18Backbone(FD)
    bb.load_state_dict(bb_sd)
    bb = bb.to(device); bb.eval()
    ed = etf.to(device)
    exps = {}
    om = {c: torch.tensor([j for j in range(NL) if j != c], device=device) for c in range(NL)}
    for c in classes:
        if c not in ccl_k: continue
        cached = preextract(bb, ccl_k[c])
        if cached.size(0) < 5: continue
        exp = ConditionalExpert(FD, FD, 128, LD).to(device)
        exp = train_exp(exp, cached, ed[c], ed, om[c], FD, EPE, 1e-3)
        exps[c] = exp.cpu()
    bb = bb.cpu(); torch.cuda.empty_cache()
    return exps


def evaluate_at_epoch_znorm(bb_sds, client_exps, client_classes, tl, etf, ccc, NC, NL, FD):
    """znorm+sqrt(n) 评估 (不用 FM)"""
    ed = etf.to(device)
    N = 10000
    all_raw = []; all_labels = []
    errors = torch.full((NC, N, NL), float('inf'))

    for k in range(NC):
        bb = ResNet18Backbone(FD)
        bb.load_state_dict(bb_sds[k])
        bb = bb.to(device); bb.eval()
        for c in client_exps[k]:
            client_exps[k][c] = client_exps[k][c].to(device)

        feats = []; offset = 0
        with torch.no_grad():
            for x, y in tl:
                x = x.to(device); bs = x.size(0)
                # raw features (不 normalize) — 跟 run_znorm_scalability.py 一致
                xx = F.relu(bb.bn1(bb.conv1(x)))
                xx = bb.layer1(xx); xx = bb.layer2(xx)
                xx = bb.layer3(xx); xx = bb.layer4(xx)
                xx = bb.pool(xx).flatten(1)
                feat = bb.fc(xx)
                feats.append(feat.cpu())
                f_norm = F.normalize(feat, dim=1)
                for c, exp in client_exps[k].items():
                    fr, _ = exp(f_norm, ed[c].unsqueeze(0).expand(bs, -1))
                    errors[k, offset:offset+bs, c] = ((f_norm - fr)**2).mean(1).cpu()
                if k == 0: all_labels.append(y)
                offset += bs
        all_raw.append(torch.cat(feats, 0))
        bb = bb.cpu()
        for c in client_exps[k]:
            client_exps[k][c] = client_exps[k][c].cpu()
        torch.cuda.empty_cache()

    labels = torch.cat(all_labels).numpy()

    sample_count = {}
    for k in range(NC):
        for c in client_exps[k]:
            sample_count[(k, c)] = ccc.get(k, {}).get(c, 0)

    # znorm + sqrt(n) 聚合
    feat = torch.zeros(N, FD); w_sum = 0
    for k in range(NC):
        f = all_raw[k]
        f_z = (f - f.mean(0, keepdim=True)) / (f.std(0, keepdim=True) + 1e-8)
        w = np.sqrt(sum(ccc.get(k, {}).values()))
        feat += f_z * w; w_sum += w
    feat_n = F.normalize(feat / w_sum, dim=1)
    union_logits = feat_n @ etf.T
    union_preds = union_logits.argmax(1).numpy()
    acc_union = (union_preds == labels).mean()

    sl, _ = union_logits.sort(dim=1, descending=True)
    data = {
        'errors': errors, 'union_logits': union_logits,
        'union_preds': union_preds,
        'union_margin': (sl[:, 0] - sl[:, 1]).numpy(),
        'labels': labels, 'sample_count': sample_count,
        'K': NC, 'N': N, 'nc': NL,
    }

    acc_expert = (expert_original(data) == labels).mean()
    acc_full = {}
    for a in [0.3, 0.5, 1.0]:
        acc_full[a] = (cross_client_per_client_logits(data, alpha=a, min_n=10) == labels).mean()
    best_a = max(acc_full, key=acc_full.get)
    best_full = acc_full[best_a]

    return {
        'union': float(acc_union),
        'expert': float(acc_expert),
        'full': float(best_full),
        'best_alpha': float(best_a),
        'acc_full': {str(k): float(v) for k, v in acc_full.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_clients', type=int, default=10)
    args = parser.parse_args()

    ALPHA = args.alpha; SEED = args.seed; NC = args.n_clients
    NL = 10; FD = 256; LD = 32
    MAX_EPB = 300; EPE = 200
    SNAPSHOT_EPOCHS = [10, 30, 50, 100, 150, 300]

    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    print(f"\n{'='*60}")
    print(f"  Epoch Analysis (znorm): alpha={ALPHA}, K={NC}, seed={SEED}")
    print(f"  Snapshots: {SNAPSHOT_EPOCHS}")
    print(f"  Max BB epochs: {MAX_EPB}")
    print(f"{'='*60}")

    etf = generate_etf(NL, FD)
    cal, ccl, tl, ccc = prepare_data(NC, ALPHA, NL)

    # Phase 1: 训练 BB, 保存 snapshots
    print(f"\n  Phase 1: 训练 Backbone (一次性, max={MAX_EPB})")
    all_snapshots = {ep: [] for ep in SNAPSHOT_EPOCHS}
    client_classes = []

    t0_total = time.time()
    for k in range(NC):
        cls = sorted(ccc[k].keys())
        client_classes.append(cls)
        print(f"\n  Client {k}: {len(cls)} cls, {sum(ccc[k].values())} samp")
        bb = ResNet18Backbone(FD)
        bb, snapshots = train_bb_with_snapshots(
            bb, cal[k], cls, etf, MAX_EPB, snapshot_epochs=SNAPSHOT_EPOCHS)
        bb.cpu(); torch.cuda.empty_cache()
        for ep in SNAPSHOT_EPOCHS:
            all_snapshots[ep].append(snapshots[ep])

    train_time = time.time() - t0_total
    print(f"\n  BB 训练完成: {train_time:.0f}s ({train_time/60:.1f}min)")

    # Phase 2: 逐 snapshot 评估
    print(f"\n  Phase 2: 逐 snapshot 评估 (znorm)")
    results = {}
    for ep in SNAPSHOT_EPOCHS:
        print(f"\n  --- Epoch {ep} ---")
        t0 = time.time()

        client_exps = []
        for k in range(NC):
            print(f"    Client {k} experts...", end=" ", flush=True)
            exps = train_experts_for_snapshot(
                all_snapshots[ep][k], ccl[k], client_classes[k], etf, NL, FD, LD, EPE)
            client_exps.append(exps)
            print(f"{len(exps)} classes")

        accs = evaluate_at_epoch_znorm(
            all_snapshots[ep], client_exps, client_classes, tl, etf, ccc, NC, NL, FD)
        results[ep] = accs
        elapsed = time.time() - t0
        print(f"    Union={accs['union']*100:.2f}%  Expert={accs['expert']*100:.2f}%  "
              f"Full={accs['full']*100:.2f}% (a={accs['best_alpha']})  ({elapsed:.0f}s)")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Summary: alpha={ALPHA}, K={NC} (znorm)")
    print(f"{'='*60}")
    print(f"  {'Epoch':>8s}  {'Union':>8s}  {'Expert':>8s}  {'Full':>8s}  {'best_a':>6s}")
    for ep in SNAPSHOT_EPOCHS:
        r = results[ep]
        print(f"  {ep:>8d}  {r['union']*100:>7.2f}%  {r['expert']*100:>7.2f}%  "
              f"{r['full']*100:>7.2f}%  {r['best_alpha']:>5.1f}")

    # Save
    os.makedirs('results', exist_ok=True)
    out = {
        'experiment': 'epoch_analysis_znorm',
        'alpha': ALPHA, 'n_clients': NC, 'seed': SEED,
        'max_epb': MAX_EPB, 'epe': EPE,
        'snapshot_epochs': SNAPSHOT_EPOCHS,
        'train_time': train_time,
        'results': {str(k): v for k, v in results.items()},
    }
    path = f"results/epoch_analysis_znorm_a{ALPHA}_k{NC}_s{SEED}.json"
    json.dump(out, open(path, 'w'), indent=2)
    print(f"\n  Saved: {path}")


if __name__ == '__main__':
    main()
