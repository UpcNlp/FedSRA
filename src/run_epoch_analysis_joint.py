"""
run_epoch_analysis_joint.py - 单阶段联合训练 + 共享专家
=========================================================
把 BB 训练和 expert 训练合并成一个 epoch loop, 单网络共享专家。
评估时只需要 forward 一次, 不再需要重新训练 expert。
→ 适合做密集 epoch 扫描 (10 → 600)。

用法:
  python run_epoch_analysis_joint.py --alpha 0.05 --n_clients 10 --seed 42
  python run_epoch_analysis_joint.py --alpha 0.1  --max_ep 600
  python run_epoch_analysis_joint.py --alpha 0.3  --snapshots 50,100,200,300,400,500,600

4 个 α 并行 (GPU 0/2/4/5):
  CUDA_VISIBLE_DEVICES=0 python run_epoch_analysis_joint.py --alpha 0.05 &
  CUDA_VISIBLE_DEVICES=2 python run_epoch_analysis_joint.py --alpha 0.1  &
  CUDA_VISIBLE_DEVICES=4 python run_epoch_analysis_joint.py --alpha 0.3  &
  CUDA_VISIBLE_DEVICES=5 python run_epoch_analysis_joint.py --alpha 0.5  &
  wait
"""
import sys
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json, time, os, copy, argparse
from tqdm import tqdm

import rebuild8
rebuild8.DL_KWARGS = dict(num_workers=4, pin_memory=True, persistent_workers=False)

from resnet18_filter_merge import ResNet18Backbone
from rebuild8 import (
    prepare_data, generate_etf, device, USE_BF16,
    cross_client_per_client_logits, expert_original,
    etf_cl, etf_al,
)
from shared_expert import SharedConditionalExpert, shared_expert_loss


def train_joint(bb, expert, loader, classes, etf, max_epochs, snapshot_epochs,
                lr_bb=1e-3, lr_exp=1e-3, NL=10, FD=256, client_id=None):
    """单阶段: BB (ETF loss) + 共享 expert (重构+排斥 loss, detach feat 不污染 BB)。

    两个独立 optimizer + cosine schedule, 互不干扰。
    每个 batch 一次 BB forward; expert 复用 feat.detach() 不需要二次前向。
    """
    bb = bb.to(device); ed = etf.to(device); expert = expert.to(device)
    bb.train(); expert.train()

    opt_bb = torch.optim.Adam(bb.parameters(), lr=lr_bb)
    opt_exp = torch.optim.Adam(expert.parameters(), lr=lr_exp)
    sch_bb = torch.optim.lr_scheduler.CosineAnnealingLR(opt_bb, T_max=max_epochs)
    sch_exp = torch.optim.lr_scheduler.CosineAnnealingLR(opt_exp, T_max=max_epochs)
    amp = (torch.amp.autocast('cuda', dtype=torch.bfloat16) if USE_BF16
           else torch.amp.autocast('cuda', enabled=False))

    ncl = len(classes)
    snap_set = set(snapshot_epochs)
    snapshots = {}

    desc = f" Joint c{client_id}" if client_id is not None else " Joint"
    pbar = tqdm(range(max_epochs), desc=desc, ncols=100, mininterval=10.0, file=sys.stdout)
    for ep in pbar:
        el_bb = 0.0; el_e = 0.0; nb = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)

            # ── BB forward + loss
            with amp:
                feat = bb(x)
                if ncl >= 2:
                    loss_bb = etf_cl(feat, y, ed) + 0.5 * etf_al(feat, y, ed)
                else:
                    loss_bb = etf_al(feat, y, ed)

            # ── Expert step (detach feat, fp32 for stability)
            feat_det = feat.detach().float()
            loss_exp, _ = shared_expert_loss(feat_det, y, expert, ed, NL, FD)
            opt_exp.zero_grad(set_to_none=True); loss_exp.backward(); opt_exp.step()

            # ── BB step (backward through bb subgraph only)
            opt_bb.zero_grad(set_to_none=True); loss_bb.backward(); opt_bb.step()

            el_bb += loss_bb.item(); el_e += loss_exp.item(); nb += 1

        sch_bb.step(); sch_exp.step()
        ep_num = ep + 1
        pbar.set_postfix(bb=f"{el_bb/max(nb,1):.3f}", exp=f"{el_e/max(nb,1):.3f}")

        if ep_num in snap_set:
            snapshots[ep_num] = (
                copy.deepcopy(bb.cpu().state_dict()),
                copy.deepcopy(expert.cpu().state_dict()),
            )
            bb.to(device); expert.to(device)
            tqdm.write(f"      ep{ep_num}/{max_epochs} "
                       f"bb={el_bb/max(nb,1):.4f} exp={el_e/max(nb,1):.4f} [snap]")
    pbar.close()
    return bb, expert, snapshots


def evaluate_at_epoch_joint(bb_sds, expert_sds, client_classes, tl, etf, ccc,
                             NC, NL, FD, hd, ld, n_blocks):
    """znorm+sqrt(n) 评估; 每个 client 用共享 expert 算所有自有 class 的重构误差。"""
    ed = etf.to(device)
    N = 10000
    all_raw = []; all_labels = []
    errors = torch.full((NC, N, NL), float('inf'))

    for k in range(NC):
        bb = ResNet18Backbone(FD)
        bb.load_state_dict(bb_sds[k])
        bb = bb.to(device); bb.eval()

        exp = SharedConditionalExpert(FD, FD, hd, ld, n_blocks)
        exp.load_state_dict(expert_sds[k])
        exp = exp.to(device); exp.eval()

        my_classes = client_classes[k]
        feats = []; offset = 0
        with torch.no_grad():
            for x, y in tl:
                x = x.to(device); bs = x.size(0)
                # raw features (不 normalize) — 跟 znorm 路径一致
                xx = F.relu(bb.bn1(bb.conv1(x)))
                xx = bb.layer1(xx); xx = bb.layer2(xx)
                xx = bb.layer3(xx); xx = bb.layer4(xx)
                xx = bb.pool(xx).flatten(1)
                feat = bb.fc(xx)
                feats.append(feat.cpu())
                f_norm = F.normalize(feat, dim=1)
                # 共享 expert: 每个自有类做一次 forward
                for c in my_classes:
                    fr, _ = exp(f_norm, ed[c].unsqueeze(0).expand(bs, -1))
                    errors[k, offset:offset+bs, c] = ((f_norm - fr) ** 2).mean(1).cpu()
                if k == 0:
                    all_labels.append(y)
                offset += bs
        all_raw.append(torch.cat(feats, 0))
        bb = bb.cpu(); exp = exp.cpu()
        torch.cuda.empty_cache()

    labels = torch.cat(all_labels).numpy()

    sample_count = {}
    for k in range(NC):
        for c in client_classes[k]:
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
    parser.add_argument('--max_ep', type=int, default=600)
    parser.add_argument('--lr_bb', type=float, default=1e-3)
    parser.add_argument('--lr_exp', type=float, default=1e-3)
    parser.add_argument('--hd', type=int, default=384, help='shared expert hidden width')
    parser.add_argument('--ld', type=int, default=64, help='shared expert latent dim')
    parser.add_argument('--n_blocks', type=int, default=3, help='res blocks per side')
    parser.add_argument('--snapshots', type=str,
                        default='10,20,30,50,75,100,150,200,250,300,400,500,600',
                        help='comma-separated epoch list for evaluation')
    args = parser.parse_args()

    ALPHA = args.alpha; SEED = args.seed; NC = args.n_clients
    NL = 10; FD = 256
    MAX_EP = args.max_ep
    HD, LD, NB = args.hd, args.ld, args.n_blocks
    SNAPSHOT_EPOCHS = sorted(set(int(x) for x in args.snapshots.split(',')))
    assert max(SNAPSHOT_EPOCHS) <= MAX_EP, \
        f"snapshot {max(SNAPSHOT_EPOCHS)} 超过 max_ep {MAX_EP}"

    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    # Count expert params (info only)
    _probe = SharedConditionalExpert(FD, FD, HD, LD, NB)
    n_params = sum(p.numel() for p in _probe.parameters())
    del _probe

    print(f"\n{'='*70}")
    print(f"  Epoch Analysis (Joint, Shared Expert): α={ALPHA}, K={NC}, seed={SEED}")
    print(f"  Max epochs: {MAX_EP}")
    print(f"  Snapshots ({len(SNAPSHOT_EPOCHS)}): {SNAPSHOT_EPOCHS}")
    print(f"  Expert: hd={HD}, ld={LD}, n_blocks={NB} → {n_params/1e6:.2f}M params")
    print(f"  LR: bb={args.lr_bb}, exp={args.lr_exp}")
    print(f"{'='*70}")

    etf = generate_etf(NL, FD)
    cal, ccl, tl, ccc = prepare_data(NC, ALPHA, NL)

    # ──────────────────────────────────────────────
    # Phase 1: 各 client 联合训练, 在 SNAPSHOT_EPOCHS 处保存 (bb, expert)
    # ──────────────────────────────────────────────
    print(f"\n  Phase 1: Joint train ({MAX_EP} ep, single-stage)")
    all_snapshots = {ep: {'bb': [], 'exp': []} for ep in SNAPSHOT_EPOCHS}
    client_classes = []
    t0_total = time.time()

    for k in range(NC):
        cls = sorted(ccc[k].keys())
        client_classes.append(cls)
        print(f"\n  Client {k}: {len(cls)} cls, {sum(ccc[k].values())} samp")
        bb = ResNet18Backbone(FD)
        expert = SharedConditionalExpert(FD, FD, HD, LD, NB)
        bb, expert, snaps = train_joint(
            bb, expert, cal[k], cls, etf, MAX_EP, SNAPSHOT_EPOCHS,
            lr_bb=args.lr_bb, lr_exp=args.lr_exp, NL=NL, FD=FD, client_id=k)
        bb.cpu(); expert.cpu()
        del bb, expert
        torch.cuda.empty_cache()
        for ep in SNAPSHOT_EPOCHS:
            bb_sd, exp_sd = snaps[ep]
            all_snapshots[ep]['bb'].append(bb_sd)
            all_snapshots[ep]['exp'].append(exp_sd)

    train_time = time.time() - t0_total
    print(f"\n  Joint training done: {train_time:.0f}s ({train_time/60:.1f} min)")

    # ──────────────────────────────────────────────
    # Phase 2: 逐 snapshot 评估 (forward only, no retrain)
    # ──────────────────────────────────────────────
    print(f"\n  Phase 2: Evaluating {len(SNAPSHOT_EPOCHS)} snapshots")
    results = {}
    for ep in SNAPSHOT_EPOCHS:
        print(f"\n  --- Epoch {ep} ---")
        t0 = time.time()
        accs = evaluate_at_epoch_joint(
            all_snapshots[ep]['bb'], all_snapshots[ep]['exp'],
            client_classes, tl, etf, ccc, NC, NL, FD,
            hd=HD, ld=LD, n_blocks=NB)
        results[ep] = accs
        elapsed = time.time() - t0
        print(f"    Union={accs['union']*100:.2f}%  Expert={accs['expert']*100:.2f}%  "
              f"Full={accs['full']*100:.2f}% (a={accs['best_alpha']})  ({elapsed:.0f}s)")

    # ──────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Summary: α={ALPHA}, K={NC} (joint, shared expert)")
    print(f"{'='*70}")
    print(f"  {'Epoch':>6s}  {'Union':>8s}  {'Expert':>8s}  {'Full':>8s}  {'best_a':>6s}")
    for ep in SNAPSHOT_EPOCHS:
        r = results[ep]
        print(f"  {ep:>6d}  {r['union']*100:>7.2f}%  {r['expert']*100:>7.2f}%  "
              f"{r['full']*100:>7.2f}%  {r['best_alpha']:>5.1f}")

    os.makedirs('results', exist_ok=True)
    out = {
        'experiment': 'epoch_analysis_joint',
        'alpha': ALPHA, 'n_clients': NC, 'seed': SEED,
        'max_epb': MAX_EP,
        'expert_config': {'hd': HD, 'ld': LD, 'n_blocks': NB, 'n_params': n_params},
        'snapshot_epochs': SNAPSHOT_EPOCHS,
        'train_time': train_time,
        'results': {str(k): v for k, v in results.items()},
    }
    path = f"results/epoch_analysis_joint_a{ALPHA}_k{NC}_s{SEED}.json"
    json.dump(out, open(path, 'w'), indent=2)
    print(f"\n  Saved: {path}")


if __name__ == '__main__':
    main()
