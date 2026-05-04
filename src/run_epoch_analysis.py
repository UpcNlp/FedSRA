"""
训练 Epoch 影响分析 (高效版)
- Backbone 训练时保存 snapshot
- Expert 只在最终 epoch 训练一次
- 评估时: 各 snapshot 的 backbone + 同一批 expert (expert 不依赖 epoch)
  
更准确的做法: Expert 也保存 snapshot
但 expert 训练很快(几秒/class), 所以直接对每个 BB snapshot 重训 expert
只训练 BB 一次, expert 对每个 snapshot 重训
"""
import torch, torch.nn.functional as F
import numpy as np, json, time, os, copy, argparse
from resnet18_filter_merge import ResNet18Backbone, union_aggregate_resnet18
from rebuild8 import (prepare_data, generate_etf, train_bb, train_experts,
                       compute_stats, precompute_all, device, USE_BF16,
                       cross_client_per_client_logits, ConditionalExpert,
                       preextract, train_exp, etf_cl, etf_al)

def train_bb_with_snapshots(bb, loader, classes, etf, max_epochs=600, lr=1e-3,
                             snapshot_epochs=[50, 100, 200, 300, 600]):
    bb = bb.to(device); ed = etf.to(device); bb.train()
    opt = torch.optim.Adam(bb.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    amp = (torch.amp.autocast('cuda', dtype=torch.bfloat16) if USE_BF16
           else torch.amp.autocast('cuda', enabled=False))
    snapshots = {}
    ncl = len(classes)
    for ep in range(max_epochs):
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
        if (ep + 1) in snapshot_epochs:
            snapshots[ep + 1] = copy.deepcopy(bb.cpu().state_dict())
            bb = bb.to(device)
            print(f"      BB {ep+1}/{max_epochs} loss={el/max(nb,1):.4f} [snapshot]")
        elif (ep + 1) % 100 == 0:
            print(f"      BB {ep+1}/{max_epochs} loss={el/max(nb,1):.4f}")
    return bb, snapshots

def train_experts_for_snapshot(bb_sd, ccl_k, classes, etf, NL=10, FD=256, LD=32, EPE=200):
    """给一个 BB snapshot, 训练对应的 experts"""
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

def evaluate_at_epoch(bb_sds, client_exps, client_classes, tl, etf, ccc, NC, NL, FD):
    """给定 BB snapshots + experts, 做完整评估"""
    bbs = []
    for sd in bb_sds:
        bb = ResNet18Backbone(FD); bb.load_state_dict(sd); bbs.append(bb)
    
    # Filter merge
    ubb = union_aggregate_resnet18([copy.deepcopy(bb) for bb in bbs], FD, 0.95, device)
    
    # Precompute
    data = precompute_all(bbs, client_exps, ubb, tl, etf, ccc, NL)
    labels = data['labels']
    
    from rebuild8 import expert_original
    acc_union = (data['union_preds'] == labels).mean()
    acc_expert = (expert_original(data) == labels).mean()
    acc_full = (cross_client_per_client_logits(data, alpha=0.3, min_n=10) == labels).mean()
    
    return {'union': float(acc_union), 'expert': float(acc_expert), 'full': float(acc_full)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_clients', type=int, default=5)
    args = parser.parse_args()

    ALPHA = args.alpha; SEED = args.seed; NC = args.n_clients
    NL = 10; FD = 256; LD = 32
    MAX_EPB = 600; EPE = 200
    SNAPSHOT_EPOCHS = [50, 100, 200, 300, 400, 600]

    torch.manual_seed(SEED); np.random.seed(SEED)

    print(f"\n{'='*60}")
    print(f"  Epoch Analysis: alpha={ALPHA}, K={NC}, seed={SEED}")
    print(f"  Snapshots: {SNAPSHOT_EPOCHS}")
    print(f"{'='*60}")

    etf = generate_etf(NL, FD)
    cal, ccl, tl, ccc = prepare_data(NC, ALPHA, NL)

    # Phase 1: 训练 BB, 保存 snapshots (每个 client 只训练一次)
    print(f"\n  Phase 1: 训练 Backbone (一次性)")
    all_snapshots = {ep: [] for ep in SNAPSHOT_EPOCHS}
    client_classes = []

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

    # Phase 2: 对每个 snapshot epoch, 训练 expert + 评估
    print(f"\n  Phase 2: 逐 snapshot 评估")
    results = {}
    for ep in SNAPSHOT_EPOCHS:
        print(f"\n  --- Epoch {ep} ---")
        t0 = time.time()
        
        # 训练 expert (每个 snapshot 的 BB 特征空间不同, 必须重训)
        client_exps = []
        for k in range(NC):
            print(f"    Client {k} experts...", end=" ", flush=True)
            exps = train_experts_for_snapshot(
                all_snapshots[ep][k], ccl[k], client_classes[k], etf, NL, FD, LD, EPE)
            client_exps.append(exps)
            print(f"{len(exps)} classes")
        
        # 评估
        accs = evaluate_at_epoch(
            all_snapshots[ep], client_exps, client_classes, tl, etf, ccc, NC, NL, FD)
        results[ep] = accs
        elapsed = time.time() - t0
        print(f"    Union={accs['union']*100:.2f}%  Expert={accs['expert']*100:.2f}%  "
              f"Full={accs['full']*100:.2f}%  ({elapsed:.0f}s)")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Summary: alpha={ALPHA}, K={NC}")
    print(f"{'='*60}")
    print(f"  {'Epoch':>8s}  {'Union(Rel)':>10s}  {'Expert(Int)':>11s}  {'Full':>8s}")
    for ep in SNAPSHOT_EPOCHS:
        r = results[ep]
        print(f"  {ep:>8d}  {r['union']*100:>9.2f}%  {r['expert']*100:>10.2f}%  {r['full']*100:>7.2f}%")

    # Save
    os.makedirs('results', exist_ok=True)
    out = {
        'experiment': 'epoch_analysis', 'alpha': ALPHA,
        'n_clients': NC, 'seed': SEED,
        'snapshot_epochs': SNAPSHOT_EPOCHS,
        'results': {str(k): v for k, v in results.items()},
    }
    path = f"results/epoch_analysis_a{ALPHA}_k{NC}_s{SEED}.json"
    json.dump(out, open(path, 'w'), indent=2)
    print(f"\n  Saved: {path}")

if __name__ == '__main__':
    main()
