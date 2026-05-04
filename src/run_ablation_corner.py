"""
消融 Corner Test: 在 K=20 上跑 A-F variants
验证: 大 K 时 Expert (intrinsic) 组件变得更重要
"""
import torch, torch.nn.functional as F
import numpy as np, json, time, os, copy, argparse
from resnet18_filter_merge import ResNet18Backbone, union_aggregate_resnet18
from rebuild8 import (prepare_data, generate_etf, train_bb, train_experts,
                       compute_stats, precompute_all, device, USE_BF16,
                       cross_client_per_client_logits, d3_softmax_ensemble)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--n_clients', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    ALPHA = args.alpha; NC = args.n_clients; SEED = args.seed
    NL = 10; FD = 256; LD = 32; EPB = 600; EPE = 600
    torch.manual_seed(SEED); np.random.seed(SEED)

    print(f"\n{'='*60}")
    print(f"  消融 Corner Test: α={ALPHA}, K={NC}, seed={SEED}")
    print(f"{'='*60}")

    etf = generate_etf(NL, FD)
    cal, ccl, tl, ccc = prepare_data(NC, ALPHA, NL)

    # 训练
    bbs = []; client_exps = []
    t0 = time.time()
    for k in range(NC):
        cls = sorted(ccc[k].keys())
        print(f"  Client {k}: {len(cls)} cls, {sum(ccc[k].values())} samp")
        bb = ResNet18Backbone(FD)
        bb = train_bb(bb, cal[k], cls, etf, EPB)
        exps = train_experts(bb, ccl[k], cls, etf, NL, FD, LD, EPE)
        bbs.append(bb.cpu())
        client_exps.append({c: exp.cpu() for c, exp in exps.items()})
        torch.cuda.empty_cache()
    train_time = time.time() - t0

    # Union (Filter Merge)
    print(f"\n  Filter Merge...")
    ubb = union_aggregate_resnet18([copy.deepcopy(bb) for bb in bbs], FD, 0.95, device)

    # Precompute
    data = precompute_all(bbs, client_exps, ubb, tl, etf, ccc, NL)
    labels = data['labels']

    # === Variants ===
    R = {}

    # A: ETF only (client avg) — 平均所有客户端的 ETF logits
    etf_d = etf.to(device)
    all_logits = torch.zeros(data['N'], NL)
    for k in range(NC):
        bbs[k] = bbs[k].to(device); bbs[k].eval()
        offset = 0
        with torch.no_grad():
            for x, y in tl:
                x = x.to(device); bs = x.size(0)
                f = bbs[k](x)
                all_logits[offset:offset+bs] += torch.mm(f, etf_d.T).cpu()
                offset += bs
        bbs[k] = bbs[k].cpu(); torch.cuda.empty_cache()
    R['A: ETF only (client avg)'] = (all_logits.argmax(1).numpy() == labels).mean()

    # B: Expert only — 用 min reconstruction error
    from rebuild8 import expert_original
    R['B: Expert only'] = (expert_original(data) == labels).mean()

    # C: ETF + Expert (client avg) — A 的 logits + expert error 融合
    # 简单版: 用 cross_client_per_client_logits 但不做 filter merge
    # (这里直接用 B6 ensemble 方案，alpha=0.3)
    from rebuild8 import route_ensemble_logits
    R['C: ETF + Expert (client avg)'] = (route_ensemble_logits(data, alpha=0.3) == labels).mean()

    # D: ETF + Filter Merge only — union 预测
    R['D: ETF + Filter Merge'] = (data['union_preds'] == labels).mean()

    # E: Expert + Filter Merge — expert only (同 B，filter merge 不影响 expert)
    R['E: Expert + Filter Merge'] = R['B: Expert only']

    # F: Full (Ours) — union + expert ensemble
    R['F: Full (Ours)'] = (cross_client_per_client_logits(data, alpha=0.3, min_n=10) == labels).mean()

    # Print
    print(f"\n{'='*60}")
    print(f"  消融结果: α={ALPHA}, K={NC}")
    print(f"{'='*60}")
    for name, acc in R.items():
        marker = " ★" if name.startswith('F') else ""
        print(f"  {name:>35s}: {acc*100:>6.2f}%{marker}")

    # Save
    os.makedirs('results', exist_ok=True)
    out = {
        'dataset': 'cifar10', 'alpha': ALPHA, 'n_clients': NC, 'seed': SEED,
        'train_time': train_time, 'results': R
    }
    path = f"results/ablation_corner_a{ALPHA}_k{NC}_s{SEED}.json"
    json.dump(out, open(path, 'w'), indent=2)
    print(f"  Saved: {path}")

if __name__ == '__main__':
    main()
