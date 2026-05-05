"""
Per-class Client Selection + FM on subset
支持: 保存/加载训练好的模型权重, 避免重复训练
"""
import torch, torch.nn.functional as F
import numpy as np, json, time, os, copy, argparse
from collections import defaultdict
from resnet18_filter_merge import ResNet18Backbone, union_aggregate_resnet18
from rebuild8 import (prepare_data, generate_etf, train_bb, train_experts,
                       compute_stats, precompute_all, device, USE_BF16,
                       cross_client_per_client_logits, expert_original)

def save_all_models(bbs, client_exps, ccc, save_dir):
    """保存所有 client 的 backbone + experts"""
    os.makedirs(save_dir, exist_ok=True)
    for k in range(len(bbs)):
        if bbs[k] is None: continue
        client_dir = os.path.join(save_dir, f"client_{k}")
        os.makedirs(client_dir, exist_ok=True)
        torch.save(bbs[k].state_dict(), os.path.join(client_dir, "backbone.pt"))
        for c, exp in client_exps[k].items():
            torch.save(exp.state_dict(), os.path.join(client_dir, f"expert_{c}.pt"))
    # 保存 ccc
    torch.save(dict(ccc), os.path.join(save_dir, "ccc.pt"))
    print(f"  模型已保存到 {save_dir}")

def load_all_models(save_dir, NC, NL, FD=256, LD=32):
    """加载所有 client 的 backbone + experts"""
    from rebuild8 import ConditionalExpert
    bbs = []; client_exps = []
    for k in range(NC):
        client_dir = os.path.join(save_dir, f"client_{k}")
        bb = ResNet18Backbone(FD)
        bb.load_state_dict(torch.load(os.path.join(client_dir, "backbone.pt"),
                                       map_location='cpu', weights_only=True))
        bbs.append(bb)
        exps = {}
        for c in range(NL):
            exp_path = os.path.join(client_dir, f"expert_{c}.pt")
            if os.path.exists(exp_path):
                exp = ConditionalExpert(FD, FD, 128, LD)
                exp.load_state_dict(torch.load(exp_path, map_location='cpu',
                                                weights_only=True))
                exps[c] = exp
        client_exps.append(exps)
    ccc = torch.load(os.path.join(save_dir, "ccc.pt"), weights_only=False)
    print(f"  模型已从 {save_dir} 加载 ({NC} clients)")
    return bbs, client_exps, ccc

def select_authorities(ccc, NC, NL):
    """对每个类, 选出训练样本最多的 client"""
    authority = {}
    for c in range(NL):
        best_k = -1; best_n = 0
        for k in range(NC):
            n = ccc.get(k,{}).get(c, 0)
            if n > best_n:
                best_n = n; best_k = k
        if best_k >= 0:
            authority[c] = best_k
    selected = sorted(set(authority.values()))
    print(f"    Authority: {authority}")
    print(f"    Selected {len(selected)} / {NC} clients: {selected}")
    return authority, selected

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--n_clients', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--load', action='store_true', help='Load saved models')
    parser.add_argument('--skip_full_fm', action='store_true', help='Skip full FM')
    args = parser.parse_args()

    ALPHA = args.alpha; NC = args.n_clients; SEED = args.seed
    NL = 10; FD = 256; LD = 32; EPB = 600; EPE = 600
    torch.manual_seed(SEED); np.random.seed(SEED)

    save_dir = f"saved_models/a{ALPHA}_k{NC}_s{SEED}"

    print(f"\n{'='*60}")
    print(f"  Selected FM: alpha={ALPHA}, K={NC}, seed={SEED}")
    print(f"{'='*60}")

    etf = generate_etf(NL, FD)

    if args.load and os.path.exists(save_dir):
        # 加载已训练的模型
        bbs, client_exps, ccc = load_all_models(save_dir, NC, NL, FD, LD)
        cal, ccl, tl, _ = prepare_data(NC, ALPHA, NL)
        train_time = 0
    else:
        # 训练
        cal, ccl, tl, ccc = prepare_data(NC, ALPHA, NL)
        bbs = []; client_exps = []
        t0 = time.time()
        for k in range(NC):
            cls = sorted(ccc.get(k,{}).keys())
            if not cls:
                print(f"  Client {k}: EMPTY, skipping")
                bbs.append(None); client_exps.append({}); continue
            print(f"  Client {k}: {len(cls)} cls, {sum(ccc.get(k,{}).values())} samp")
            bb = ResNet18Backbone(FD)
            bb = train_bb(bb, cal[k], cls, etf, EPB)
            exps = train_experts(bb, ccl[k], cls, etf, NL, FD, LD, EPE)
            bbs.append(bb.cpu())
            client_exps.append({c: exp.cpu() for c, exp in exps.items()})
            torch.cuda.empty_cache()
        train_time = time.time() - t0
        # 保存
        save_all_models(bbs, client_exps, ccc, save_dir)

    # === A: Full FM ===
    if not args.skip_full_fm:
        print(f"\n  === A: Full FM (all {NC} clients) ===")
        t0 = time.time()
        ubb_full = union_aggregate_resnet18(
            [copy.deepcopy(bb) for bb in bbs], FD, 0.95, device)
        t_full = time.time() - t0
        data_full = precompute_all(bbs, client_exps, ubb_full, tl, etf, ccc, NL)
        labels = data_full['labels']
        acc_full_union = (data_full['union_preds'] == labels).mean()
        acc_full_ours = (cross_client_per_client_logits(
            data_full, alpha=0.3, min_n=10) == labels).mean()
        print(f"    Union: {acc_full_union*100:.2f}%, Full: {acc_full_ours*100:.2f}%, FM time: {t_full:.1f}s")
    else:
        print(f"\n  === A: Full FM SKIPPED ===")
        labels = None; acc_full_union = None; acc_full_ours = None; t_full = None

    # === B: Selected FM ===
    print(f"\n  === B: Selected FM ===")
    authority, selected = select_authorities(ccc, NC, NL)

    t0 = time.time()
    selected_bbs = [bbs[k] for k in selected]
    ubb_sel = union_aggregate_resnet18(
        [copy.deepcopy(bb) for bb in selected_bbs], FD, 0.95, device)
    t_sel = time.time() - t0

    data_sel = precompute_all(bbs, client_exps, ubb_sel, tl, etf, ccc, NL)
    if labels is None:
        labels = data_sel['labels']
    acc_sel_union = (data_sel['union_preds'] == labels).mean()
    acc_sel_ours = (cross_client_per_client_logits(
        data_sel, alpha=0.3, min_n=10) == labels).mean()
    print(f"    Union: {acc_sel_union*100:.2f}%, Full: {acc_sel_ours*100:.2f}%, FM time: {t_sel:.1f}s")
    print(f"    FM on {len(selected)} clients instead of {NC}")

    # === C: Client avg (no FM) ===
    print(f"\n  === C: Client avg (no FM) ===")
    ed = etf.to(device)
    N = len(labels)
    union_logits = torch.zeros(N, NL)
    for k in range(NC):
        bbs[k] = bbs[k].to(device); bbs[k].eval()
        offset = 0
        with torch.no_grad():
            for x, y in tl:
                x = x.to(device); bs = x.size(0)
                f = bbs[k](x)
                union_logits[offset:offset+bs] += torch.mm(f, ed.T).cpu()
                offset += bs
        bbs[k] = bbs[k].cpu(); torch.cuda.empty_cache()
    acc_avg = (union_logits.argmax(1).numpy() == labels).mean()
    print(f"    Union: {acc_avg*100:.2f}%")

    # === D: Expert only ===
    print(f"\n  === D: Expert only ===")
    acc_expert = (expert_original(data_sel) == labels).mean()
    print(f"    Expert: {acc_expert*100:.2f}%")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Results: alpha={ALPHA}, K={NC}")
    print(f"{'='*60}")
    print(f"    Client avg (no FM):      {acc_avg*100:.2f}%")
    print(f"    Expert only:             {acc_expert*100:.2f}%")
    print(f"    Selected FM ({len(selected):>2d}/{NC:>2d}):    "
          f"{acc_sel_union*100:.2f}% union, {acc_sel_ours*100:.2f}% full  ({t_sel:.1f}s)")
    if acc_full_ours is not None:
        print(f"    Full FM ({NC:>2d}/{NC:>2d}):         "
              f"{acc_full_union*100:.2f}% union, {acc_full_ours*100:.2f}% full  ({t_full:.1f}s)")
        print(f"    Gap (Full-Selected):     {(acc_full_ours-acc_sel_ours)*100:+.2f} pp")

    # Save
    os.makedirs('results', exist_ok=True)
    out = {
        'experiment': 'selected_fm', 'alpha': ALPHA,
        'n_clients': NC, 'seed': SEED, 'train_time': train_time,
        'n_selected': len(selected), 'selected_clients': selected,
        'results': {
            'client_avg': float(acc_avg),
            'expert_only': float(acc_expert),
            'selected_fm_union': float(acc_sel_union),
            'selected_fm_full': float(acc_sel_ours),
            'full_fm_union': float(acc_full_union) if acc_full_union else None,
            'full_fm_full': float(acc_full_ours) if acc_full_ours else None,
            'fm_time_selected': t_sel,
            'fm_time_full': t_full,
        }
    }
    path = f"results/selected_fm_a{ALPHA}_k{NC}_s{SEED}.json"
    json.dump(out, open(path, 'w'), indent=2)
    print(f"\n  Saved: {path}")

if __name__ == '__main__':
    main()
