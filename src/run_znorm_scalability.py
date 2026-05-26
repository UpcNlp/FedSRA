"""
Scalability 实验: znorm+sqrt(n) 聚合
训练 + 保存模型 + 评估
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json, time, os, argparse
from resnet18_filter_merge import ResNet18Backbone
from rebuild8 import (prepare_data, generate_etf, train_bb, train_experts,
                       ConditionalExpert, device, USE_BF16,
                       cross_client_per_client_logits, expert_original)

DL_KWARGS = dict(num_workers=0, pin_memory=True, persistent_workers=False)

def save_models(bbs, client_exps, ccc, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for k in range(len(bbs)):
        if bbs[k] is None: continue
        cd = os.path.join(save_dir, f"client_{k}")
        os.makedirs(cd, exist_ok=True)
        torch.save(bbs[k].state_dict(), os.path.join(cd, "backbone.pt"))
        for c, exp in client_exps[k].items():
            torch.save(exp.state_dict(), os.path.join(cd, f"expert_{c}.pt"))
    torch.save(dict(ccc), os.path.join(save_dir, "ccc.pt"))
    print(f"  模型已保存到 {save_dir}")

def load_models(save_dir, NC, NL, FD=256):
    bbs = []; client_exps = []
    for k in range(NC):
        cd = os.path.join(save_dir, f"client_{k}")
        if not os.path.exists(cd):
            bbs.append(None); client_exps.append({}); continue
        bb = ResNet18Backbone(FD)
        bb.load_state_dict(torch.load(f"{cd}/backbone.pt", map_location='cpu', weights_only=True))
        bbs.append(bb)
        exps = {}
        for c in range(NL):
            ep = f"{cd}/expert_{c}.pt"
            if os.path.exists(ep):
                exp = ConditionalExpert(FD, FD, 128, 32)
                exp.load_state_dict(torch.load(ep, map_location='cpu', weights_only=True))
                exps[c] = exp
        client_exps.append(exps)
    ccc = torch.load(os.path.join(save_dir, "ccc.pt"), weights_only=False)
    return bbs, client_exps, ccc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--n_clients', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--no_save', action='store_true')
    args = parser.parse_args()

    ALPHA = args.alpha; NC = args.n_clients; SEED = args.seed
    NL = 10; FD = 256; LD = 32; EPB = 600; EPE = 600
    torch.manual_seed(SEED); np.random.seed(SEED)

    save_dir = f"saved_models/a{ALPHA}_k{NC}_s{SEED}"

    print(f"\n{'='*60}")
    print(f"  znorm Scalability: α={ALPHA}, K={NC}, seed={SEED}")
    print(f"{'='*60}")

    etf = generate_etf(NL, FD)
    cal, ccl, tl, ccc = prepare_data(NC, ALPHA, NL)

    gpu_peak_mb = 0.0  # default for --load mode
    if args.load and os.path.exists(save_dir):
        bbs, client_exps, ccc = load_models(save_dir, NC, NL, FD)
        print(f"  Loaded {sum(1 for b in bbs if b is not None)}/{NC} clients")
        train_time = 0
    else:
        bbs = []; client_exps = []
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for k in range(NC):
            cls = sorted(ccc.get(k, {}).keys())
            if not cls:
                print(f"  Client {k}: EMPTY")
                bbs.append(None); client_exps.append({}); continue
            print(f"  Client {k}: {len(cls)} cls, {sum(ccc[k].values())} samp")
            bb = ResNet18Backbone(FD)
            bb = train_bb(bb, cal[k], cls, etf, EPB, save_dir=save_dir, client_id=k)
            exps = train_experts(bb, ccl[k], cls, etf, NL, FD, LD, EPE)
            bbs.append(bb.cpu())
            client_exps.append({c: exp.cpu() for c, exp in exps.items()})
            torch.cuda.empty_cache()
        train_time = time.time() - t0
        gpu_peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        if not args.no_save: save_models(bbs, client_exps, ccc, save_dir)

    # 评估
    ed = etf.to(device)
    N = 10000
    all_raw = []; all_labels = []
    errors = torch.full((NC, N, NL), float('inf'))

    for k in range(NC):
        if bbs[k] is None: all_raw.append(None); continue
        bbs[k] = bbs[k].to(device); bbs[k].eval()
        for c in client_exps[k]:
            client_exps[k][c] = client_exps[k][c].to(device)
        feats = []; offset = 0
        with torch.no_grad():
            for x, y in tl:
                x = x.to(device); bs = x.size(0)
                xx = F.relu(bbs[k].bn1(bbs[k].conv1(x)))
                xx = bbs[k].layer1(xx); xx = bbs[k].layer2(xx)
                xx = bbs[k].layer3(xx); xx = bbs[k].layer4(xx)
                xx = bbs[k].pool(xx).flatten(1)
                feat = bbs[k].fc(xx)
                feats.append(feat.cpu())
                f_norm = F.normalize(feat, dim=1)
                for c, exp in client_exps[k].items():
                    fr, _ = exp(f_norm, ed[c].unsqueeze(0).expand(bs, -1))
                    errors[k, offset:offset+bs, c] = ((f_norm - fr)**2).mean(1).cpu()
                if k == 0: all_labels.append(y)
                offset += bs
        all_raw.append(torch.cat(feats, 0))
        bbs[k] = bbs[k].cpu()
        for c in client_exps[k]: client_exps[k][c] = client_exps[k][c].cpu()
        torch.cuda.empty_cache()
        if (k+1) % 10 == 0: print(f"    eval {k+1}/{NC}")
    labels = torch.cat(all_labels).numpy()

    sample_count = {}
    for k in range(NC):
        for c in client_exps[k]:
            sample_count[(k, c)] = ccc.get(k, {}).get(c, 0)

    # znorm + sqrt(n)
    feat = torch.zeros(N, FD); w_sum = 0
    for k in range(NC):
        if all_raw[k] is None: continue
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
    for a in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        acc_full[a] = (cross_client_per_client_logits(data, alpha=a, min_n=10) == labels).mean()
    best_a = max(acc_full, key=acc_full.get)
    best_full = acc_full[best_a]

    # 动态 αf = 0.2 * avg_coverage
    coverage = np.zeros(NL)
    for c in range(NL):
        coverage[c] = sum(1 for k in range(NC) if ccc.get(k, {}).get(c, 0) >= 10) / NC
    af_dynamic = 0.2 * coverage.mean()
    acc_dynamic = (cross_client_per_client_logits(data, alpha=af_dynamic, min_n=10) == labels).mean()

    print(f"\n{'='*60}")
    print(f"  Results: α={ALPHA}, K={NC}")
    print(f"{'='*60}")
    print(f"  znorm+sqrt(n) union:  {acc_union*100:.2f}%")
    print(f"  Expert only:          {acc_expert*100:.2f}%")
    for a in [0.3, 0.5, 1.0]:
        print(f"  +Expert(a={a}):        {acc_full[a]*100:.2f}%")
    print(f"  Best:                 {best_full*100:.2f}% (a={best_a})")

    os.makedirs('results', exist_ok=True)
    out = {
        'method': 'znorm_sqrt', 'alpha': ALPHA, 'n_clients': NC, 'seed': SEED,
        'train_time': train_time,
        'gpu_peak_mb': round(gpu_peak_mb, 1),
        'acc_union': float(acc_union),
        'acc_expert': float(acc_expert),
        'acc_full': {str(k): float(v) for k, v in acc_full.items()},
        'best_alpha': float(best_a),
        'best_acc': float(best_full),
        'acc_dynamic': float(acc_dynamic),
        'af_dynamic': round(float(af_dynamic), 4),
        'avg_coverage': round(float(coverage.mean()), 4),
    }
    path = f"results/znorm_scale_a{ALPHA}_k{NC}_s{SEED}.json"
    json.dump(out, open(path, 'w'), indent=2)
    print(f"  Saved: {path}")

if __name__ == '__main__':
    main()
