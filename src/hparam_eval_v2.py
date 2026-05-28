"""
hparam_eval_v2.py - 扫 αf 敏感性
================================
加载 saved_models 里的 backbones+experts(无需重训),
扫描 28 档 αf,记录 acc。

用法:
  python hparam_eval_v2.py --alpha 0.05 --n_clients 10 --seed 42
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json, os, argparse
from rebuild8 import (prepare_data, generate_etf, ConditionalExpert, device,
                       cross_client_per_client_logits, expert_original)
from resnet18_filter_merge import ResNet18Backbone


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
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--n_clients', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    ALPHA = args.alpha; NC = args.n_clients; SEED = args.seed
    NL = 10; FD = 256

    save_dir = f"saved_models/a{ALPHA}_k{NC}_s{SEED}"
    if not os.path.exists(save_dir):
        print(f"saved_models 不存在: {save_dir}")
        return

    print(f"\nHparam scan: α={ALPHA}, K={NC}, seed={SEED}")
    print(f"Loading from {save_dir}...")

    torch.manual_seed(SEED); np.random.seed(SEED)
    etf = generate_etf(NL, FD)

    # 加载 models
    bbs, client_exps, ccc = load_models(save_dir, NC, NL, FD)
    n_loaded = sum(1 for b in bbs if b is not None)
    print(f"  Loaded {n_loaded}/{NC} clients")

    # 构造 test loader
    _, _, tl, _ = prepare_data(NC, ALPHA, NL)

    # forward 一次拿 features (znorm 版本)
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
    labels = torch.cat(all_labels).numpy()

    sample_count = {}
    for k in range(NC):
        for c in client_exps[k]:
            sample_count[(k, c)] = ccc.get(k, {}).get(c, 0)

    # znorm union
    feat = torch.zeros(N, FD); w_sum = 0
    for k in range(NC):
        if all_raw[k] is None: continue
        f = all_raw[k]
        f_z = (f - f.mean(0, keepdim=True)) / (f.std(0, keepdim=True) + 1e-8)
        w = np.sqrt(sum(ccc.get(k, {}).values()))
        feat += f_z * w; w_sum += w
    feat_n = F.normalize(feat / w_sum, dim=1)
    union_logits = feat_n @ etf.T
    sl, _ = union_logits.sort(dim=1, descending=True)

    data = {
        'errors': errors, 'union_logits': union_logits,
        'union_preds': union_logits.argmax(1).numpy(),
        'union_margin': (sl[:, 0] - sl[:, 1]).numpy(),
        'labels': labels, 'sample_count': sample_count,
        'K': NC, 'N': N, 'nc': NL,
    }

    # 扫描 αf — 28 档密网格. 三段:
    #   [0, 0.1):  adaptive α_f 落点区, 11 档 步长 0.01
    #   [0.1, 1):  峰值区, 11 档 0.05–0.1 步
    #   [1, 5]:    下坡区, 6 档稀疏
    alphas = [
        0.0,
        0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
        0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        1.0, 1.25, 1.5, 2.0, 3.0, 5.0,
    ]
    acc_union = float((data['union_preds'] == labels).mean())
    acc_expert = float((expert_original(data) == labels).mean())

    print(f"\n  Acc Union (αf=0)    : {acc_union:.4f}")
    print(f"  Acc Expert (αf=∞)   : {acc_expert:.4f}")
    print(f"  扫描 αf:")

    accs = {'union_only': acc_union, 'expert_only': acc_expert}
    for af in alphas:
        if af == 0:
            accs[str(af)] = acc_union
        else:
            preds = cross_client_per_client_logits(data, alpha=af, min_n=10)
            accs[str(af)] = float((preds == labels).mean())
        print(f"    αf={af:5.2f}: {accs[str(af)]:.4f}")

    # Adaptive α_f (same formula as ablation_components_v2.py / run_znorm_*.py)
    coverage = np.zeros(NL)
    for c in range(NL):
        coverage[c] = sum(1 for k in range(NC)
                         if ccc.get(k, {}).get(c, 0) >= 10) / NC
    af_adaptive = float(0.2 * coverage.mean())
    if af_adaptive > 0:
        preds_adaptive = cross_client_per_client_logits(data, alpha=af_adaptive, min_n=10)
        acc_adaptive = float((preds_adaptive == labels).mean())
    else:
        acc_adaptive = acc_union
    print(f"\n  Adaptive α_f = {af_adaptive:.4f}  (coverage.mean={coverage.mean():.4f})")
    print(f"  Acc @ adaptive = {acc_adaptive:.4f}")

    out = {
        'method': 'hparam_v2',
        'alpha': ALPHA, 'n_clients': NC, 'seed': SEED,
        'min_n': 10,
        'accs': accs,
        'adaptive_af': af_adaptive,
        'adaptive_acc': acc_adaptive,
        'coverage_mean': float(coverage.mean()),
    }
    os.makedirs('results', exist_ok=True)
    path = f"results/hparam_v2_a{ALPHA}_k{NC}_s{SEED}.json"
    json.dump(out, open(path, 'w'), indent=2)
    print(f"\n  Saved: {path}")

if __name__ == '__main__':
    main()
