"""
Masked Ensemble: 每个 client 只对见过的类投票
验证是否等价于 FM 的效果
"""
import torch, torch.nn.functional as F
import numpy as np, json, time, os, argparse
from resnet18_filter_merge import ResNet18Backbone
from rebuild8 import (generate_etf, prepare_data, device, USE_BF16,
                       ConditionalExpert, expert_original,
                       cross_client_per_client_logits)

def load_all_models(save_dir, NC, NL, FD=256, LD=32):
    bbs = []; client_exps = []
    for k in range(NC):
        client_dir = os.path.join(save_dir, f"client_{k}")
        if not os.path.exists(client_dir):
            bbs.append(None); client_exps.append({}); continue
        bb = ResNet18Backbone(FD)
        bb.load_state_dict(torch.load(os.path.join(client_dir, "backbone.pt"),
                                       map_location='cpu', weights_only=True))
        bbs.append(bb)
        exps = {}
        for c in range(NL):
            exp_path = os.path.join(client_dir, f"expert_{c}.pt")
            if os.path.exists(exp_path):
                exp = ConditionalExpert(FD, FD, 128, LD)
                exp.load_state_dict(torch.load(exp_path, map_location='cpu', weights_only=True))
                exps[c] = exp
        client_exps.append(exps)
    ccc = torch.load(os.path.join(save_dir, "ccc.pt"), weights_only=False)
    return bbs, client_exps, ccc

def masked_ensemble(bbs, client_exps, ccc, tl, etf, NC, NL):
    """每个 client 只对自己见过的类贡献 ETF logit"""
    ed = etf.to(device)
    N = 10000
    
    logit_sum = torch.zeros(N, NL)
    vote_count = torch.zeros(N, NL)
    all_labels = []
    
    # 同时收集 expert errors 做对比
    errors = torch.full((NC, N, NL), float('inf'))
    
    for k in range(NC):
        if bbs[k] is None: continue
        seen_classes = [c for c in range(NL) if ccc[k].get(c, 0) > 0]
        if not seen_classes: continue
        
        bbs[k] = bbs[k].to(device); bbs[k].eval()
        for c in client_exps[k]:
            client_exps[k][c] = client_exps[k][c].to(device)
            client_exps[k][c].eval()
        
        offset = 0
        with torch.no_grad():
            for x, y in tl:
                x = x.to(device); bs = x.size(0)
                f = bbs[k](x)
                logits = torch.mm(f, ed.T).cpu()  # (bs, NL)
                
                # 只对见过的类贡献
                for c in seen_classes:
                    logit_sum[offset:offset+bs, c] += logits[:, c]
                    vote_count[offset:offset+bs, c] += 1
                
                # Expert errors
                for c, exp in client_exps[k].items():
                    fr, _ = exp(f, ed[c].unsqueeze(0).expand(bs, -1))
                    errors[k, offset:offset+bs, c] = ((f - fr)**2).mean(1).cpu()
                
                if k == 0: all_labels.append(y)
                offset += bs
        
        bbs[k] = bbs[k].cpu()
        for c in client_exps[k]:
            client_exps[k][c] = client_exps[k][c].cpu()
        torch.cuda.empty_cache()
    
    labels = torch.cat(all_labels).numpy()
    
    R = {}
    
    # 1. Masked ensemble (equal weight)
    safe_count = vote_count.clamp(min=1)
    masked_logits = logit_sum / safe_count
    R['Masked Ensemble'] = (masked_logits.argmax(1).numpy() == labels).mean()
    
    # 2. Masked ensemble weighted by sample count
    logit_sum_w = torch.zeros(N, NL)
    weight_sum = torch.zeros(N, NL)
    for k in range(NC):
        if bbs[k] is None: continue
        seen_classes = [c for c in range(NL) if ccc[k].get(c, 0) > 0]
        if not seen_classes: continue
        
        bbs[k] = bbs[k].to(device); bbs[k].eval()
        offset = 0
        with torch.no_grad():
            for x, y in tl:
                x = x.to(device); bs = x.size(0)
                f = bbs[k](x)
                logits = torch.mm(f, ed.T).cpu()
                for c in seen_classes:
                    w = np.log(ccc[k].get(c, 0) + 1)
                    logit_sum_w[offset:offset+bs, c] += logits[:, c] * w
                    weight_sum[offset:offset+bs, c] += w
                offset += bs
        bbs[k] = bbs[k].cpu(); torch.cuda.empty_cache()
    
    safe_w = weight_sum.clamp(min=1e-8)
    masked_logits_w = logit_sum_w / safe_w
    R['Masked Ensemble (log-n weighted)'] = (masked_logits_w.argmax(1).numpy() == labels).mean()
    
    # 3. Masked ensemble with min_n threshold
    for min_n in [10, 50, 100]:
        ls = torch.zeros(N, NL); vc = torch.zeros(N, NL)
        for k in range(NC):
            if bbs[k] is None: continue
            qual_classes = [c for c in range(NL) if ccc[k].get(c, 0) >= min_n]
            if not qual_classes: continue
            bbs[k] = bbs[k].to(device); bbs[k].eval()
            offset = 0
            with torch.no_grad():
                for x, y in tl:
                    x = x.to(device); bs = x.size(0)
                    f = bbs[k](x)
                    logits = torch.mm(f, ed.T).cpu()
                    for c in qual_classes:
                        ls[offset:offset+bs, c] += logits[:, c]
                        vc[offset:offset+bs, c] += 1
                    offset += bs
            bbs[k] = bbs[k].cpu(); torch.cuda.empty_cache()
        sc = vc.clamp(min=1)
        R[f'Masked Ensemble (n>={min_n})'] = ((ls/sc).argmax(1).numpy() == labels).mean()
    
    # 4. Masked ensemble + expert fusion
    # normalize masked logits, normalize expert logits, combine
    union_logits = masked_logits_w
    um = union_logits.mean(1, keepdim=True); us = union_logits.std(1, keepdim=True) + 1e-8
    un = (union_logits - um) / us
    
    sample_count = {}
    for k in range(NC):
        for c in client_exps[k]:
            sample_count[(k, c)] = ccc[k].get(c, 0)
    
    data = {
        'errors': errors, 'union_logits': union_logits,
        'union_preds': union_logits.argmax(1).numpy(),
        'union_margin': (union_logits.sort(1, descending=True)[0][:, 0] - 
                         union_logits.sort(1, descending=True)[0][:, 1]).numpy(),
        'labels': labels, 'sample_count': sample_count,
        'K': NC, 'N': N, 'nc': NL,
    }
    R['Masked + Expert (C4, a=0.3)'] = (cross_client_per_client_logits(data, alpha=0.3, min_n=10) == labels).mean()
    R['Masked + Expert (C4, a=0.5)'] = (cross_client_per_client_logits(data, alpha=0.5, min_n=10) == labels).mean()
    R['Masked + Expert (C4, a=1.0)'] = (cross_client_per_client_logits(data, alpha=1.0, min_n=10) == labels).mean()
    
    # 5. Client avg (no masking, baseline)
    R['Client avg (no mask)'] = ((logit_sum / NC).argmax(1).numpy() == labels).mean()
    
    # 6. Expert only
    R['Expert only'] = (expert_original(data) == labels).mean()
    
    return R

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--n_clients', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    ALPHA = args.alpha; NC = args.n_clients; SEED = args.seed
    NL = 10; FD = 256; LD = 32
    torch.manual_seed(SEED); np.random.seed(SEED)

    save_dir = f"saved_models/a{ALPHA}_k{NC}_s{SEED}"
    print(f"\n{'='*60}")
    print(f"  Masked Ensemble: alpha={ALPHA}, K={NC}, seed={SEED}")
    print(f"  Loading from: {save_dir}")
    print(f"{'='*60}")

    etf = generate_etf(NL, FD)
    _, _, tl, _ = prepare_data(NC, ALPHA, NL)
    bbs, client_exps, ccc = load_all_models(save_dir, NC, NL, FD, LD)
    print(f"  Loaded {sum(1 for b in bbs if b is not None)} models")

    R = masked_ensemble(bbs, client_exps, ccc, tl, etf, NC, NL)

    print(f"\n{'='*60}")
    print(f"  Results: alpha={ALPHA}, K={NC}")
    print(f"{'='*60}")
    for name, acc in sorted(R.items(), key=lambda x: -x[1]):
        print(f"    {name:>40s}: {acc*100:.2f}%")

    # Save
    os.makedirs('results', exist_ok=True)
    out = {
        'experiment': 'masked_ensemble', 'alpha': ALPHA,
        'n_clients': NC, 'seed': SEED,
        'results': {k: float(v) for k, v in R.items()},
    }
    path = f"results/masked_ensemble_a{ALPHA}_k{NC}_s{SEED}.json"
    json.dump(out, open(path, 'w'), indent=2)
    print(f"\n  Saved: {path}")

if __name__ == '__main__':
    main()
