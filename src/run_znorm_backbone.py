"""
run_znorm_backbone.py - Backbone 对比实验 (znorm 版)
====================================================
Backbone:
  cnn2:     2-layer CNN  (~73K params)
  cnn3:     3-layer CNN  (~270K params)
  cnn4:     4-layer CNN  (~1M params)
  cnn5:     5-layer CNN  (~2.5M params)
  resnet18: ResNet-18    (~11M params, 主表已有)

用法:
  python run_znorm_backbone.py --backbone cnn2 --alpha 0.05 --seed 42
"""
import sys
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json, time, os, argparse
from tqdm import tqdm

import rebuild8
rebuild8.DL_KWARGS = dict(num_workers=4, pin_memory=True, persistent_workers=False)

from rebuild8 import (prepare_data, generate_etf, ConditionalExpert,
                       device, USE_BF16,
                       cross_client_per_client_logits, expert_original,
                       etf_cl, etf_al, train_exp, preextract)
from resnet18_filter_merge import ResNet18Backbone


class GenericCNN(nn.Module):
    """通用 CNN backbone: n_blocks × (Conv3x3-BN-ReLU-MaxPool) + AdaptivePool + FC
    n_blocks=2: [64, 128]            ~73K params
    n_blocks=3: [64, 128, 256]       ~270K params
    n_blocks=4: [64, 128, 256, 512]  ~1M params
    n_blocks=5: [64, 128, 256, 512, 512] ~2.5M params
    """
    def __init__(self, n_blocks, fd=256, base_ch=64):
        super().__init__()
        channels = [base_ch * (2 ** min(i, 3)) for i in range(n_blocks)]
        layers = []
        in_c = 3
        for out_c in channels:
            layers += [
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(True),
                nn.MaxPool2d(2),
            ]
            in_c = out_c
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], fd)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return F.normalize(self.fc(x), dim=1)

    def extract_raw(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


BACKBONE_MAP = {
    'cnn2': lambda fd: GenericCNN(2, fd),
    'cnn3': lambda fd: GenericCNN(3, fd),
    'cnn4': lambda fd: GenericCNN(4, fd),
    'cnn5': lambda fd: GenericCNN(5, fd),
    'resnet18': lambda fd: ResNet18Backbone(fd),
}

def get_backbone(name, fd=256):
    if name not in BACKBONE_MAP:
        raise ValueError(f"Unknown backbone: {name}. Choose from {list(BACKBONE_MAP.keys())}")
    return BACKBONE_MAP[name](fd)

def extract_raw_features(bb, x, bb_type):
    if bb_type == 'resnet18':
        xx = F.relu(bb.bn1(bb.conv1(x)))
        xx = bb.layer1(xx); xx = bb.layer2(xx)
        xx = bb.layer3(xx); xx = bb.layer4(xx)
        xx = bb.pool(xx).flatten(1)
        return bb.fc(xx)
    else:
        return bb.extract_raw(x)

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def train_bb_backbone(bb, loader, classes, etf, epochs=600, lr=1e-3):
    bb = bb.to(device); ed = etf.to(device); ncl = len(classes); bb.train()
    opt = torch.optim.Adam(bb.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    amp = (torch.amp.autocast('cuda', dtype=torch.bfloat16) if USE_BF16
           else torch.amp.autocast('cuda', enabled=False))
    pbar = tqdm(range(epochs), desc=f"  BB", ncols=100, mininterval=10.0, file=sys.stdout)
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
    pbar.close()
    return bb

def train_experts_backbone(bb, cls_loaders, classes, etf, nc=10, fdim=256, ldim=32, epochs=600, lr=1e-3):
    bb.eval(); ed = etf.to(device); exps = {}
    om = {c: torch.tensor([k for k in range(nc) if k != c], device=device) for c in range(nc)}
    for cls in classes:
        print(f"      Exp {cls}...", end=" ", flush=True); t0 = time.time()
        cached = preextract(bb, cls_loaders[cls])
        exp = ConditionalExpert(fdim, fdim, 128, ldim).to(device)
        exp = train_exp(exp, cached, ed[cls], ed, om[cls], fdim, epochs, lr)
        exps[cls] = exp
        with torch.no_grad():
            ne = min(256, cached.size(0))
            fr, _ = exp(cached[:ne], ed[cls].unsqueeze(0).expand(ne, -1))
            print(f"done ({time.time()-t0:.1f}s) MSE={((cached[:ne]-fr)**2).mean().item():.6f}")
    return exps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, required=True,
                        choices=['cnn2', 'cnn3', 'cnn4', 'cnn5', 'resnet18'])
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    ALPHA = args.alpha; SEED = args.seed; BB_TYPE = args.backbone
    NC = 5; NL = 10; FD = 256; LD = 32; EPB = 600; EPE = 600
    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    sample_bb = get_backbone(BB_TYPE, FD)
    n_params = count_params(sample_bb)
    del sample_bb

    print(f"\n{'='*60}")
    print(f"  Backbone: {BB_TYPE} ({n_params:,} params) | a={ALPHA} | K={NC} | seed={SEED}")
    print(f"{'='*60}")

    etf = generate_etf(NL, FD)
    cal, ccl, tl, ccc = prepare_data(NC, ALPHA, NL)

    bbs = []; client_exps = []
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for k in range(NC):
        cls = sorted(ccc.get(k, {}).keys())
        if not cls:
            bbs.append(None); client_exps.append({}); continue
        print(f"  Client {k}: {len(cls)} cls, {sum(ccc[k].values())} samp")
        bb = get_backbone(BB_TYPE, FD)
        bb = train_bb_backbone(bb, cal[k], cls, etf, EPB)
        exps = train_experts_backbone(bb, ccl[k], cls, etf, NL, FD, LD, EPE)
        bbs.append(bb.cpu())
        client_exps.append({c: exp.cpu() for c, exp in exps.items()})
        torch.cuda.empty_cache()
    train_time = time.time() - t0
    gpu_peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    ed = etf.to(device)
    N = 10000
    all_raw = []; all_labels = []
    errors = torch.full((NC, N, NL), float('inf'))
    for k in range(NC):
        if bbs[k] is None: all_raw.append(None); continue
        bbs[k] = bbs[k].to(device); bbs[k].eval()
        for c in client_exps[k]: client_exps[k][c] = client_exps[k][c].to(device)
        feats = []; offset = 0
        with torch.no_grad():
            for x, y in tl:
                x = x.to(device); bs = x.size(0)
                feat = extract_raw_features(bbs[k], x, BB_TYPE)
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
    for a in [0.3, 0.5, 1.0]:
        acc_full[a] = (cross_client_per_client_logits(data, alpha=a, min_n=10) == labels).mean()
    best_a = max(acc_full, key=acc_full.get)
    best_full = acc_full[best_a]

    print(f"\n{'='*60}")
    print(f"  Results: {BB_TYPE} ({n_params:,} params) | a={ALPHA} | K={NC}")
    print(f"{'='*60}")
    print(f"  znorm+sqrt(n) union:  {acc_union*100:.2f}%")
    print(f"  Expert only:          {acc_expert*100:.2f}%")
    for a in [0.3, 0.5, 1.0]:
        print(f"  +Expert(a={a}):        {acc_full[a]*100:.2f}%")
    print(f"  Best:                 {best_full*100:.2f}% (a={best_a})")

    os.makedirs('results', exist_ok=True)
    out = {
        'method': 'znorm_sqrt', 'backbone': BB_TYPE, 'backbone_params': n_params,
        'alpha': ALPHA, 'n_clients': NC, 'seed': SEED,
        'train_time': train_time, 'gpu_peak_mb': round(gpu_peak_mb, 1),
        'acc_union': float(acc_union), 'acc_expert': float(acc_expert),
        'acc_full': {str(k): float(v) for k, v in acc_full.items()},
        'best_alpha': float(best_a), 'best_acc': float(best_full),
    }
    path = f"results/backbone_{BB_TYPE}_a{ALPHA}_k{NC}_s{SEED}.json"
    json.dump(out, open(path, 'w'), indent=2)
    print(f"  Saved: {path}")

if __name__ == '__main__':
    main()
