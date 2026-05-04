"""
模型异构实验: 5 个 client 用不同 backbone
验证: intrinsic signal (expert) 不受架构限制
      relational signal (filter merge) 无法跨架构 → 只能用 client avg
"""
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np, json, time, os, argparse
from collections import defaultdict

from rebuild8 import (generate_etf, train_bb, train_experts, compute_stats,
                       precompute_all, device, USE_BF16, DL_KWARGS,
                       cross_client_per_client_logits, expert_original,
                       route_ensemble_logits, d3_softmax_ensemble)

# ═══════════════════════════════════════════
# 5 种不同的 Backbone
# ═══════════════════════════════════════════

class SmallCNN(nn.Module):
    """LeNet-style small CNN"""
    def __init__(self, fd=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(128 * 4 * 4, fd)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return F.normalize(self.fc(x), dim=1)


class MediumCNN(nn.Module):
    """VGG-style medium CNN"""
    def __init__(self, fd=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(256 * 4 * 4, fd)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return F.normalize(self.fc(x), dim=1)


class WideCNN(nn.Module):
    """Wide but shallow CNN"""
    def __init__(self, fd=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(512 * 4 * 4, fd)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return F.normalize(self.fc(x), dim=1)


from rebuild8 import Backbone  # 原始 4-layer CNN
from resnet18_filter_merge import ResNet18Backbone


ARCH_MAP = {
    0: ('SmallCNN', SmallCNN),
    1: ('MediumCNN', MediumCNN),
    2: ('Backbone', Backbone),      # 原始 CNN
    3: ('WideCNN', WideCNN),
    4: ('ResNet18', ResNet18Backbone),
}


def dirichlet_split(dataset, n_clients, alpha, n_classes=10):
    targets = np.array(dataset.targets)
    ci = defaultdict(list)
    for idx, l in enumerate(targets): ci[l].append(idx)
    client_idx = defaultdict(list)
    client_cc = defaultdict(lambda: defaultdict(int))
    for c in range(n_classes):
        idxs = np.array(ci[c]); np.random.shuffle(idxs)
        props = np.random.dirichlet([alpha]*n_clients)
        props = (props*len(idxs)).astype(int); props[-1] = len(idxs)-props[:-1].sum()
        s = 0
        for k in range(n_clients):
            e = s+props[k]
            if e > s:
                client_idx[k].extend(idxs[s:e].tolist())
                client_cc[k][c] = props[k]
            s = e
    return dict(client_idx), dict(client_cc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    ALPHA = args.alpha; SEED = args.seed
    NC = 5; NL = 10; FD = 256; LD = 32; EPB = 600; EPE = 600
    torch.manual_seed(SEED); np.random.seed(SEED)

    print(f"\n{'='*60}")
    print(f"  模型异构实验: α={ALPHA}, seed={SEED}")
    print(f"  架构: {[v[0] for v in ARCH_MAP.values()]}")
    print(f"{'='*60}")

    etf = generate_etf(NL, FD)

    # Data
    tt = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
        transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2), transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
        transforms.RandomErasing(p=0.25, scale=(0.02,0.2)),
    ])
    te = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=tt)
    test_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=te)
    cidx, ccc = dirichlet_split(train_ds, NC, ALPHA, NL)
    tl = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    targets = np.array(train_ds.targets)
    cal = {}
    for k in range(NC):
        cal[k] = DataLoader(Subset(train_ds, cidx[k]), batch_size=128, shuffle=True,
                           drop_last=True, num_workers=4, pin_memory=True)
    ccl = {}
    for k in range(NC):
        ccl[k] = {}
        cm = defaultdict(list)
        for idx in cidx[k]: cm[targets[idx]].append(idx)
        for c, idxs in cm.items():
            ccl[k][c] = DataLoader(Subset(train_ds, idxs), batch_size=64, shuffle=True,
                                   num_workers=2, pin_memory=True)

    # 训练: 每个 client 用不同架构
    bbs = []; client_exps = []
    t0 = time.time()
    for k in range(NC):
        arch_name, arch_cls = ARCH_MAP[k]
        cls = sorted(ccc[k].keys())
        print(f"\n  Client {k} [{arch_name}]: {len(cls)} cls, {sum(ccc[k].values())} samp")
        bb = arch_cls(FD)
        bb = train_bb(bb, cal[k], cls, etf, EPB)
        exps = train_experts(bb, ccl[k], cls, etf, NL, FD, LD, EPE)
        for c in cls:
            mu, _ = compute_stats(bb, exps[c], ccl[k][c], etf[c])
            print(f"    c{c}: n={ccc[k][c]:5d} μ={mu:.6f}")
        bbs.append(bb.cpu())
        client_exps.append({c: exp.cpu() for c, exp in exps.items()})
        torch.cuda.empty_cache()
    train_time = time.time() - t0

    # 因为架构不同, 不能做 filter merge
    # 用 client avg 的 ETF logits 作为 union
    print(f"\n  Computing ETF client avg (no filter merge)...")
    etf_d = etf.to(device)
    N = len(test_ds)
    union_logits = torch.zeros(N, NL)
    all_labels = []
    for k in range(NC):
        bbs[k] = bbs[k].to(device); bbs[k].eval()
        offset = 0
        with torch.no_grad():
            for x, y in tl:
                x = x.to(device); bs = x.size(0)
                f = bbs[k](x)
                union_logits[offset:offset+bs] += torch.mm(f, etf_d.T).cpu()
                offset += bs
                if k == 0: all_labels.append(y)
        bbs[k] = bbs[k].cpu(); torch.cuda.empty_cache()
    labels = torch.cat(all_labels).numpy()

    # Expert errors
    errors = torch.full((NC, N, NL), float('inf'))
    with torch.no_grad():
        for k in range(NC):
            bbs[k] = bbs[k].to(device); bbs[k].eval()
            for c in client_exps[k]:
                client_exps[k][c] = client_exps[k][c].to(device)
                client_exps[k][c].eval()
            offset = 0
            for x, y in tl:
                x = x.to(device); bs = x.size(0)
                f = bbs[k](x)
                for c, exp in client_exps[k].items():
                    fr, _ = exp(f, etf_d[c].unsqueeze(0).expand(bs, -1))
                    errors[k, offset:offset+bs, c] = ((f - fr)**2).mean(1).cpu()
                offset += bs
            bbs[k] = bbs[k].cpu()
            for c in client_exps[k]:
                client_exps[k][c] = client_exps[k][c].cpu()
            torch.cuda.empty_cache()

    # Build data dict
    union_preds = union_logits.argmax(1).numpy()
    sorted_ul, _ = union_logits.sort(dim=1, descending=True)
    union_margin = (sorted_ul[:, 0] - sorted_ul[:, 1]).numpy()
    sample_count = {}
    for k in range(NC):
        for c in client_exps[k]:
            sample_count[(k, c)] = ccc[k].get(c, 0)

    data = {
        'errors': errors, 'union_logits': union_logits,
        'union_preds': union_preds, 'union_margin': union_margin,
        'labels': labels, 'sample_count': sample_count,
        'K': NC, 'N': N, 'nc': NL,
    }

    # === Results ===
    R = {}
    R['A: ETF only (client avg)'] = (union_preds == labels).mean()
    R['B: Expert only'] = (expert_original(data) == labels).mean()
    R['C: ETF + Expert ensemble'] = (cross_client_per_client_logits(data, alpha=0.3, min_n=10) == labels).mean()

    # Also ensemble baseline (CE heads)
    print(f"\n  Training CE ensemble baseline...")
    head_logits = torch.zeros(N, NL)
    for k in range(NC):
        head = nn.Linear(FD, NL).to(device)
        bb_k = bbs[k].to(device); bb_k.eval()
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        dl = cal[k]
        for ep in range(50):
            head.train()
            for x, y in dl:
                x, y = x.to(device), y.to(device)
                with torch.no_grad(): feat = bb_k(x)
                loss = F.cross_entropy(head(feat), y)
                opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        offset = 0
        with torch.no_grad():
            for x, y in tl:
                x = x.to(device); bs = x.size(0)
                feat = bb_k(x)
                head_logits[offset:offset+bs] += head(feat).cpu()
                offset += bs
        bbs[k] = bbs[k].cpu(); torch.cuda.empty_cache()
    R['D: CE Ensemble'] = (head_logits.argmax(1).numpy() == labels).mean()

    # Print
    print(f"\n{'='*60}")
    print(f"  模型异构结果: α={ALPHA}, seed={SEED}")
    print(f"  架构: {[ARCH_MAP[k][0] for k in range(NC)]}")
    print(f"{'='*60}")
    for name, acc in R.items():
        print(f"  {name:>35s}: {acc*100:>6.2f}%")

    # Save
    os.makedirs('results', exist_ok=True)
    out = {
        'experiment': 'model_heterogeneity',
        'alpha': ALPHA, 'seed': SEED, 'n_clients': NC,
        'architectures': [ARCH_MAP[k][0] for k in range(NC)],
        'train_time': train_time, 'results': R
    }
    path = f"results/model_hetero_a{ALPHA}_s{SEED}.json"
    json.dump(out, open(path, 'w'), indent=2)
    print(f"  Saved: {path}")

if __name__ == '__main__':
    main()
