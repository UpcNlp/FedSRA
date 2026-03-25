"""
onto_fl.py — 伪联邦本体性知识实验
═══════════════════════════════════════════════════════════════
用法: python onto_fl.py --pipeline dir2 --alpha 0.05 --seed 42 --gpu 0

Pipeline:
  ce    : CE监督训练 (关系性baseline)
  dir2  : VICReg 增强不变性 (本体性)
  dir4  : V-REx 因果不变性  (本体性)
  dir24 : VICReg + V-REx    (本体性)

一次运行 = 一个(pipeline, alpha, seed)的完整流程:
  1. Dirichlet分割数据到K个client
  2. 每个client独立训练backbone
  3. 每个client计算本地prototype
  4. 联邦聚合 + 推理 + 输出准确率
═══════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import json, os, time, warnings, argparse
from collections import defaultdict

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════
N_CLIENTS   = 5
N_CLASSES   = 10
FEAT_DIM    = 256
EXPAND_DIM  = 512
BATCH_SIZE  = 512
LR          = 1e-3
EPOCHS_CE   = 300
EPOCHS_SSL  = 400
VREX_LAMBDA = 10.0

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)


# ═══════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════
def dirichlet_split(targets, n_clients, alpha, seed=42):
    rng = np.random.RandomState(seed)
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)
    client_indices = defaultdict(list)
    client_class_counts = defaultdict(dict)
    for c in range(N_CLASSES):
        idxs = np.array(class_indices[c])
        rng.shuffle(idxs)
        props = rng.dirichlet([alpha] * n_clients)
        counts = (props * len(idxs)).astype(int)
        counts[-1] = len(idxs) - counts[:-1].sum()
        start = 0
        for k in range(n_clients):
            end = start + counts[k]
            if end > start:
                client_indices[k].extend(idxs[start:end].tolist())
                client_class_counts[k][c] = int(counts[k])
            start = end
    return dict(client_indices), dict(client_class_counts)

def get_test_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

def get_ce_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
        transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


# ═══════════════════════════════════════════════════════════════
# 增强环境
# ═══════════════════════════════════════════════════════════════
class EnvColor:
    def __init__(self):
        self.aug = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.8,0.8,0.8,0.2)], p=0.9),
            transforms.RandomGrayscale(p=0.3),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1,2.0))], p=0.4),
            transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    def __call__(self, x): return self.aug(x), self.aug(x)

class EnvSpatial:
    def __init__(self):
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4), transforms.RandomRotation(25),
            transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    def __call__(self, x): return self.aug(x), self.aug(x)

class EnvCombined:
    def __init__(self):
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4), transforms.RandomRotation(15),
            transforms.RandomApply([transforms.ColorJitter(0.8,0.8,0.8,0.2)], p=0.9),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    def __call__(self, x): return self.aug(x), self.aug(x)

class MultiEnvTransform:
    def __init__(self):
        self.envs = [EnvColor(), EnvSpatial(), EnvCombined()]
    def __call__(self, x):
        return tuple(env(x) for env in self.envs)


# ═══════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(256,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(256*2*2, FEAT_DIM)
    def forward(self, x):
        return self.fc(self.features(x).view(x.size(0), -1))

class Expander(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEAT_DIM, EXPAND_DIM), nn.BatchNorm1d(EXPAND_DIM), nn.ReLU(True),
            nn.Linear(EXPAND_DIM, EXPAND_DIM), nn.BatchNorm1d(EXPAND_DIM), nn.ReLU(True),
            nn.Linear(EXPAND_DIM, EXPAND_DIM),
        )
    def forward(self, x): return self.net(x)


# ═══════════════════════════════════════════════════════════════
# Losses
# ═══════════════════════════════════════════════════════════════
def vicreg_loss(z1, z2, sim_w=25.0, std_w=25.0, cov_w=1.0):
    N, D = z1.shape
    sim = F.mse_loss(z1, z2)
    std = F.relu(1-torch.sqrt(z1.var(0)+1e-4)).mean() + F.relu(1-torch.sqrt(z2.var(0)+1e-4)).mean()
    z1c, z2c = z1-z1.mean(0), z2-z2.mean(0)
    c1, c2 = (z1c.T@z1c)/(N-1), (z2c.T@z2c)/(N-1)
    off = lambda c: c.pow(2).sum() - c.diagonal().pow(2).sum()
    cov = (off(c1)+off(c2)) / D
    return sim_w*sim + std_w*std + cov_w*cov

def vrex_penalty(losses):
    s = torch.stack(losses)
    return ((s-s.mean())**2).mean()


# ═══════════════════════════════════════════════════════════════
# Client Training
# ═══════════════════════════════════════════════════════════════
def train_client_ce(indices, device):
    ds = datasets.CIFAR10('./data', train=True, transform=get_ce_transform())
    loader = DataLoader(Subset(ds, indices), batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=len(indices)>BATCH_SIZE, num_workers=16, pin_memory=True,persistent_workers=True)
    bb = Backbone().to(device)
    head = nn.Linear(FEAT_DIM, N_CLASSES).to(device)
    opt = torch.optim.Adam(list(bb.parameters())+list(head.parameters()), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_CE)
    bb.train(); head.train()
    from tqdm import tqdm
    epoch_bar = tqdm(range(EPOCHS_CE), desc='Epochs', position=0)
    for ep in epoch_bar:
        loss_ = []
        for x, y in tqdm(loader, desc=f'Ep{ep+1}', position=1, leave=False):
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(head(bb(x)), y)
            loss_.append(loss.item())
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sch.step()
        avg_loss = sum(loss_) / len(loss_)
        epoch_bar.set_postfix(loss=f'{avg_loss:.4f}')
    return bb, head


def train_client_onto(pipeline, indices, device):
    ds = datasets.CIFAR10('./data', train=True, transform=MultiEnvTransform())
    loader = DataLoader(Subset(ds, indices), batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=len(indices)>BATCH_SIZE, num_workers=16, pin_memory=True,persistent_workers=True)
    bb = Backbone().to(device)
    exp = Expander().to(device)
    opt = torch.optim.Adam(list(bb.parameters())+list(exp.parameters()), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_SSL)

    use_dir2 = pipeline in ('dir2', 'dir24')
    use_dir4 = pipeline in ('dir4', 'dir24')

    bb.train(); exp.train()
    
    from tqdm import tqdm
    epoch_bar = tqdm(range(EPOCHS_CE), desc='Epochs', position=0)
    for ep in epoch_bar:
        loss_ = []
        for views, _ in tqdm(loader, desc=f'Ep{ep+1}', position=1, leave=False):
            opt.zero_grad(set_to_none=True)
            loss = torch.tensor(0.0, device=device)

            if use_dir2:
                v1, v2 = views[2]
                loss = loss + vicreg_loss(exp(bb(v1.to(device))), exp(bb(v2.to(device))))

            if use_dir4:
                env_range = range(2) if use_dir2 else range(3)
                el = []
                for ei in env_range:
                    v1e, v2e = views[ei]
                    el.append(F.mse_loss(exp(bb(v1e.to(device))), exp(bb(v2e.to(device)))))
                loss = loss + torch.stack(el).mean() + VREX_LAMBDA * vrex_penalty(el)

            loss.backward(); opt.step()
            
            loss_.append(loss.item())
        sch.step()
        avg_loss = sum(loss_) / len(loss_)
        epoch_bar.set_postfix(loss=f'{avg_loss:.4f}')

    return bb


@torch.no_grad()
def compute_prototypes(bb, indices, class_counts, device):
    ds = datasets.CIFAR10('./data', train=True, transform=get_test_transform())
    targets = np.array(ds.targets)
    bb.eval()
    prototypes = {}
    for c in class_counts.keys():
        cidx = [i for i in indices if targets[i] == c]
        if not cidx: continue
        loader = DataLoader(Subset(ds, cidx), batch_size=256, shuffle=False, num_workers=16,persistent_workers=True)
        feats = torch.cat([bb(x.to(device)).cpu() for x, _ in loader], 0)
        prototypes[c] = F.normalize(feats.mean(0), dim=0)
    return prototypes


# ═══════════════════════════════════════════════════════════════
# Federated Eval
# ═══════════════════════════════════════════════════════════════
@torch.no_grad()
def fed_eval_ce(models, ccs, test_loader, device):
    all_logits = None; test_labels = None; tw = 0
    for (bb, head), cc in zip(models, ccs):
        bb.to(device).eval(); head.to(device).eval()
        lg, lb = [], []
        for x, y in test_loader:
            lg.append(head(bb(x.to(device))).cpu()); lb.append(y)
        lk = torch.cat(lg, 0)
        if test_labels is None:
            test_labels = torch.cat(lb, 0); all_logits = torch.zeros_like(lk)
        w = sum(cc.values())
        if w > 0: all_logits += F.softmax(lk, dim=1)*w; tw += w
        bb.cpu(); head.cpu(); torch.cuda.empty_cache()
    if tw > 0: all_logits /= tw
    return float((all_logits.argmax(1).numpy() == test_labels.numpy()).mean())


@torch.no_grad()
def fed_eval_proto(bbs, protos, ccs, test_loader, device):
    N = 10000
    scores = torch.zeros(N, N_CLASSES); weights = torch.zeros(N_CLASSES); tl = None
    for bb, pr, cc in zip(bbs, protos, ccs):
        if not pr: continue
        bb.to(device).eval()
        fl, ll = [], []
        for x, y in test_loader:
            fl.append(bb(x.to(device)).cpu()); ll.append(y)
        tf = F.normalize(torch.cat(fl, 0), dim=1)
        if tl is None: tl = torch.cat(ll, 0)
        for c, p in pr.items():
            sim = torch.mm(tf, F.normalize(p.unsqueeze(0), dim=1).T).squeeze(1)
            w = float(cc.get(c, 0))
            if w > 0: scores[:,c] += sim*w; weights[c] += w
        bb.cpu(); torch.cuda.empty_cache()
    for c in range(N_CLASSES):
        if weights[c] > 0: scores[:,c] /= weights[c]
        else: scores[:,c] = -float('inf')
    return float((scores.argmax(1).numpy() == tl.numpy()).mean())


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pipeline', required=True, choices=['ce','dir2','dir4','dir24'])
    p.add_argument('--alpha', required=True, type=float)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print(f'\n{"="*60}')
    print(f'  {args.pipeline}  alpha={args.alpha}  seed={args.seed}')
    print(f'{"="*60}')

    # 数据分割
    train_ds = datasets.CIFAR10('./data', train=True, download=True)
    ci, cc = dirichlet_split(train_ds.targets, N_CLIENTS, args.alpha, seed=args.seed)

    for k in range(N_CLIENTS):
        c_cc = cc.get(k, {})
        top = sorted(c_cc.items(), key=lambda x:-x[1])[:3]
        print(f'  Client {k}: {len(c_cc)}cls {sum(c_cc.values())}smp  top: {", ".join(f"c{c}={n}" for c,n in top)}')

    test_ds = datasets.CIFAR10('./data', train=False, transform=get_test_transform())
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True,persistent_workers=True)

    # 训练
    t0 = time.time()
    if args.pipeline == 'ce':
        models, ccs = [], []
        for k in range(N_CLIENTS):
            print(f'\n  Client {k} (CE):')
            sk = args.seed + hash(('ce', args.alpha, k)) % 100000
            torch.manual_seed(sk); np.random.seed(sk%(2**31))
            bb, head = train_client_ce(ci[k], device)
            models.append((bb.cpu(), head.cpu())); ccs.append(cc.get(k,{}))
            torch.cuda.empty_cache()
        print(f'\n  Train: {time.time()-t0:.0f}s. Eval...')
        acc = fed_eval_ce(models, ccs, test_loader, device)
    else:
        bbs, prs, ccs = [], [], []
        for k in range(N_CLIENTS):
            print(f'\n  Client {k} ({args.pipeline}):')
            sk = args.seed + hash((args.pipeline, args.alpha, k)) % 100000
            torch.manual_seed(sk); np.random.seed(sk%(2**31))
            bb = train_client_onto(args.pipeline, ci[k], device)
            pr = compute_prototypes(bb, ci[k], cc.get(k,{}), device)
            bbs.append(bb.cpu()); prs.append(pr); ccs.append(cc.get(k,{}))
            torch.cuda.empty_cache()
        print(f'\n  Train: {time.time()-t0:.0f}s. Eval...')
        acc = fed_eval_proto(bbs, prs, ccs, test_loader, device)

    print(f'\n  ★ {args.pipeline} a={args.alpha} s={args.seed}: {acc*100:.1f}%')
    print(f'  Time: {time.time()-t0:.0f}s')

    os.makedirs('results', exist_ok=True)
    out = f'results/{args.pipeline}_a{args.alpha}_s{args.seed}.json'
    with open(out, 'w') as f:
        json.dump({'pipeline':args.pipeline, 'alpha':args.alpha, 'seed':args.seed, 'acc':acc}, f, indent=2)
    print(f'  Saved: {out}\n')

if __name__ == '__main__':
    main()