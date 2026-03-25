"""
onto_fl.py — 伪联邦本体性知识实验 (自动多卡并行)
═══════════════════════════════════════════════════════════════
用法: python onto_fl.py --pipeline dir2 --alpha 0.05 --seed 42

自动检测GPU数量, 每张卡跑1个client, 并行训练。
--max-parallel 控制同时跑几个 (默认=GPU数量)
═══════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
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
# 单client训练 (子进程入口)
# ═══════════════════════════════════════════════════════════════
def train_one_client(args_tuple):
    pipeline, client_id, indices, class_counts, seed, gpu_id = args_tuple
    device = torch.device(f'cuda:{gpu_id}')
    sk = seed + hash((pipeline, client_id)) % 100000
    torch.manual_seed(sk); np.random.seed(sk % (2**31))

    epochs = EPOCHS_CE if pipeline == 'ce' else EPOCHS_SSL
    t0 = time.time()

    if pipeline == 'ce':
        ds = datasets.CIFAR10('./data', train=True, transform=get_ce_transform())
        loader = DataLoader(Subset(ds, indices), batch_size=BATCH_SIZE, shuffle=True,
                            drop_last=len(indices)>BATCH_SIZE, num_workers=4,
                            pin_memory=True, persistent_workers=True)
        bb = Backbone().to(device)
        head = nn.Linear(FEAT_DIM, N_CLASSES).to(device)
        opt = torch.optim.Adam(list(bb.parameters())+list(head.parameters()), lr=LR)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        bb.train(); head.train()
        epoch_losses = []
        for ep in range(epochs):
            ep_loss = []
            for x, y in loader:
                loss = F.cross_entropy(head(bb(x.to(device))), y.to(device))
                ep_loss.append(loss.item())
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            sch.step()
            avg = sum(ep_loss)/len(ep_loss)
            epoch_losses.append(avg)
            if (ep+1) % 50 == 0:
                print(f'    C{client_id}[G{gpu_id}] ep{ep+1}/{epochs} loss={avg:.4f}')
        protos = _compute_protos(bb, indices, class_counts, device)
        result = {'backbone': bb.cpu().state_dict(), 'head': head.cpu().state_dict(),
                  'protos': protos, 'class_counts': class_counts,
                  'final_loss': epoch_losses[-1], 'loss_curve': epoch_losses}

    else:
        ds = datasets.CIFAR10('./data', train=True, transform=MultiEnvTransform())
        loader = DataLoader(Subset(ds, indices), batch_size=BATCH_SIZE, shuffle=True,
                            drop_last=len(indices)>BATCH_SIZE, num_workers=4,
                            pin_memory=True, persistent_workers=True)
        bb = Backbone().to(device)
        exp = Expander().to(device)
        opt = torch.optim.Adam(list(bb.parameters())+list(exp.parameters()), lr=LR)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        use2 = pipeline in ('dir2','dir24')
        use4 = pipeline in ('dir4','dir24')
        bb.train(); exp.train()
        epoch_losses = []
        for ep in range(epochs):
            ep_loss = []
            for views, _ in loader:
                opt.zero_grad(set_to_none=True)
                loss = torch.tensor(0.0, device=device)
                if use2:
                    v1, v2 = views[2]
                    loss = loss + vicreg_loss(exp(bb(v1.to(device))), exp(bb(v2.to(device))))
                if use4:
                    er = range(2) if use2 else range(3)
                    el = []
                    for ei in er:
                        v1e, v2e = views[ei]
                        el.append(F.mse_loss(exp(bb(v1e.to(device))), exp(bb(v2e.to(device)))))
                    loss = loss + torch.stack(el).mean() + VREX_LAMBDA * vrex_penalty(el)
                loss.backward(); opt.step()
                ep_loss.append(loss.item())
            sch.step()
            avg = sum(ep_loss)/len(ep_loss)
            epoch_losses.append(avg)
            if (ep+1) % 50 == 0:
                print(f'    C{client_id}[G{gpu_id}] ep{ep+1}/{epochs} loss={avg:.4f}')
        protos = _compute_protos(bb, indices, class_counts, device)
        result = {'backbone': bb.cpu().state_dict(), 'protos': protos, 'class_counts': class_counts,
                  'final_loss': epoch_losses[-1], 'loss_curve': epoch_losses}

    del loader; torch.cuda.empty_cache()
    print(f'  ✓ Client {client_id} [GPU{gpu_id}] {time.time()-t0:.0f}s  final_loss={result["final_loss"]:.4f}')
    return client_id, result


def _compute_protos(bb, indices, class_counts, device):
    ds = datasets.CIFAR10('./data', train=True, transform=get_test_transform())
    targets = np.array(ds.targets)
    bb.eval()
    protos = {}
    for c in class_counts.keys():
        cidx = [i for i in indices if targets[i] == c]
        if not cidx: continue
        loader = DataLoader(Subset(ds, cidx), batch_size=256, shuffle=False, num_workers=0)
        feats = torch.cat([bb(x.to(device)).cpu() for x, _ in loader], 0)
        protos[c] = F.normalize(feats.mean(0), dim=0)
    return protos


# ═══════════════════════════════════════════════════════════════
# Federated Eval
# ═══════════════════════════════════════════════════════════════
@torch.no_grad()
def fed_eval_ce(results, test_loader, device):
    all_logits = None; tl = None; tw = 0
    for cr in results:
        bb = Backbone(); bb.load_state_dict(cr['backbone'])
        head = nn.Linear(FEAT_DIM, N_CLASSES); head.load_state_dict(cr['head'])
        bb.to(device).eval(); head.to(device).eval()
        lg, lb = [], []
        for x, y in test_loader:
            lg.append(head(bb(x.to(device))).cpu()); lb.append(y)
        lk = torch.cat(lg, 0)
        if tl is None: tl = torch.cat(lb, 0); all_logits = torch.zeros_like(lk)
        w = sum(cr['class_counts'].values())
        if w > 0: all_logits += F.softmax(lk, dim=1)*w; tw += w
        bb.cpu(); head.cpu(); torch.cuda.empty_cache()
    if tw > 0: all_logits /= tw
    return float((all_logits.argmax(1).numpy() == tl.numpy()).mean())


@torch.no_grad()
def fed_eval_proto(results, test_loader, device):
    N = 10000
    scores = torch.zeros(N, N_CLASSES); weights = torch.zeros(N_CLASSES); tl = None
    for cr in results:
        pr = cr['protos']; cc = cr['class_counts']
        if not pr: continue
        bb = Backbone(); bb.load_state_dict(cr['backbone'])
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
    p.add_argument('--max-parallel', type=int, default=0,
                   help='最大并行数, 0=自动(GPU数), 1=串行')
    args = p.parse_args()

    n_gpus = torch.cuda.device_count()
    max_par = args.max_parallel if args.max_parallel > 0 else n_gpus
    max_par = min(max_par, N_CLIENTS)

    print(f'\n{"="*60}')
    print(f'  {args.pipeline}  alpha={args.alpha}  seed={args.seed}')
    print(f'  GPUs={n_gpus}  parallel={max_par}')
    print(f'{"="*60}')

    train_ds = datasets.CIFAR10('./data', train=True, download=True)
    ci, cc = dirichlet_split(train_ds.targets, N_CLIENTS, args.alpha, seed=args.seed)
    for k in range(N_CLIENTS):
        c_cc = cc.get(k, {})
        top = sorted(c_cc.items(), key=lambda x:-x[1])[:3]
        print(f'  C{k}: {len(c_cc)}cls {sum(c_cc.values())}smp  '
              f'{", ".join(f"c{c}={n}" for c,n in top)}')

    # 构建任务
    jobs = [(args.pipeline, k, ci[k], cc.get(k,{}), args.seed, k % n_gpus)
            for k in range(N_CLIENTS)]

    # 训练
    t0 = time.time()
    client_results = [None] * N_CLIENTS

    if max_par <= 1:
        for job in jobs:
            cid, res = train_one_client(job)
            client_results[cid] = res
    else:
        ctx = mp.get_context('spawn')
        for i in range(0, N_CLIENTS, max_par):
            batch = jobs[i:i+max_par]
            gpu_str = ','.join(str(j[5]) for j in batch)
            print(f'\n  ── Batch: C{i}-C{i+len(batch)-1} on GPU[{gpu_str}] ──')
            with ctx.Pool(len(batch)) as pool:
                for cid, res in pool.imap_unordered(train_one_client, batch):
                    client_results[cid] = res

    print(f'\n  Train: {time.time()-t0:.0f}s')
    print(f'\n  ── Loss Summary ──')
    for k in range(N_CLIENTS):
        cr = client_results[k]
        n_cls = len(cr['class_counts'])
        n_smp = sum(cr['class_counts'].values())
        fl = cr['final_loss']
        print(f'    Client {k}: {n_cls}cls {n_smp}smp  final_loss={fl:.4f}')
    avg_fl = np.mean([cr['final_loss'] for cr in client_results])
    print(f'    Average final_loss: {avg_fl:.4f}')

    print(f'\n  Eval...')

    test_ds = datasets.CIFAR10('./data', train=False, transform=get_test_transform())
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    if args.pipeline == 'ce':
        acc = fed_eval_ce(client_results, test_loader, torch.device('cuda:0'))
    else:
        acc = fed_eval_proto(client_results, test_loader, torch.device('cuda:0'))

    print(f'\n  ★ {args.pipeline} a={args.alpha} s={args.seed}: {acc*100:.1f}%')
    print(f'  Total: {time.time()-t0:.0f}s')

    os.makedirs('results', exist_ok=True)
    out = f'results/{args.pipeline}_a{args.alpha}_s{args.seed}.json'
    per_client_loss = {f'client_{k}': client_results[k]['final_loss'] for k in range(N_CLIENTS)}
    with open(out, 'w') as f:
        json.dump({'pipeline':args.pipeline, 'alpha':args.alpha, 'seed':args.seed,
                   'acc':acc, 'avg_final_loss': float(avg_fl),
                   'per_client_loss': per_client_loss}, f, indent=2)
    print(f'  Saved: {out}\n')

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()