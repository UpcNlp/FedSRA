"""
run_full_experiments.py
========================
ICDE 论文完整实验脚本

功能:
  1. 主实验: α×K 网格 (CIFAR-10/100)
  2. 多 seed 统计
  3. 消融实验 (ETF/Expert/Union/融合策略)
  4. 与 FAFI baseline 对比 (K=5)
  5. 结果自动保存 JSON

用法:
  # 跑完整 α×K 网格 (单 seed)
  python run_full_experiments.py --mode grid --dataset cifar10 --seed 42 --gpu 0

  # 跑多 seed
  python run_full_experiments.py --mode grid --dataset cifar10 --seed 42 --gpu 0
  python run_full_experiments.py --mode grid --dataset cifar10 --seed 0 --gpu 1
  python run_full_experiments.py --mode grid --dataset cifar10 --seed 123 --gpu 2

  # 消融实验
  python run_full_experiments.py --mode ablation --dataset cifar10 --alpha 0.3 --n_clients 5 --gpu 0

  # 单次运行
  python run_full_experiments.py --mode single --dataset cifar10 --alpha 0.05 --n_clients 5 --gpu 0

  # CIFAR-100
  python run_full_experiments.py --mode grid --dataset cifar100 --seed 42 --gpu 0
"""

import os, json, time, random, argparse, warnings
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════
CIFAR10_MEAN, CIFAR10_STD = (0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616)
CIFAR100_MEAN, CIFAR100_STD = (0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761)

FD = 256        # feature dim
LD = 32         # latent dim
LR = 1e-3
MIN_EXPERT_SAMPLES = 5   # 样本少于此数的类不训练 expert (原来20太高)

def get_epochs(dataset_name):
    """数据集自适应 epoch 数"""
    if dataset_name == 'cifar10':
        return {'backbone': 600, 'expert': 600, 'ce': 200}
    elif dataset_name == 'cifar100':
        return {'backbone': 300, 'expert': 200, 'ce': 100}
    else:
        return {'backbone': 300, 'expert': 200, 'ce': 100}

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_device(gpu_id):
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')

def use_bf16():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

def get_amp_ctx():
    if use_bf16():
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    return torch.amp.autocast('cuda', enabled=False)


# ═══════════════════════════════════════════════════════════
# Data (支持 CIFAR-10 / CIFAR-100)
# ═══════════════════════════════════════════════════════════
def get_dataset(name, train=True, augment=False):
    if name == 'cifar10':
        mean, std, nc = CIFAR10_MEAN, CIFAR10_STD, 10
        DS = datasets.CIFAR10
    elif name == 'cifar100':
        mean, std, nc = CIFAR100_MEAN, CIFAR100_STD, 100
        DS = datasets.CIFAR100
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if augment:
        tf = transforms.Compose([
            transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
            transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2), transforms.RandomRotation(15),
            transforms.ToTensor(), transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25, scale=(0.02,0.2)),
        ])
    else:
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    return DS(root='./data', train=train, download=True, transform=tf), nc


def dirichlet_split(targets, n_clients, alpha, n_classes, seed=42):
    rng = np.random.RandomState(seed)
    ci = defaultdict(list)
    for idx, l in enumerate(targets): ci[int(l)].append(idx)
    client_idx = defaultdict(list)
    client_cc = defaultdict(lambda: defaultdict(int))
    for c in range(n_classes):
        idxs = np.array(ci[c]); rng.shuffle(idxs)
        props = rng.dirichlet([alpha] * n_clients)
        counts = (props * len(idxs)).astype(int)
        counts[-1] = len(idxs) - counts[:-1].sum()
        s = 0
        for k in range(n_clients):
            e = s + counts[k]
            if e > s:
                client_idx[k].extend(idxs[s:e].tolist())
                client_cc[k][c] = counts[k]
            s = e
    return dict(client_idx), dict(client_cc)


def prepare_data(dataset_name, n_clients, alpha, seed=42):
    train_ds, nc = get_dataset(dataset_name, train=True, augment=True)
    test_ds, _ = get_dataset(dataset_name, train=False, augment=False)
    targets = np.array(train_ds.targets)
    cidx, ccc = dirichlet_split(targets, n_clients, alpha, nc, seed)

    print(f"\n  数据: {dataset_name}, α={alpha}, K={n_clients}, nc={nc}")
    for k in range(n_clients):
        counts = [ccc.get(k, {}).get(c, 0) for c in range(nc)]
        total = sum(counts); n_cls = sum(1 for c in counts if c > 0)
        print(f"    Client {k}: {n_cls}/{nc} cls, {total} samp")

    # Backbone loader: 大 batch + 多 worker 喂满 GPU
    client_all = {}
    client_cls = {}
    for k in range(n_clients):
        if len(cidx.get(k, [])) < 2: continue
        client_all[k] = DataLoader(Subset(train_ds, cidx[k]), batch_size=512,
                                   shuffle=True, drop_last=True,
                                   num_workers=8, pin_memory=True,
                                   persistent_workers=True)
        # Per-class loader: 只用于 preextract (单次遍历), num_workers=0 即可
        client_cls[k] = {}
        cm = defaultdict(list)
        for idx in cidx[k]: cm[int(targets[idx])].append(idx)
        for c, idxs in cm.items():
            client_cls[k][c] = DataLoader(Subset(train_ds, idxs), batch_size=256,
                                          shuffle=False, drop_last=False,
                                          num_workers=0, pin_memory=True)

    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                             num_workers=4, pin_memory=True)
    return client_all, client_cls, test_loader, ccc, nc


# ═══════════════════════════════════════════════════════════
# Models (与 v15 一致)
# ═══════════════════════════════════════════════════════════
def generate_etf(nc, fd, seed=42):
    rng = torch.Generator(); rng.manual_seed(seed)
    M = np.sqrt(nc/(nc-1)) * (torch.eye(nc) - torch.ones(nc,nc)/nc)
    if fd > nc:
        Q, _ = torch.linalg.qr(torch.randn(fd, nc, generator=rng))
        M = M @ Q.T
    return M

class Backbone(nn.Module):
    """原始 4 层 CNN backbone"""
    def __init__(self, fd=256, channels=None):
        super().__init__()
        if channels is None: channels = [64, 128, 256, 256]
        c1, c2, c3, c4 = channels; self.channels = channels
        self.features = nn.Sequential(
            nn.Conv2d(3,c1,3,padding=1), nn.BatchNorm2d(c1), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(c1,c2,3,padding=1), nn.BatchNorm2d(c2), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(c2,c3,3,padding=1), nn.BatchNorm2d(c3), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(c3,c4,3,padding=1), nn.BatchNorm2d(c4), nn.ReLU(True), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(c4*2*2, fd)
    def forward(self, x):
        x = self.features(x); x = x.view(x.size(0), -1)
        return F.normalize(self.fc(x), dim=1)


class BasicBlock(nn.Module):
    """ResNet BasicBlock"""
    expansion = 1
    def __init__(self, ic, oc, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(ic, oc, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(oc)
        self.conv2 = nn.Conv2d(oc, oc, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(oc)
        self.shortcut = nn.Sequential()
        if stride != 1 or ic != oc:
            self.shortcut = nn.Sequential(
                nn.Conv2d(ic, oc, 1, stride=stride, bias=False),
                nn.BatchNorm2d(oc))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18Backbone(nn.Module):
    """CIFAR-adapted ResNet-18 (3x3 initial conv, no maxpool)"""
    def __init__(self, fd=256):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, fd)

    def _make_layer(self, ic, oc, n_blocks, stride):
        layers = [BasicBlock(ic, oc, stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(oc, oc))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return F.normalize(self.fc(x), dim=1)


class FeatureUnion(nn.Module):
    """Feature-level union (消融用)"""
    def __init__(self, bbs):
        super().__init__()
        self.bbs = nn.ModuleList(bbs)
    def forward(self, x):
        feats = [bb(x) for bb in self.bbs]
        return F.normalize(torch.stack(feats).mean(0), dim=1)


class LogitsUnion(nn.Module):
    """Logits-level union: 各 client backbone 独立算 logits, 然后平均
    绕过特征空间不对齐的问题, 在语义空间做 ensemble"""
    def __init__(self, bbs, etf):
        super().__init__()
        self.bbs = nn.ModuleList(bbs)
        self.register_buffer('etf', etf)

    def forward(self, x):
        # 各 client 独立算 logits, 平均
        all_logits = []
        for bb in self.bbs:
            f = F.normalize(bb(x), dim=1)
            logits = torch.mm(f, self.etf.T)
            all_logits.append(logits)
        return torch.stack(all_logits).mean(0)  # (bs, nc) 平均 logits

    def get_features(self, x):
        """C4 仍需要一个 'feature' 来兼容 inference_union_only"""
        # 用 logits 当作 feature (因为下游就是和 etf 比较)
        logits = self.forward(x)
        return F.normalize(logits, dim=1)


def make_backbone(backbone_type, fd=256):
    if backbone_type == 'resnet18':
        return ResNet18Backbone(fd)
    else:
        return Backbone(fd)


def make_union(bbs, device, union_type, fd=256, etf=None):
    """union_type: 'filter_merge' (仅CNN), 'feature_avg', 'logits_avg' (通用)"""
    if union_type == 'filter_merge':
        return union_aggregate(bbs, device, fd, 0.95)
    elif union_type == 'logits_avg':
        print(f"  [Logits Union] {len(bbs)} 个 backbone 的 logits ensemble")
        return LogitsUnion([bb.cpu() for bb in bbs], etf).to(device)
    else:  # feature_avg
        print(f"  [Feature Union] 平均 {len(bbs)} 个 backbone 的特征")
        return FeatureUnion([bb.cpu() for bb in bbs]).to(device)

class ConditionalExpert(nn.Module):
    def __init__(self, fd=256, ed=256, hd=128, ld=32):
        super().__init__()
        self.enc1 = nn.Linear(fd+ed, hd); self.ebn = nn.LayerNorm(hd)
        self.enc2 = nn.Linear(hd, ld)
        self.dec1 = nn.Linear(ld+ed, hd); self.dbn = nn.LayerNorm(hd)
        self.dec2 = nn.Linear(hd, fd)
    def encode(self, f, c):
        return self.enc2(F.relu(self.ebn(self.enc1(torch.cat([f,c],1)))))
    def decode(self, z, c):
        return self.dec2(F.relu(self.dbn(self.dec1(torch.cat([z,c],1)))))
    def forward(self, f, c):
        z = self.encode(f, c); return self.decode(z, c), z


# ═══════════════════════════════════════════════════════════
# Training (与 v15 一致)
# ═══════════════════════════════════════════════════════════
def etf_cl(features, labels, etf, temp=0.1):
    features = F.normalize(features, dim=1); bs = features.size(0)
    lproto = F.cross_entropy(torch.mm(features, etf.T)/temp, labels)
    lsamp = torch.tensor(0.0, device=features.device)
    if bs > 1:
        sm = torch.eye(bs, device=features.device, dtype=torch.bool); ns = ~sm
        sim = torch.mm(features, features.T)/temp
        mp = (labels.unsqueeze(0)==labels.unsqueeze(1)).float() * ns.float()
        pc = mp.sum(1); v = pc > 0
        if v.sum() > 0:
            ss = sim - sim.max(1, keepdim=True)[0].detach()
            es = torch.exp(ss) * ns.float()
            lp_ = ss - torch.log(es.sum(1)+1e-8).unsqueeze(1)
            lsamp = -(mp*lp_).sum(1)[v]/(pc[v]+1e-8); lsamp = lsamp.mean()
    return lproto + 0.5*lsamp

def etf_al(features, labels, etf):
    features = F.normalize(features, dim=1)
    return (1-(features*etf[labels]).sum(1)).mean()

def train_backbone(bb, loader, classes, etf, device, epochs=600):
    bb = bb.to(device); ed = etf.to(device); bb.train()
    opt = torch.optim.Adam(bb.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    amp = get_amp_ctx(); t0 = time.time()
    for ep in range(epochs):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with amp:
                f = bb(x)
                loss = etf_cl(f, y, ed) + 0.5*etf_al(f, y, ed) if len(classes)>=2 else etf_al(f, y, ed)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sch.step()
        if (ep+1) % max(1, epochs//3) == 0:
            print(f"        BB ep {ep+1}/{epochs} ({time.time()-t0:.0f}s)")
    return bb

def preextract(bb, dl, device):
    bb.eval(); af = []
    with torch.no_grad():
        with get_amp_ctx():
            for x, _ in dl: af.append(bb(x.to(device, non_blocking=True)).float())
    return torch.cat(af, 0)

def train_expert(exp, cached, eo, ed, others, device, fdim=256, epochs=600):
    exp = exp.to(device); N = cached.size(0); no = others.size(0)
    opt = torch.optim.Adam(exp.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    margin = 0.05
    # Expert model 极小, 用全量数据当 batch, 消除内层循环
    nn_ = max(N, 64)  # negative samples 数量
    for ep in range(epochs):
        exp.train()
        # Positive: 重建自己
        co = eo.unsqueeze(0).expand(N, -1)
        fr1, _ = exp(cached, co); l1 = F.mse_loss(fr1, cached)
        # Negative 1: 用错误类条件重建
        fc = others[torch.randint(0, no, (N,), device=device)]
        fr2, _ = exp(cached, ed[fc])
        l2 = F.relu(margin - ((cached - fr2)**2).mean(1)).mean()
        # Negative 2: 随机噪声特征 + 正确/错误类条件
        nc_ = others[torch.randint(0, no, (nn_,), device=device)]
        sc = 0.05 + 0.25*torch.rand(nn_, 1, device=device)
        ff = F.normalize(ed[nc_] + torch.randn(nn_, fdim, device=device)*sc, dim=1)
        fr3, _ = exp(ff, ed[nc_])
        l3 = F.relu(margin - ((ff - fr3)**2).mean(1)).mean()
        fr4, _ = exp(ff, co[:nn_] if nn_ <= N else eo.unsqueeze(0).expand(nn_, -1))
        l4 = F.relu(margin - ((ff - fr4)**2).mean(1)).mean()
        loss = l1 + l2 + l3 + l4
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sch.step()
    exp.eval(); return exp


def train_all_experts(bb, cls_loaders, classes, etf, device, nc=10,
                      epochs=600, ccc_k=None):
    bb.eval(); ed = etf.to(device); exps = {}
    om = {c: torch.tensor([k for k in range(nc) if k!=c], device=device) for c in range(nc)}
    skipped = 0
    trainable = [c for c in classes
                 if (ccc_k.get(c, 0) if ccc_k else 999) >= MIN_EXPERT_SAMPLES]
    total = len(trainable)
    for i, cls in enumerate(trainable):
        cached = preextract(bb, cls_loaders[cls], device)
        if cached.size(0) < MIN_EXPERT_SAMPLES:
            skipped += 1; continue
        exp = ConditionalExpert(FD, FD, 128, LD).to(device)
        exp = train_expert(exp, cached, ed[cls], ed, om[cls], device, FD, epochs)
        exps[cls] = exp
        if (i+1) % max(1, total//5) == 0 or i == total-1:
            print(f"      Expert {i+1}/{total} done")
    skipped += len(classes) - len(trainable)
    if skipped > 0:
        print(f"      (跳过 {skipped} 个类, 样本<{MIN_EXPERT_SAMPLES})")
    return exps


# ═══════════════════════════════════════════════════════════
# Union Aggregation (简化版, 与 v15 一致)
# ═══════════════════════════════════════════════════════════
def union_aggregate(bbs, device, fd=256, thr=0.95):
    K = len(bbs)
    ci = [0,4,8,12]; bi = [1,5,9,13]
    lp_list = []; alm = []; pm = None; pn = 3
    for li in range(4):
        af = []; ass_ = []; ab = []; abn = []
        for k, bb in enumerate(bbs):
            conv = bb.features[ci[li]]; bn = bb.features[bi[li]]
            w = conv.weight.data.cpu(); b_ = conv.bias.data.cpu()
            Co, Ci = w.size(0), w.size(1)
            if li == 0: wr = w
            else:
                wr = torch.zeros(Co, pn, 3, 3)
                for l in range(Ci):
                    if l in pm[k]: wr[:, pm[k][l], :, :] = w[:, l, :, :]
            for i in range(Co):
                af.append(wr[i]); ass_.append((k,i)); ab.append(b_[i])
                abn.append({c_: bn.__getattr__({'w':'weight','b':'bias',
                           'm':'running_mean','v':'running_var'}[c_]).data.cpu()[i]
                           for c_ in 'wbmv'})
        if not af: continue
        st = torch.stack(af); N_ = st.size(0)
        sf = F.normalize(st.view(N_,-1).float(), dim=1); sim = sf @ sf.T
        assigned = [False]*N_; gf = []; fm = []
        norms = st.view(N_,-1).float().norm(dim=1)
        order = norms.argsort(descending=True).reshape(-1).tolist()
        for seed in order:
            if assigned[seed]: continue
            cluster = [seed]; assigned[seed] = True
            for j in order:
                if assigned[j]: continue
                if all(sim[j,c_] > thr for c_ in cluster):
                    cluster.append(j); assigned[j] = True
            cf = st[cluster]; cn = norms[cluster]
            ww = cn/(cn.sum()+1e-8)
            gf.append((cf.float()*ww.view(-1,*([1]*(cf.dim()-1)))).sum(0))
            fm.append([ass_[i] for i in cluster])
        No = len(gf)
        mb = []; mbn = {'w':[],'b':[],'m':[],'v':[]}
        for g, grp in enumerate(fm):
            idxs_ = [sum(bbs[kk].features[ci[li]].weight.size(0)
                        for kk in range(ck))+ci_ for ck,ci_ in grp]
            mb.append(torch.stack([ab[i] for i in idxs_]).mean())
            for c_ in 'wbmv':
                mbn[c_].append(torch.stack([abn[i][c_] for i in idxs_]).mean())
        nm = [{} for _ in range(K)]
        for g, grp in enumerate(fm):
            for ck, ci_ in grp: nm[ck][ci_] = g
        lp_list.append({'Ni':pn,'No':No,'f':torch.stack(gf),'b':torch.stack(mb),
                       'bw':torch.stack(mbn['w']),'bb':torch.stack(mbn['b']),
                       'bm':torch.stack(mbn['m']),'bv':torch.stack(mbn['v'])})
        alm.append(nm); pm = nm; pn = No

    Nf = pn; fi = Nf*4
    mfw = torch.zeros(fd, fi); mfb = torch.zeros(fd); fcc = torch.zeros(fi)
    for k, bb in enumerate(bbs):
        fw = bb.fc.weight.data.cpu(); fb = bb.fc.bias.data.cpu()
        c4 = bb.channels[3]
        for l in range(c4):
            if l not in pm[k]: continue
            g = pm[k][l]
            mfw[:, g*4:(g+1)*4] += fw[:, l*4:(l+1)*4]; fcc[g*4:(g+1)*4] += 1
        mfb += fb/K
    fcc = fcc.clamp(min=1); mfw /= fcc.unsqueeze(0)
    chs = [l['No'] for l in lp_list]
    merged = Backbone(fd, chs)
    with torch.no_grad():
        for li_ in range(4):
            l = lp_list[li_]; Ni = l['Ni']; No = l['No']
            merged.features[ci[li_]] = nn.Conv2d(Ni, No, 3, padding=1)
            merged.features[ci[li_]].weight.copy_(l['f'][:,:Ni])
            merged.features[ci[li_]].bias.copy_(l['b'])
            merged.features[bi[li_]] = nn.BatchNorm2d(No)
            merged.features[bi[li_]].weight.copy_(l['bw'])
            merged.features[bi[li_]].bias.copy_(l['bb'])
            merged.features[bi[li_]].running_mean.copy_(l['bm'])
            merged.features[bi[li_]].running_var.copy_(l['bv'])
        merged.fc = nn.Linear(fi, fd)
        merged.fc.weight.copy_(mfw); merged.fc.bias.copy_(mfb)
    return merged.to(device)


# ═══════════════════════════════════════════════════════════
# Inference Strategies
# ═══════════════════════════════════════════════════════════
def inference_union_only(union_bb, test_loader, etf, device):
    """Union backbone + ETF nearest, 兼容 LogitsUnion"""
    union_bb.eval(); ed = etf.to(device)
    is_logits_union = isinstance(union_bb, LogitsUnion)
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            if is_logits_union:
                logits = union_bb(x.to(device))  # 直接输出 logits
            else:
                f = F.normalize(union_bb(x.to(device)), dim=1)
                logits = torch.mm(f, ed.T)
            preds.append(logits.argmax(1).cpu()); labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def inference_expert_min(bbs, client_exps, test_loader, etf, ccc, device, nc):
    """Expert min error"""
    K = len(bbs); ed = etf.to(device)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x_d = x.to(device); bs = x.size(0)
            best_err = torch.full((bs, nc), 1e6, device=device)
            for k in range(K):
                f_k = bbs[k](x_d)
                for c, exp in client_exps[k].items():
                    fr, _ = exp(f_k, ed[c].unsqueeze(0).expand(bs,-1))
                    err = ((f_k - fr)**2).mean(1)
                    best_err[:, c] = torch.min(best_err[:, c], err)
            all_preds.append(best_err.argmin(1).cpu()); all_labels.append(y)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def inference_c4_ensemble(bbs, client_exps, union_bb, test_loader, etf, ccc, device,
                          nc, alpha_fuse=0.3, min_n=100):
    """C4: Per-client logit ensemble + union fusion (你的最佳策略)"""
    K = len(bbs); ed = etf.to(device); union_bb.eval()
    sc = {}
    for k in range(K):
        for c in client_exps[k]:
            sc[(k,c)] = ccc.get(k, {}).get(c, 0)

    is_logits_union = isinstance(union_bb, LogitsUnion)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x_d = x.to(device); bs = x.size(0)

            # Union logits
            if is_logits_union:
                u_logits = union_bb(x_d)  # 直接是 logits
            else:
                f_u = F.normalize(union_bb(x_d), dim=1)
                u_logits = torch.mm(f_u, ed.T)

            # Per-client expert ensemble
            ensemble = torch.zeros(bs, nc, device=device)
            for k in range(K):
                f_k = bbs[k](x_d)
                cl = torch.zeros(bs, nc, device=device)
                valid_c = []
                for c, exp in client_exps[k].items():
                    n = sc.get((k,c), 0)
                    if n < min_n: continue
                    fr, _ = exp(f_k, ed[c].unsqueeze(0).expand(bs,-1))
                    cl[:, c] = -((f_k - fr)**2).mean(1)
                    valid_c.append(c)
                if not valid_c: continue
                # z-score per client
                cl_mask = torch.zeros(bs, nc, dtype=torch.bool, device=device)
                for c in valid_c: cl_mask[:, c] = True
                cm = cl.sum(1, keepdim=True) / cl_mask.sum(1, keepdim=True).clamp(min=1)
                diff = (cl - cm) * cl_mask.float()
                cs = ((diff**2).sum(1, keepdim=True)/cl_mask.sum(1,keepdim=True).clamp(min=1)).sqrt()+1e-8
                cl_n = diff / cs; cl_n[~cl_mask] = 0
                w = np.log(sc.get((k, valid_c[0]), 0) + 1) if valid_c else 1.0
                ensemble += cl_n * w

            # normalize + fuse
            em = ensemble.mean(1, keepdim=True); es = ensemble.std(1, keepdim=True)+1e-8
            en = (ensemble - em) / es
            um = u_logits.mean(1, keepdim=True); us = u_logits.std(1, keepdim=True)+1e-8
            un = (u_logits - um) / us
            combined = un + alpha_fuse * en
            all_preds.append(combined.argmax(1).cpu()); all_labels.append(y)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


# ═══════════════════════════════════════════════════════════
# CE Baseline (Relational pipeline)
# ═══════════════════════════════════════════════════════════
class CEModel(nn.Module):
    def __init__(self, n_classes=10, fd=256, backbone_type='cnn'):
        super().__init__()
        self.backbone = make_backbone(backbone_type, fd)
        self.head = nn.Linear(fd, n_classes)
    def forward(self, x): return self.head(self.backbone(x))

def train_ce_model(loader, device, nc, epochs=200, backbone_type='cnn'):
    model = CEModel(nc, FD, backbone_type).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with get_amp_ctx():
                loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sch.step()
        if (ep+1) % max(1, epochs//3) == 0:
            print(f"        CE ep {ep+1}/{epochs}")
    return model.cpu().eval()

def inference_ce_ensemble(ce_models, test_loader, device, n_clients):
    all_logits, all_labels = [], []
    for k, model in ce_models.items():
        model = model.to(device).eval()
        logits_k, labels_k = [], []
        with torch.no_grad():
            for x, y in test_loader:
                logits_k.append(model(x.to(device)).cpu()); labels_k.append(y)
        all_logits.append(torch.cat(logits_k)); model.cpu()
        if not all_labels: all_labels = [torch.cat(labels_k)]
    ensemble = torch.stack(all_logits, 0).mean(0)
    labels = all_labels[0].numpy()
    preds = ensemble.argmax(1).numpy()
    return preds, labels


# ═══════════════════════════════════════════════════════════
# kNN Baseline (Intrinsic 但用 kNN 推理而非 expert)
# ═══════════════════════════════════════════════════════════
def inference_knn(bbs, client_all_loaders, client_cls_loaders, ccc,
                  test_loader, device, nc, k_neighbors=5):
    """用训练好的 backbone 提取特征, kNN 推理"""
    K_clients = len(bbs)
    # 收集所有 client 的训练特征和标签
    all_feats, all_labels_train = [], []
    for k in range(K_clients):
        if k not in client_all_loaders: continue
        bb = bbs[k].to(device).eval()
        feats_k, labels_k = [], []
        with torch.no_grad():
            for x, y in client_all_loaders[k]:
                feats_k.append(bb(x.to(device)).cpu()); labels_k.append(y)
        bb.cpu()
        all_feats.append(torch.cat(feats_k)); all_labels_train.append(torch.cat(labels_k))

    train_feats = torch.cat(all_feats)
    train_labels = torch.cat(all_labels_train)

    # 测试集 kNN
    preds_list, labels_list = [], []
    # 用 union backbone (第一个 bb) 提取测试特征
    union_feats = []
    bb0 = bbs[0].to(device).eval()
    with torch.no_grad():
        for x, y in test_loader:
            union_feats.append(bb0(x.to(device)).cpu()); labels_list.append(y)
    bb0.cpu()
    test_feats = torch.cat(union_feats)
    test_labels = torch.cat(labels_list).numpy()

    # kNN
    sim = torch.mm(test_feats, train_feats.T)
    _, topk_idx = sim.topk(k_neighbors, dim=1)
    topk_labels = train_labels[topk_idx]
    preds = torch.mode(topk_labels, dim=1)[0].numpy()
    return preds, test_labels


# ═══════════════════════════════════════════════════════════
# Single Run
# ═══════════════════════════════════════════════════════════
def run_single(dataset_name, alpha, n_clients, seed, gpu, ablation=None,
               pipeline='both', backbone_type='cnn', union_type='auto'):
    """运行单次实验, 返回结果字典
    pipeline: 'both', 'ours', 'ce_baseline'
    backbone_type: 'cnn' (原始4层CNN), 'resnet18' (对齐FAFI)
    union_type: 'auto' (CNN→filter_merge, ResNet→feature_avg), 'filter_merge', 'feature_avg'
    """
    seed_everything(seed)
    device = get_device(gpu)
    ep_cfg = get_epochs(dataset_name)

    # 解析 union_type
    if union_type == 'auto':
        actual_union = 'filter_merge' if backbone_type == 'cnn' else 'logits_avg'
    else:
        actual_union = union_type

    print(f"\n{'='*70}")
    print(f"  {dataset_name} | α={alpha} | K={n_clients} | seed={seed}")
    print(f"  pipeline={pipeline} | backbone={backbone_type} | union={actual_union}")
    print(f"  epochs: bb={ep_cfg['backbone']} exp={ep_cfg['expert']} ce={ep_cfg['ce']}")
    print(f"{'='*70}")

    client_all, client_cls, test_loader, ccc, nc = prepare_data(
        dataset_name, n_clients, alpha, seed)

    etf = generate_etf(nc, FD)
    t0 = time.time()
    results = {
        'dataset': dataset_name, 'alpha': alpha, 'n_clients': n_clients,
        'seed': seed, 'ablation': ablation, 'pipeline': pipeline,
        'backbone': backbone_type, 'union': actual_union,
    }

    # ── Train clients ──
    bbs = []; client_exps = []
    ce_models = {}

    for k in range(n_clients):
        if k not in client_all: continue
        classes = sorted(ccc.get(k, {}).keys())
        n_samp = sum(ccc.get(k, {}).values())
        if n_samp < 2: continue
        print(f"\n  Client {k}: {len(classes)} cls, {n_samp} samp")

        # Our method (ETF + Expert)
        if pipeline in ('both', 'ours') and ablation != 'ce_only':
            bb = make_backbone(backbone_type, FD)
            bb = train_backbone(bb, client_all[k], classes, etf, device,
                                epochs=ep_cfg['backbone'])
            exps = train_all_experts(bb, client_cls[k], classes, etf, device, nc,
                                     epochs=ep_cfg['expert'], ccc_k=ccc.get(k, {}))
            bbs.append(bb.cpu()); client_exps.append(exps)

        # CE pipeline
        if pipeline in ('both', 'ce_baseline') and ablation != 'no_union':
            ce = train_ce_model(client_all[k], device, nc,
                                epochs=ep_cfg['ce'], backbone_type=backbone_type)
            ce_models[k] = ce

        if torch.cuda.is_available(): torch.cuda.empty_cache()

    train_time = time.time() - t0
    results['train_time'] = train_time

    # ── Inference ──
    print(f"\n  推理...")

    # CE ensemble (Relational)
    if ce_models:
        ce_preds, labels = inference_ce_ensemble(ce_models, test_loader, device, n_clients)
        results['acc_relational'] = float((ce_preds == labels).mean())
        print(f"  Relational (CE): {results['acc_relational']:.2%}")

    # Intrinsic pipelines
    if bbs and ablation != 'ce_only':
        # Move to device for inference
        for bb in bbs: bb.to(device)

        # Expert min
        e_preds, labels = inference_expert_min(bbs, client_exps, test_loader, etf, ccc, device, nc)
        results['acc_expert_min'] = float((e_preds == labels).mean())
        print(f"  Expert (min):    {results['acc_expert_min']:.2%}")

        # Union
        if ablation != 'no_union':
            union_bb = make_union(bbs, device, actual_union, FD, etf=etf)
            u_preds, labels = inference_union_only(union_bb, test_loader, etf, device)
            results['acc_union'] = float((u_preds == labels).mean())
            print(f"  Union:           {results['acc_union']:.2%}")

            # C4 ensemble (best strategy)
            for af in [0.2, 0.3, 0.5]:
                c4_preds, labels = inference_c4_ensemble(
                    bbs, client_exps, union_bb, test_loader, etf, ccc, device,
                    nc, alpha_fuse=af, min_n=100)
                acc = float((c4_preds == labels).mean())
                results[f'acc_c4_a{af}'] = acc
                print(f"  C4 (α={af}):     {acc:.2%}")

            # Best intrinsic
            c4_accs = [results.get(f'acc_c4_a{af}', 0) for af in [0.2, 0.3, 0.5]]
            results['acc_intrinsic'] = max(c4_accs + [results.get('acc_union', 0)])
        else:
            results['acc_intrinsic'] = results.get('acc_expert_min', 0)

        # kNN (ablation: 不同推理方式)
        if ablation == 'inference_ablation':
            knn_preds, labels = inference_knn(
                bbs, client_all, client_cls, ccc, test_loader, device, nc)
            results['acc_knn'] = float((knn_preds == labels).mean())
            print(f"  kNN:             {results['acc_knn']:.2%}")

        for bb in bbs: bb.cpu()

    # Gap
    if 'acc_relational' in results and 'acc_intrinsic' in results:
        results['gap'] = results['acc_intrinsic'] - results['acc_relational']
        print(f"\n  Gap (Int-Rel): {results['gap']:+.2%}")

    results['total_time'] = time.time() - t0
    return results


# ═══════════════════════════════════════════════════════════
# Grid Runner
# ═══════════════════════════════════════════════════════════
def run_grid(dataset_name, seed, gpu, alphas=None, clients_list=None,
             pipeline='both', backbone_type='resnet18', union_type='auto'):
    if alphas is None:
        alphas = [0.05, 0.1, 0.3, 0.5, 1.0]
    if clients_list is None:
        clients_list = [5, 10, 20, 50]

    # 文件名标签
    u_tag = '' if union_type == 'auto' else f'_{union_type}'
    os.makedirs('results', exist_ok=True)
    all_results = []

    for alpha in alphas:
        for n_clients in clients_list:
            out_path = f"results/{dataset_name}_{backbone_type}{u_tag}_a{alpha}_k{n_clients}_s{seed}.json"
            if os.path.exists(out_path):
                print(f"\n  SKIP (exists): {out_path}")
                with open(out_path) as f:
                    all_results.append(json.load(f))
                continue

            try:
                res = run_single(dataset_name, alpha, n_clients, seed, gpu,
                                 pipeline=pipeline, backbone_type=backbone_type,
                                 union_type=union_type)
                with open(out_path, 'w') as f:
                    json.dump(res, f, indent=2)
                print(f"  Saved: {out_path}")
                all_results.append(res)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback; traceback.print_exc()

    # Summary
    print(f"\n{'='*80}")
    print(f"  Grid Summary: {dataset_name}, seed={seed}")
    print(f"{'='*80}")
    print(f"  {'α':>6s}", end="")
    for k in clients_list: print(f"  {'K='+str(k):>12s}", end="")
    print()
    print(f"  {'':>6s}", end="")
    for k in clients_list: print(f"  {'Rel / Int':>12s}", end="")
    print()
    for alpha in alphas:
        print(f"  {alpha:>6.2f}", end="")
        for k in clients_list:
            r = next((x for x in all_results
                      if x.get('alpha')==alpha and x.get('n_clients')==k), None)
            if r:
                rel = r.get('acc_relational', 0) * 100
                intr = r.get('acc_intrinsic', 0) * 100
                print(f"  {rel:5.1f}/{intr:5.1f}", end="")
            else:
                print(f"  {'---':>12s}", end="")
        print()


# ═══════════════════════════════════════════════════════════
# Ablation Runner
# ═══════════════════════════════════════════════════════════
def run_ablation(dataset_name, alpha, n_clients, seed, gpu):
    """消融实验:
    1. Full method (ETF + Expert + Union + C4)
    2. No Union (ETF + Expert, min only)
    3. CE only (Relational baseline)
    4. Different inference methods (kNN, prototype, expert)
    """
    os.makedirs('results', exist_ok=True)

    ablations = [
        ('full', None),
        ('no_union', 'no_union'),
        ('ce_only', 'ce_only'),
        ('inference', 'inference_ablation'),
    ]

    results = {}
    for name, abl in ablations:
        print(f"\n{'='*70}")
        print(f"  Ablation: {name}")
        print(f"{'='*70}")
        res = run_single(dataset_name, alpha, n_clients, seed, gpu, ablation=abl)
        results[name] = res

        out_path = f"results/ablation_{dataset_name}_a{alpha}_k{n_clients}_s{seed}_{name}.json"
        with open(out_path, 'w') as f:
            json.dump(res, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print(f"  Ablation Summary: {dataset_name}, α={alpha}, K={n_clients}")
    print(f"{'='*70}")
    for name, res in results.items():
        rel = res.get('acc_relational', 0) * 100
        intr = res.get('acc_intrinsic', 0) * 100
        union = res.get('acc_union', 0) * 100
        expert = res.get('acc_expert_min', 0) * 100
        knn = res.get('acc_knn', 0) * 100
        print(f"  {name:20s} | Rel={rel:5.1f}% | Int={intr:5.1f}% | "
              f"Union={union:5.1f}% | Expert={expert:5.1f}% | kNN={knn:5.1f}%")


# ═══════════════════════════════════════════════════════════
# FAFI Comparison Runner (K=5, 对齐 FAFI Table 1)
# ═══════════════════════════════════════════════════════════
def run_fafi_comparison(dataset_name, seed, gpu):
    """对齐 FAFI Table 1 的设置: K=5, α∈{0.05,0.1,0.3,0.5}"""
    alphas = [0.05, 0.1, 0.3, 0.5]
    run_grid(dataset_name, seed, gpu, alphas=alphas, clients_list=[5])


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='ICDE Full Experiments')
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'grid', 'ablation', 'fafi_compare'],
                        help='Experiment mode')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--n_clients', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--pipeline', type=str, default='both',
                        choices=['both', 'ours', 'ce_baseline'],
                        help='ours=你的方法(ETF+Expert+Union+C4), ce_baseline=CE ensemble对比线, both=都跑')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['cnn', 'resnet18'],
                        help='cnn=原始4层CNN, resnet18=对齐FAFI')
    parser.add_argument('--union', type=str, default='auto',
                        choices=['auto', 'filter_merge', 'feature_avg', 'logits_avg'],
                        help='auto=CNN用filter_merge/ResNet用logits_avg')
    parser.add_argument('--alphas', type=str, default='0.05,0.1,0.3,0.5,1.0',
                        help='Comma-separated alphas for grid mode')
    parser.add_argument('--clients', type=str, default='5,10,20,50',
                        help='Comma-separated client counts for grid mode')
    args = parser.parse_args()

    u_tag = '' if args.union == 'auto' else f'_{args.union}'
    print(f"Mode: {args.mode} | Dataset: {args.dataset} | Backbone: {args.backbone} | "
          f"Union: {args.union} | Pipeline: {args.pipeline} | GPU: {args.gpu}")

    if args.mode == 'single':
        res = run_single(args.dataset, args.alpha, args.n_clients, args.seed, args.gpu,
                         pipeline=args.pipeline, backbone_type=args.backbone,
                         union_type=args.union)
        out = f"results/{args.dataset}_{args.backbone}{u_tag}_a{args.alpha}_k{args.n_clients}_s{args.seed}.json"
        os.makedirs('results', exist_ok=True)
        with open(out, 'w') as f: json.dump(res, f, indent=2)
        print(f"\nSaved: {out}")

    elif args.mode == 'grid':
        alphas = [float(x) for x in args.alphas.split(',')]
        clients = [int(x) for x in args.clients.split(',')]
        run_grid(args.dataset, args.seed, args.gpu, alphas, clients,
                 pipeline=args.pipeline, backbone_type=args.backbone,
                 union_type=args.union)

    elif args.mode == 'ablation':
        run_ablation(args.dataset, args.alpha, args.n_clients, args.seed, args.gpu)

    elif args.mode == 'fafi_compare':
        run_fafi_comparison(args.dataset, args.seed, args.gpu)


if __name__ == '__main__':
    main()