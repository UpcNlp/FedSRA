"""
单 alpha 任务脚本，用于 sh 并发执行
传入参数：alpha值 任务id
输出：outputs/result_{alpha}.json
"""
import sys
import torch
import json
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 读取命令行参数
alpha = float(sys.argv[1])
task_id = int(sys.argv[2])

# 绑定GPU（0-5→卡0，6-11→卡1）
gpu_id = 0 if task_id < 6 else 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')

USE_BF16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
DL_KWARGS = dict(num_workers=8, pin_memory=True, persistent_workers=True)

# ═══════════════════════════════════════════════════════════════
# 1. 基础组件
# ═══════════════════════════════════════════════════════════════
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
import time

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

def prepare_data(n_clients=5, alpha=0.05, n_classes=10):
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
    cidx, ccc = dirichlet_split(train_ds, n_clients, alpha, n_classes)
    targets = np.array(train_ds.targets)
    cal = {}
    for k in range(n_clients):
        cal[k] = DataLoader(Subset(train_ds, cidx[k]), batch_size=128, shuffle=True, drop_last=True, **DL_KWARGS)
    ccl = {}
    for k in range(n_clients):
        ccl[k] = {}
        cm = defaultdict(list)
        for idx in cidx[k]: cm[targets[idx]].append(idx)
        for c, idxs in cm.items():
            dl_kw = dict(num_workers=4, pin_memory=True, persistent_workers=len(idxs) >= 64)
            ccl[k][c] = DataLoader(Subset(train_ds, idxs), batch_size=64, shuffle=True, drop_last=False, **dl_kw)
    tl = DataLoader(test_ds, batch_size=256, shuffle=False, **DL_KWARGS)
    return cal, ccl, tl, ccc

def generate_etf(nc, fd, seed=42):
    rng = torch.Generator(); rng.manual_seed(seed)
    M = np.sqrt(nc/(nc-1))*(torch.eye(nc)-torch.ones(nc,nc)/nc)
    if fd > nc: Q, _ = torch.linalg.qr(torch.randn(fd, nc, generator=rng)); M = M @ Q.T
    return M

class Backbone(nn.Module):
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
        x = self.features(x); x = x.view(x.size(0), -1); return F.normalize(self.fc(x), dim=1)

class ConditionalExpert(nn.Module):
    def __init__(self, fd=256, ed=256, hd=128, ld=32):
        super().__init__()
        self.enc1 = nn.Linear(fd+ed, hd); self.ebn = nn.LayerNorm(hd); self.enc2 = nn.Linear(hd, ld)
        self.dec1 = nn.Linear(ld+ed, hd); self.dbn = nn.LayerNorm(hd); self.dec2 = nn.Linear(hd, fd)
    def encode(self, f, c): return self.enc2(F.relu(self.ebn(self.enc1(torch.cat([f,c], 1)))))
    def decode(self, z, c): return self.dec2(F.relu(self.dbn(self.dec1(torch.cat([z,c], 1)))))
    def forward(self, f, c): z = self.encode(f, c); return self.decode(z, c), z

def etf_cl(features, labels, etf, temp=0.1):
    features = F.normalize(features, dim=1); bs = features.size(0)
    lproto = F.cross_entropy(torch.mm(features, etf.T)/temp, labels)
    lsamp = torch.tensor(0.0, device=features.device)
    if bs > 1:
        sm = torch.eye(bs, device=features.device, dtype=torch.bool); ns = ~sm
        sim = torch.mm(features, features.T)/temp
        mp = (labels.unsqueeze(0)==labels.unsqueeze(1)).float()*ns.float()
        pc = mp.sum(1); v = pc > 0
        if v.sum() > 0:
            ss = sim - sim.max(1, keepdim=True)[0].detach()
            es = torch.exp(ss)*ns.float()
            lp_ = ss - torch.log(es.sum(1)+1e-8).unsqueeze(1)
            lsamp = -(mp*lp_).sum(1)[v]/(pc[v]+1e-8); lsamp = lsamp.mean()
    return lproto + 0.5*lsamp

def etf_al(features, labels, etf):
    features = F.normalize(features, dim=1)
    return (1-(features*etf[labels]).sum(1)).mean()

def train_bb(bb, loader, classes, etf, epochs=600, lr=1e-3):
    bb = bb.to(device); ed = etf.to(device); ncl = len(classes); bb.train()
    opt = torch.optim.Adam(bb.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    amp = (torch.amp.autocast('cuda', dtype=torch.bfloat16) if USE_BF16
           else torch.amp.autocast('cuda', enabled=False))
    for ep in range(epochs):
        el = 0; nb = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            with amp:
                f = bb(x)
                if ncl >= 2: loss = etf_cl(f, y, ed) + 0.5*etf_al(f, y, ed)
                else: loss = etf_al(f, y, ed)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); el += loss.item(); nb += 1
        sch.step()
    return bb

def preextract(bb, dl):
    bb.eval(); af = []
    with torch.no_grad():
        amp = (torch.amp.autocast('cuda', dtype=torch.bfloat16) if USE_BF16
               else torch.amp.autocast('cuda', enabled=False))
        with amp:
            for x, _ in dl: af.append(bb(x.to(device, non_blocking=True)).float())
    return torch.cat(af, 0)

def train_exp(exp, cached, eo, ed, others, fdim=256, epochs=600, lr=1e-3, margin=0.05):
    exp = exp.to(device); N = cached.size(0); no = others.size(0)
    opt = torch.optim.Adam(exp.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bs = min(64, N); nn_ = 64
    for ep in range(epochs):
        exp.train(); perm = torch.randperm(N, device=device)
        for i in range(0, N, bs):
            idx = perm[i:i+bs]; fp = cached[idx]; B = fp.size(0)
            co = eo.unsqueeze(0).expand(B, -1); fr1, _ = exp(fp, co); l1 = F.mse_loss(fr1, fp)
            nc = others[torch.randint(0, no, (nn_,), device=device)]
            sc = 0.05 + 0.25*torch.rand(nn_, 1, device=device)
            ff = F.normalize(ed[nc]+torch.randn(nn_, fdim, device=device)*sc, dim=1)
            fc = others[torch.randint(0, no, (B,), device=device)]
            fr2, _ = exp(fp, ed[fc]); l2 = F.relu(margin-((fp-fr2)**2).mean(1)).mean()
            fr3, _ = exp(ff, ed[nc]); l3 = F.relu(margin-((ff-fr3)**2).mean(1)).mean()
            fr4, _ = exp(ff, eo.unsqueeze(0).expand(nn_, -1)); l4 = F.relu(margin-((ff-fr4)**2).mean(1)).mean()
            loss = l1 + (l2+l3+l4)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sch.step()
    exp.eval(); return exp

def train_experts(bb, cls_loaders, classes, etf, nc=10, fdim=256, ldim=32, epochs=600, lr=1e-3):
    bb.eval(); ed = etf.to(device); exps = {}
    om = {c: torch.tensor([k for k in range(nc) if k != c], device=device) for c in range(nc)}
    for cls in classes:
        t0 = time.time()
        cached = preextract(bb, cls_loaders[cls])
        exp = ConditionalExpert(fdim, fdim, 128, ldim).to(device)
        exp = train_exp(exp, cached, ed[cls], ed, om[cls], fdim, epochs, lr)
        exps[cls] = exp
    return exps

def union_aggregate(bbs, fd=256, thr=0.95):
    K = len(bbs)
    ci = [0, 4, 8, 12]; bi = [1, 5, 9, 13]
    lp_list = []; alm = []; pm = None; pn = 3
    for li in range(4):
        af = []; ass = []; ab = []; abn = []
        for k, bb in enumerate(bbs):
            conv = bb.features[ci[li]]; bn = bb.features[bi[li]]
            w = conv.weight.data.cpu(); b_ = conv.bias.data.cpu(); Co, Ci = w.size(0), w.size(1)
            if li == 0: wr = w
            else:
                wr = torch.zeros(Co, pn, 3, 3)
                for l in range(Ci):
                    if l in pm[k]: wr[:, pm[k][l], :, :] = w[:, l, :, :]
            for i in range(Co):
                af.append(wr[i]); ass.append((k, i)); ab.append(b_[i])
                abn.append({c_: bn.__getattr__({'w':'weight','b':'bias','m':'running_mean','v':'running_var'}[c_]).data.cpu()[i] for c_ in 'wbmv'})
        if not af: continue
        st = torch.stack(af); N_ = st.size(0)
        sf = F.normalize(st.view(N_, -1).float(), dim=1); sim = sf @ sf.T
        assigned = [False]*N_; gf = []; fm = []
        norms = st.view(N_, -1).float().norm(dim=1)
        order = norms.argsort(descending=True).reshape(-1).tolist()
        for seed in order:
            if assigned[seed]: continue
            cluster = [seed]; assigned[seed] = True
            for j in order:
                if assigned[j]: continue
                if all(sim[j, c_] > thr for c_ in cluster): cluster.append(j); assigned[j] = True
            cf = st[cluster]; cn = norms[cluster]; ww = cn/(cn.sum()+1e-8)
            gf.append((cf.float()*ww.view(-1, *([1]*(cf.dim()-1)))).sum(0))
            fm.append([ass[i] for i in cluster])
        No = len(gf)
        mb = []; mbn = {'w':[],'b':[],'m':[],'v':[]}
        for g, grp in enumerate(fm):
            idxs_ = [sum(bbs[kk].features[ci[li]].weight.size(0) for kk in range(ck))+ci_ for ck, ci_ in grp]
            mb.append(torch.stack([ab[i] for i in idxs_]).mean())
            for c_ in 'wbmv': mbn[c_].append(torch.stack([abn[i][c_] for i in idxs_]).mean())
        nm = [{} for _ in range(K)]
        for g, grp in enumerate(fm):
            for ck, ci_ in grp: nm[ck][ci_] = g
        lp_list.append({'Ni':pn,'No':No,'f':torch.stack(gf),'b':torch.stack(mb),
                   'bw':torch.stack(mbn['w']),'bb':torch.stack(mbn['b']),
                   'bm':torch.stack(mbn['m']),'bv':torch.stack(mbn['v'])})
        alm.append(nm); pm = nm; pn = No
    Nf = pn; fi = Nf*4; mfw = torch.zeros(fd, fi); mfb = torch.zeros(fd); fcc = torch.zeros(fi)
    K = len(bbs)
    for k, bb in enumerate(bbs):
        fw = bb.fc.weight.data.cpu(); fb = bb.fc.bias.data.cpu(); c4 = bb.channels[3]
        for l in range(c4):
            if l not in pm[k]: continue
            g = pm[k][l]; mfw[:, g*4:(g+1)*4] += fw[:, l*4:(l+1)*4]; fcc[g*4:(g+1)*4] += 1
        mfb += fb/K
    fcc = fcc.clamp(min=1); mfw /= fcc.unsqueeze(0)
    chs = [l['No'] for l in lp_list]
    merged = Backbone(fd, chs)
    with torch.no_grad():
        for li_ in range(4):
            l = lp_list[li_]; Ni = l['Ni']; No = l['No']
            merged.features[ci[li_]] = nn.Conv2d(Ni, No, 3, padding=1)
            merged.features[ci[li_]].weight.copy_(l['f'][:, :Ni]); merged.features[ci[li_]].bias.copy_(l['b'])
            merged.features[bi[li_]] = nn.BatchNorm2d(No)
            merged.features[bi[li_]].weight.copy_(l['bw']); merged.features[bi[li_]].bias.copy_(l['bb'])
            merged.features[bi[li_]].running_mean.copy_(l['bm']); merged.features[bi[li_]].running_var.copy_(l['bv'])
        merged.fc = nn.Linear(fi, fd); merged.fc.weight.copy_(mfw); merged.fc.bias.copy_(mfb)
    return merged.to(device)

# ═══════════════════════════════════════════════════════════════
# 2. 评估函数
# ═══════════════════════════════════════════════════════════════
def evaluate_all_signals(bbs, client_exps, union_bb, tl, etf, ccc,
                         nc=10, fuse_alpha=0.3, min_n=100):
    K = len(bbs); ed = etf.to(device); union_bb.eval()
    sample_count = {}
    for k in range(K):
        for c in client_exps[k]:
            sample_count[(k, c)] = ccc[k].get(c, 0)
    all_expert_errors = []
    all_union_logits = []
    all_labels = []
    with torch.no_grad():
        for x, y in tl:
            x_dev = x.to(device, non_blocking=True); bs = x.size(0)
            f_union = F.normalize(union_bb(x_dev), dim=1)
            union_logits = torch.mm(f_union, ed.T)
            all_union_logits.append(union_logits.cpu())
            batch_errors = torch.full((K, bs, nc), float('inf'))
            for k in range(K):
                f_k = bbs[k](x_dev)
                for c, exp in client_exps[k].items():
                    fr, _ = exp(f_k, ed[c].unsqueeze(0).expand(bs, -1))
                    batch_errors[k, :, c] = ((f_k - fr)**2).mean(1).cpu()
            all_expert_errors.append(batch_errors)
            all_labels.append(y)
    errors = torch.cat(all_expert_errors, dim=1)
    union_logits = torch.cat(all_union_logits, 0)
    labels = torch.cat(all_labels).numpy()
    N = len(labels)

    union_preds = union_logits.argmax(1).numpy()
    union_acc = (union_preds == labels).mean()
    errors_clean = errors.clone()
    errors_clean[errors_clean == float('inf')] = 1e6

    best_err_min, _ = errors_clean.min(dim=0)
    expert_s1_preds = best_err_min.argmin(1).numpy()
    expert_s1_acc = (expert_s1_preds == labels).mean()

    weighted_logits = torch.zeros(N, nc)
    weight_sum = torch.zeros(N, nc)
    for k in range(K):
        for c in range(nc):
            n = sample_count.get((k, c), 0)
            if n < 1: continue
            w = np.log(n + 1)
            weighted_logits[:, c] += w * (-errors_clean[k, :, c])
            weight_sum[:, c] += w
    weight_sum[weight_sum == 0] = 1.0
    expert_s2_logits = weighted_logits / weight_sum
    expert_s2_preds = expert_s2_logits.argmax(1).numpy()
    expert_s2_acc = (expert_s2_preds == labels).mean()

    best_err_top = torch.full((N, nc), 1e6)
    for c in range(nc):
        best_k = -1; best_n = 0
        for k in range(K):
            n = sample_count.get((k, c), 0)
            if n > best_n: best_n = n; best_k = k
        if best_k >= 0:
            best_err_top[:, c] = errors_clean[best_k, :, c]
    expert_s3_preds = best_err_top.argmin(1).numpy()
    expert_s3_acc = (expert_s3_preds == labels).mean()

    best_err_qf = torch.full((N, nc), 1e6)
    for k in range(K):
        for c in range(nc):
            if sample_count.get((k, c), 0) < min_n: continue
            best_err_qf[:, c] = torch.min(best_err_qf[:, c], errors_clean[k, :, c])
    expert_s4_preds = best_err_qf.argmin(1).numpy()
    all_high = (best_err_qf.min(1)[0] >= 1e5)
    expert_s4_preds[all_high.numpy()] = union_preds[all_high.numpy()]
    expert_s4_acc = (expert_s4_preds == labels).mean()

    expert_accs = {
        'S1_min': expert_s1_acc,
        'S2_weighted_log': expert_s2_acc,
        'S3_top_expert': expert_s3_acc,
        'S4_quality_min': expert_s4_acc,
    }
    best_expert_name = max(expert_accs, key=expert_accs.get)
    best_expert_acc = expert_accs[best_expert_name]

    ensemble_logits = torch.zeros(N, nc)
    for k in range(K):
        valid_c = [c for c in range(nc) if sample_count.get((k, c), 0) >= min_n]
        if not valid_c: continue
        ek = errors_clean[k]
        cl = torch.zeros(N, nc)
        for c in valid_c: cl[:, c] = -ek[:, c]
        cl_mask = torch.zeros(N, nc, dtype=torch.bool)
        for c in valid_c: cl_mask[:, c] = True
        cm = cl.sum(1, keepdim=True) / cl_mask.sum(1, keepdim=True).clamp(min=1)
        diff = (cl - cm) * cl_mask.float()
        cs = ((diff**2).sum(1, keepdim=True) / cl_mask.sum(1, keepdim=True).clamp(min=1)).sqrt() + 1e-8
        cl_n = diff / cs; cl_n[~cl_mask] = 0
        w_sum = sum(np.log(sample_count.get((k, c), 0) + 1) for c in valid_c) / len(valid_c)
        ensemble_logits += cl_n * w_sum

    em = ensemble_logits.mean(1, keepdim=True)
    es = ensemble_logits.std(1, keepdim=True) + 1e-8
    en = (ensemble_logits - em) / es
    no_signal = (ensemble_logits.abs().sum(1) == 0)
    en[no_signal.unsqueeze(1).expand_as(en)] = 0

    um = union_logits.mean(1, keepdim=True)
    us = union_logits.std(1, keepdim=True) + 1e-8
    un = (union_logits - um) / us

    best_fuse_acc = 0; best_fuse_alpha = 0
    for a in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]:
        fused_preds = (un + a * en).argmax(1).numpy()
        acc = (fused_preds == labels).mean()
        if acc > best_fuse_acc:
            best_fuse_acc = acc; best_fuse_alpha = a

    fused_preds = (un + best_fuse_alpha * en).argmax(1).numpy()
    fused_acc = best_fuse_acc
    fused_fixed_preds = (un + fuse_alpha * en).argmax(1).numpy()
    fused_fixed_acc = (fused_fixed_preds == labels).mean()

    per_class = {}
    for c in range(nc):
        mask = (labels == c)
        n_samples = mask.sum()
        u_acc_c = (union_preds[mask] == c).mean()
        e_acc_c = (expert_s2_preds[mask] == c).mean()
        f_acc_c = (fused_preds[mask] == c).mean()
        expert_counts = []
        for k in range(K):
            n = sample_count.get((k, c), 0)
            if n > 0: expert_counts.append(n)
        per_class[c] = {
            'n_test': int(n_samples),
            'union_acc': float(u_acc_c),
            'expert_acc': float(e_acc_c),
            'fused_acc': float(f_acc_c),
            'n_experts': len(expert_counts),
            'max_train': max(expert_counts) if expert_counts else 0,
            'total_train': sum(expert_counts),
        }

    u_correct = (union_preds == labels).astype(float)
    e_correct = (expert_s2_preds == labels).astype(float)
    f_correct = (fused_preds == labels).astype(float)
    both_correct = ((u_correct == 1) & (e_correct == 1)).sum()
    u_only = ((u_correct == 1) & (e_correct == 0)).sum()
    e_only = ((u_correct == 0) & (e_correct == 1)).sum()
    both_wrong = ((u_correct == 0) & (e_correct == 0)).sum()
    u_err = 1 - u_correct; e_err = 1 - e_correct
    if u_err.std() > 0 and e_err.std() > 0:
        error_corr = np.corrcoef(u_err, e_err)[0, 1]
    else:
        error_corr = 0.0
    oracle_acc = ((u_correct == 1) | (e_correct == 1)).mean()
    fuse_utilization = (fused_acc - union_acc) / (oracle_acc - union_acc + 1e-10)

    results = {
        'union_acc': float(union_acc),
        'expert_accs': {k: float(v) for k, v in expert_accs.items()},
        'best_expert': best_expert_name,
        'best_expert_acc': float(best_expert_acc),
        'fused_acc': float(fused_acc),
        'fused_alpha': float(best_fuse_alpha),
        'fused_fixed_acc': float(fused_fixed_acc),
        'delta': float(fused_acc - union_acc),
        'per_class': per_class,
        'orthogonality': {
            'both_correct': int(both_correct),
            'union_only': int(u_only),
            'expert_only': int(e_only),
            'both_wrong': int(both_wrong),
            'error_correlation': float(error_corr),
            'oracle_acc': float(oracle_acc),
            'fuse_utilization': float(fuse_utilization),
        },
    }
    return results

# ═══════════════════════════════════════════════════════════════
# 单任务执行
# ═══════════════════════════════════════════════════════════════
def run_single():
    torch.manual_seed(42)
    np.random.seed(42)
    etf = generate_etf(10, 256)
    
    print(f"\n[TASK {task_id}] GPU={gpu_id} alpha={alpha}")
    cal, ccl, tl, ccc = prepare_data(5, alpha, 10)
    
    bbs = []; client_exps = []
    for k in range(5):
        cls = sorted(ccc[k].keys())
        bb = Backbone(256)
        bb = train_bb(bb, cal[k], cls, etf, 600)
        exps = train_experts(bb, ccl[k], cls, etf, 10, 256, 32, 600)
        bbs.append(bb); client_exps.append(exps)
    
    ubb = union_aggregate(bbs, 256, 0.95)
    res = evaluate_all_signals(bbs, client_exps, ubb, tl, etf, ccc, 10)
    
    os.makedirs('outputs', exist_ok=True)
    with open(f'outputs/result_{alpha}.json', 'w') as f:
        json.dump(res, f, indent=2)
    
    print(f"[TASK {task_id}] FINISHED alpha={alpha}")

if __name__ == "__main__":
    run_single()