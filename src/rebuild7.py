"""
MoE v14 — 跨 Client 交叉信号 (基于 V4, 只加推理策略)

假说: 单 client expert 无法拒绝未见类, 但多 client 交叉验证可能有用
  - 如果 truck 被 client0(dog-expert) 和 client1(cat-expert) 都接受
    → 说明不是真 dog 也不是真 cat (真 dog 不会被 cat-expert 接受)
  - 如果只有 client0(dog-expert) 接受, 其他拒绝
    → 更可能是真 dog

策略 C: 跨 client 交叉验证 (纯推理, 不改训练)
  C1: 一致性投票 — expert 预测一致时信 expert
  C2: 唯一接受 — 只有一个 client 的 expert 低误差时信 expert
  C3: 交叉排斥 — 用其他 client 的拒绝信号修正 expert logits
  C4: 多 client ensemble — 各 client 的 expert logits 投票

训练与 V4 完全相同, 只加推理策略

运行: python moe_v14_cross.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import os

warnings.filterwarnings('ignore')

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
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}, BF16: {'ON' if USE_BF16 else 'OFF'}")
print("=" * 80)

DL_KWARGS = dict(num_workers=8, pin_memory=True, persistent_workers=True)
CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']


# ═══════════════════════════════════════════════════════════
# 1. 基础组件 (与原始代码一致)
# ═══════════════════════════════════════════════════════════

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
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(32,padding=4),
        transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)],p=0.8),
        transforms.RandomGrayscale(p=0.2), transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
        transforms.RandomErasing(p=0.25,scale=(0.02,0.2)),
    ])
    te = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    train_ds = datasets.CIFAR10(root='./data',train=True,download=True,transform=tt)
    test_ds = datasets.CIFAR10(root='./data',train=False,download=True,transform=te)
    cidx, ccc = dirichlet_split(train_ds, n_clients, alpha, n_classes)
    print(f"\n数据分布 (α={alpha}):")
    for k in range(n_clients):
        counts=[ccc[k].get(c,0) for c in range(n_classes)]
        top=sorted(range(n_classes),key=lambda c:counts[c],reverse=True)[:3]
        print(f"  Client {k}: {sum(1 for c in counts if c>0)} cls, {sum(counts)} samp, "
              f"top: {', '.join(f'c{c}={counts[c]}' for c in top)}")
    targets = np.array(train_ds.targets)
    cal = {}
    for k in range(n_clients):
        cal[k] = DataLoader(Subset(train_ds,cidx[k]),batch_size=128,shuffle=True,drop_last=True,**DL_KWARGS)
    ccl = {}
    for k in range(n_clients):
        ccl[k] = {}
        cm = defaultdict(list)
        for idx in cidx[k]: cm[targets[idx]].append(idx)
        for c,idxs in cm.items():
            dl_kw = dict(num_workers=4,pin_memory=True,persistent_workers=len(idxs)>=64)
            ccl[k][c] = DataLoader(Subset(train_ds,idxs),batch_size=64,shuffle=True,drop_last=False,**dl_kw)
    tl = DataLoader(test_ds,batch_size=256,shuffle=False,**DL_KWARGS)
    return cal, ccl, tl, ccc

def generate_etf(nc, fd, seed=42):
    rng=torch.Generator();rng.manual_seed(seed)
    M=np.sqrt(nc/(nc-1))*(torch.eye(nc)-torch.ones(nc,nc)/nc)
    if fd>nc: Q,_=torch.linalg.qr(torch.randn(fd,nc,generator=rng));M=M@Q.T
    print(f"  ETF: norm={torch.norm(M,dim=1).mean():.4f}"); return M

class Backbone(nn.Module):
    def __init__(self, fd=256, channels=None):
        super().__init__()
        if channels is None: channels=[64,128,256,256]
        c1,c2,c3,c4=channels; self.channels=channels
        self.features=nn.Sequential(
            nn.Conv2d(3,c1,3,padding=1),nn.BatchNorm2d(c1),nn.ReLU(True),nn.MaxPool2d(2),
            nn.Conv2d(c1,c2,3,padding=1),nn.BatchNorm2d(c2),nn.ReLU(True),nn.MaxPool2d(2),
            nn.Conv2d(c2,c3,3,padding=1),nn.BatchNorm2d(c3),nn.ReLU(True),nn.MaxPool2d(2),
            nn.Conv2d(c3,c4,3,padding=1),nn.BatchNorm2d(c4),nn.ReLU(True),nn.MaxPool2d(2),
        )
        self.fc=nn.Linear(c4*2*2,fd)
    def forward(self,x):
        x=self.features(x);x=x.view(x.size(0),-1);return F.normalize(self.fc(x),dim=1)
    def forward_conv(self,x):
        return self.features(x)

class ConditionalExpert(nn.Module):
    def __init__(self, fd=256, ed=256, hd=128, ld=32):
        super().__init__()
        self.enc1=nn.Linear(fd+ed,hd);self.ebn=nn.LayerNorm(hd);self.enc2=nn.Linear(hd,ld)
        self.dec1=nn.Linear(ld+ed,hd);self.dbn=nn.LayerNorm(hd);self.dec2=nn.Linear(hd,fd)
    def encode(self,f,c): return self.enc2(F.relu(self.ebn(self.enc1(torch.cat([f,c],1)))))
    def decode(self,z,c): return self.dec2(F.relu(self.dbn(self.dec1(torch.cat([z,c],1)))))
    def forward(self,f,c): z=self.encode(f,c);return self.decode(z,c),z

def etf_cl(features,labels,etf,temp=0.1):
    features=F.normalize(features,dim=1);bs=features.size(0)
    lproto=F.cross_entropy(torch.mm(features,etf.T)/temp,labels)
    lsamp=torch.tensor(0.0,device=features.device)
    if bs>1:
        sm=torch.eye(bs,device=features.device,dtype=torch.bool);ns=~sm
        sim=torch.mm(features,features.T)/temp
        mp=(labels.unsqueeze(0)==labels.unsqueeze(1)).float()*ns.float()
        pc=mp.sum(1);v=pc>0
        if v.sum()>0:
            ss=sim-sim.max(1,keepdim=True)[0].detach()
            es=torch.exp(ss)*ns.float()
            lp_=ss-torch.log(es.sum(1)+1e-8).unsqueeze(1)
            lsamp=-(mp*lp_).sum(1)[v]/(pc[v]+1e-8);lsamp=lsamp.mean()
    return lproto+0.5*lsamp

def etf_al(features,labels,etf):
    features=F.normalize(features,dim=1)
    return (1-(features*etf[labels]).sum(1)).mean()

def train_bb(bb,loader,classes,etf,epochs=600,lr=1e-3):
    bb=bb.to(device);ed=etf.to(device);ncl=len(classes);bb.train()
    opt=torch.optim.Adam(bb.parameters(),lr=lr)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    amp=(torch.amp.autocast('cuda',dtype=torch.bfloat16) if USE_BF16
         else torch.amp.autocast('cuda',enabled=False))
    for ep in range(epochs):
        el=0;nb=0
        for x,y in loader:
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            with amp:
                f=bb(x)
                if ncl>=2: loss=etf_cl(f,y,ed)+0.5*etf_al(f,y,ed)
                else: loss=etf_al(f,y,ed)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step();el+=loss.item();nb+=1
        sch.step()
        if (ep+1)%100==0 or ep==0: print(f"      BB {ep+1}/{epochs} loss={el/max(nb,1):.4f}")
    return bb

def preextract(bb,dl):
    bb.eval();af=[]
    with torch.no_grad():
        amp=(torch.amp.autocast('cuda',dtype=torch.bfloat16) if USE_BF16
             else torch.amp.autocast('cuda',enabled=False))
        with amp:
            for x,_ in dl: af.append(bb(x.to(device,non_blocking=True)).float())
    return torch.cat(af,0)

def train_exp(exp,cached,eo,ed,others,fdim=256,epochs=600,lr=1e-3,margin=0.05):
    exp=exp.to(device);N=cached.size(0);no=others.size(0)
    opt=torch.optim.Adam(exp.parameters(),lr=lr)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    bs=min(64,N);nn_=64
    for ep in range(epochs):
        exp.train();perm=torch.randperm(N,device=device)
        for i in range(0,N,bs):
            idx=perm[i:i+bs];fp=cached[idx];B=fp.size(0)
            co=eo.unsqueeze(0).expand(B,-1);fr1,_=exp(fp,co);l1=F.mse_loss(fr1,fp)
            nc=others[torch.randint(0,no,(nn_,),device=device)]
            sc=0.05+0.25*torch.rand(nn_,1,device=device)
            ff=F.normalize(ed[nc]+torch.randn(nn_,fdim,device=device)*sc,dim=1)
            fc=others[torch.randint(0,no,(B,),device=device)]
            fr2,_=exp(fp,ed[fc]);l2=F.relu(margin-((fp-fr2)**2).mean(1)).mean()
            fr3,_=exp(ff,ed[nc]);l3=F.relu(margin-((ff-fr3)**2).mean(1)).mean()
            fr4,_=exp(ff,eo.unsqueeze(0).expand(nn_,-1));l4=F.relu(margin-((ff-fr4)**2).mean(1)).mean()
            loss=l1+(l2+l3+l4)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        sch.step()
    exp.eval();return exp

def train_experts(bb,cls_loaders,classes,etf,nc=10,fdim=256,ldim=32,epochs=600,lr=1e-3):
    bb.eval();ed=etf.to(device);exps={}
    om={c:torch.tensor([k for k in range(nc) if k!=c],device=device) for c in range(nc)}
    for cls in classes:
        print(f"      Exp {cls}...",end=" ",flush=True);t0=time.time()
        cached=preextract(bb,cls_loaders[cls])
        exp=ConditionalExpert(fdim,fdim,128,ldim).to(device)
        exp=train_exp(exp,cached,ed[cls],ed,om[cls],fdim,epochs,lr)
        exps[cls]=exp
        with torch.no_grad():
            ne=min(256,cached.size(0));fr,_=exp(cached[:ne],ed[cls].unsqueeze(0).expand(ne,-1))
            print(f"done ({time.time()-t0:.1f}s) MSE={((cached[:ne]-fr)**2).mean().item():.6f}")
    return exps

def compute_stats(bb,exp,dl,ev):
    bb.eval();exp.eval();ev_=ev.to(device);errs=[]
    with torch.no_grad():
        for x,_ in dl:
            f=bb(x.to(device,non_blocking=True));B=f.size(0)
            fr,_=exp(f,ev_.unsqueeze(0).expand(B,-1));errs.append(((f-fr)**2).mean(1))
    errs=torch.cat(errs);return errs.mean().item(),errs.std().item()


# ═══════════════════════════════════════════════════════════
# 2. Union
# ═══════════════════════════════════════════════════════════

def union_aggregate(bbs, fd=256, thr=0.95):
    K=len(bbs);print(f"\n  [Union] {K} backbones, thr={thr}")
    ci=[0,4,8,12];bi=[1,5,9,13]
    lp_list=[];alm=[];pm=None;pn=3
    for li in range(4):
        af=[];ass=[];ab=[];abn=[]
        for k,bb in enumerate(bbs):
            conv=bb.features[ci[li]];bn=bb.features[bi[li]]
            w=conv.weight.data.cpu();b_=conv.bias.data.cpu();Co,Ci=w.size(0),w.size(1)
            if li==0: wr=w
            else:
                wr=torch.zeros(Co,pn,3,3)
                for l in range(Ci):
                    if l in pm[k]: wr[:,pm[k][l],:,:]=w[:,l,:,:]
            for i in range(Co):
                af.append(wr[i]);ass.append((k,i));ab.append(b_[i])
                abn.append({c_:bn.__getattr__({'w':'weight','b':'bias','m':'running_mean','v':'running_var'}[c_]).data.cpu()[i] for c_ in 'wbmv'})
        if not af: continue
        st=torch.stack(af);N_=st.size(0)
        sf=F.normalize(st.view(N_,-1).float(),dim=1);sim=sf@sf.T
        assigned=[False]*N_;gf=[];fm=[]
        norms=st.view(N_,-1).float().norm(dim=1)
        order=norms.argsort(descending=True).reshape(-1).tolist()
        for seed in order:
            if assigned[seed]:continue
            cluster=[seed];assigned[seed]=True
            for j in order:
                if assigned[j]:continue
                if all(sim[j,c_]>thr for c_ in cluster):cluster.append(j);assigned[j]=True
            cf=st[cluster];cn=norms[cluster];ww=cn/(cn.sum()+1e-8)
            gf.append((cf.float()*ww.view(-1,*([1]*(cf.dim()-1)))).sum(0))
            fm.append([ass[i] for i in cluster])
        No=len(gf)
        mb=[];mbn={'w':[],'b':[],'m':[],'v':[]}
        for g,grp in enumerate(fm):
            idxs_=[sum(bbs[kk].features[ci[li]].weight.size(0) for kk in range(ck))+ci_ for ck,ci_ in grp]
            mb.append(torch.stack([ab[i] for i in idxs_]).mean())
            for c_ in 'wbmv': mbn[c_].append(torch.stack([abn[i][c_] for i in idxs_]).mean())
        nm=[{} for _ in range(K)]
        for g,grp in enumerate(fm):
            for ck,ci_ in grp: nm[ck][ci_]=g
        oc=bbs[0].features[ci[li]].weight.size(0)
        print(f"    Conv{li+1}: {pn}→{No} (from {K*oc})")
        lp_list.append({'Ni':pn,'No':No,'f':torch.stack(gf),'b':torch.stack(mb),
                   'bw':torch.stack(mbn['w']),'bb':torch.stack(mbn['b']),
                   'bm':torch.stack(mbn['m']),'bv':torch.stack(mbn['v'])})
        alm.append(nm);pm=nm;pn=No
    Nf=pn;fi=Nf*4;mfw=torch.zeros(fd,fi);mfb=torch.zeros(fd);fcc=torch.zeros(fi)
    for k,bb in enumerate(bbs):
        fw=bb.fc.weight.data.cpu();fb=bb.fc.bias.data.cpu();c4=bb.channels[3]
        for l in range(c4):
            if l not in pm[k]:continue
            g=pm[k][l];mfw[:,g*4:(g+1)*4]+=fw[:,l*4:(l+1)*4];fcc[g*4:(g+1)*4]+=1
        mfb+=fb/K
    fcc=fcc.clamp(min=1);mfw/=fcc.unsqueeze(0)
    chs=[l['No'] for l in lp_list]
    merged=Backbone(fd,chs)
    with torch.no_grad():
        for li_ in range(4):
            l=lp_list[li_];Ni=l['Ni'];No=l['No']
            merged.features[ci[li_]]=nn.Conv2d(Ni,No,3,padding=1)
            merged.features[ci[li_]].weight.copy_(l['f'][:,:Ni]);merged.features[ci[li_]].bias.copy_(l['b'])
            merged.features[bi[li_]]=nn.BatchNorm2d(No)
            merged.features[bi[li_]].weight.copy_(l['bw']);merged.features[bi[li_]].bias.copy_(l['bb'])
            merged.features[bi[li_]].running_mean.copy_(l['bm']);merged.features[bi[li_]].running_var.copy_(l['bv'])
        merged.fc=nn.Linear(fi,fd);merged.fc.weight.copy_(mfw);merged.fc.bias.copy_(mfb)
    print(f"    合并: {sum(p.numel() for p in merged.parameters()):,} params, ch={chs}")
    return merged.to(device)


# ═══════════════════════════════════════════════════════════
# 3. ★ 预计算: 收集所有推理信号
# ═══════════════════════════════════════════════════════════

def precompute_all(bbs, client_exps, union_bb, tl, etf, ccc, nc=10):
    """一次遍历测试集, 收集所有需要的信号"""
    K = len(bbs); ed = etf.to(device); union_bb.eval()

    # 构建样本量表: sample_count[k][c] = 训练样本数
    sample_count = {}
    for k in range(K):
        for c in client_exps[k]:
            sample_count[(k, c)] = ccc[k].get(c, 0)

    all_expert_errors = []   # (K, bs, nc) per batch
    all_union_logits = []    # (bs, nc) per batch
    all_labels = []

    with torch.no_grad():
        for x, y in tl:
            x_dev = x.to(device, non_blocking=True); bs = x.size(0)

            # Union logits
            f_union = F.normalize(union_bb(x_dev), dim=1)
            union_logits = torch.mm(f_union, ed.T)  # (bs, nc)
            all_union_logits.append(union_logits.cpu())

            # Expert errors
            batch_errors = torch.full((K, bs, nc), float('inf'))
            for k in range(K):
                f_k = bbs[k](x_dev)
                for c, exp in client_exps[k].items():
                    fr, _ = exp(f_k, ed[c].unsqueeze(0).expand(bs, -1))
                    batch_errors[k, :, c] = ((f_k - fr)**2).mean(1).cpu()

            all_expert_errors.append(batch_errors)
            all_labels.append(y)

    errors = torch.cat(all_expert_errors, dim=1)    # (K, N, nc)
    union_logits = torch.cat(all_union_logits, 0)    # (N, nc)
    labels = torch.cat(all_labels).numpy()
    N = len(labels)

    # Union 预测
    union_preds = union_logits.argmax(1).numpy()
    union_margin = torch.zeros(N)
    sorted_ul, _ = union_logits.sort(dim=1, descending=True)
    union_margin = (sorted_ul[:, 0] - sorted_ul[:, 1]).numpy()

    print(f"  预计算: {K} clients × {N} samples")
    print(f"  Union acc: {(union_preds == labels).mean():.2%}")

    return {
        'errors': errors,           # (K, N, nc)
        'union_logits': union_logits,  # (N, nc)
        'union_preds': union_preds,  # (N,)
        'union_margin': union_margin,  # (N,)
        'labels': labels,            # (N,)
        'sample_count': sample_count,  # {(k,c): int}
        'K': K, 'N': N, 'nc': nc,
    }


# ═══════════════════════════════════════════════════════════
# 4. ★ 策略 A: 改进 Expert 聚合
# ═══════════════════════════════════════════════════════════

def expert_original(data):
    """Baseline: 原始 min_k"""
    errors = data['errors'].clone()
    errors[errors == float('inf')] = 1e6
    best_err, _ = errors.min(dim=0)  # (N, nc)
    return best_err.argmin(1).numpy()


def expert_quality_filter(data, min_n=100):
    """A1: 质量过滤 — 只用训练 >= min_n 的 expert"""
    K, N, nc = data['errors'].shape
    sc = data['sample_count']
    filtered = torch.full((K, N, nc), float('inf'))
    for k in range(K):
        for c in range(nc):
            if sc.get((k, c), 0) >= min_n:
                filtered[k, :, c] = data['errors'][k, :, c]
    filtered[filtered == float('inf')] = 1e6
    best_err, _ = filtered.min(dim=0)
    preds = best_err.argmin(1).numpy()
    # 如果某个样本所有类都是 1e6, fallback 到 union
    all_inf = (best_err.min(1)[0] >= 1e5)
    preds[all_inf.numpy()] = data['union_preds'][all_inf.numpy()]
    return preds


def expert_weighted_avg(data, min_n=50):
    """A2: 加权平均 — 按 log(n+1) 加权误差, 而非取 min"""
    K, N, nc = data['errors'].shape
    sc = data['sample_count']
    weighted_err = torch.zeros(N, nc)
    weight_sum = torch.zeros(N, nc)
    for k in range(K):
        for c in range(nc):
            n = sc.get((k, c), 0)
            if n < min_n: continue
            w = np.log(n + 1)
            e = data['errors'][k, :, c].clone()
            e[e == float('inf')] = 1e6
            weighted_err[:, c] += w * e
            weight_sum[:, c] += w
    # 没有 expert 的类设为高误差
    no_expert = (weight_sum == 0)
    weight_sum[no_expert] = 1.0
    weighted_err[no_expert] = 1e6
    avg_err = weighted_err / weight_sum
    preds = avg_err.argmin(1).numpy()
    # fallback
    all_high = (avg_err.min(1)[0] >= 1e5)
    preds[all_high.numpy()] = data['union_preds'][all_high.numpy()]
    return preds


def expert_quality_min(data, min_n=100):
    """A3: 质量 min — 只在合格 expert 中取 min"""
    K, N, nc = data['errors'].shape
    sc = data['sample_count']
    best_err = torch.full((N, nc), 1e6)
    for k in range(K):
        for c in range(nc):
            if sc.get((k, c), 0) < min_n: continue
            e = data['errors'][k, :, c].clone()
            e[e == float('inf')] = 1e6
            best_err[:, c] = torch.min(best_err[:, c], e)
    preds = best_err.argmin(1).numpy()
    all_high = (best_err.min(1)[0] >= 1e5)
    preds[all_high.numpy()] = data['union_preds'][all_high.numpy()]
    return preds


def expert_top_quality(data):
    """A4: Top-Quality — 每类只用训练量最大的那个 expert"""
    K, N, nc = data['errors'].shape
    sc = data['sample_count']
    best_err = torch.full((N, nc), 1e6)
    for c in range(nc):
        # 找训练量最大的 client
        best_k = -1; best_n = 0
        for k in range(K):
            n = sc.get((k, c), 0)
            if n > best_n:
                best_n = n; best_k = k
        if best_k >= 0 and best_n > 0:
            e = data['errors'][best_k, :, c].clone()
            e[e == float('inf')] = 1e6
            best_err[:, c] = e
    preds = best_err.argmin(1).numpy()
    all_high = (best_err.min(1)[0] >= 1e5)
    preds[all_high.numpy()] = data['union_preds'][all_high.numpy()]
    return preds


def expert_weighted_min(data, min_n=50):
    """A5: 加权 min — 误差除以 log(n+1), 然后取 min
    高质量 expert 的误差被缩小, 更容易被选中"""
    K, N, nc = data['errors'].shape
    sc = data['sample_count']
    scaled_err = torch.full((K, N, nc), 1e6)
    for k in range(K):
        for c in range(nc):
            n = sc.get((k, c), 0)
            if n < min_n: continue
            w = np.log(n + 1)
            e = data['errors'][k, :, c].clone()
            e[e == float('inf')] = 1e6
            scaled_err[k, :, c] = e / w  # 质量越高, 误差缩放越大
    best_err, _ = scaled_err.min(dim=0)
    preds = best_err.argmin(1).numpy()
    all_high = (best_err.min(1)[0] >= 1e5)
    preds[all_high.numpy()] = data['union_preds'][all_high.numpy()]
    return preds


# ═══════════════════════════════════════════════════════════
# 5. ★ 策略 B: Union/Expert 路由
# ═══════════════════════════════════════════════════════════

def _get_expert_preds_and_margin(data, expert_fn=expert_original):
    """辅助: 获取 expert 预测 + 误差 margin"""
    errors = data['errors'].clone()
    errors[errors == float('inf')] = 1e6
    best_err, _ = errors.min(dim=0)  # (N, nc)

    expert_preds = best_err.argmin(1).numpy()

    sorted_err, _ = best_err.sort(dim=1)
    expert_margin = (sorted_err[:, 1] - sorted_err[:, 0]).numpy()  # top1-top2 误差差

    expert_best_err = sorted_err[:, 0].numpy()  # 最小误差

    return expert_preds, expert_margin, expert_best_err


def route_expert_margin(data, thr=0.0):
    """B1: Expert margin 路由 — expert margin 大时信 expert"""
    e_pred, e_margin, _ = _get_expert_preds_and_margin(data)
    use_expert = e_margin > thr
    return np.where(use_expert, e_pred, data['union_preds'])


def route_union_margin(data, thr=0.5):
    """B2: Union confidence 路由 — union margin 小时信 expert"""
    e_pred, _, _ = _get_expert_preds_and_margin(data)
    use_expert = data['union_margin'] < thr
    return np.where(use_expert, e_pred, data['union_preds'])


def route_dual(data, e_thr=0.0, u_thr=0.5):
    """B3: 双重信号 — expert margin 大 AND union margin 小"""
    e_pred, e_margin, _ = _get_expert_preds_and_margin(data)
    use_expert = (e_margin > e_thr) & (data['union_margin'] < u_thr)
    return np.where(use_expert, e_pred, data['union_preds'])


def route_per_class(data, expert_classes=None):
    """B4: 类别路由 — 某些类信 expert, 其他信 union
    用 oracle 知识选最优类（上界分析）"""
    labels = data['labels']
    e_pred, _, _ = _get_expert_preds_and_margin(data)
    u_pred = data['union_preds']
    nc = data['nc']

    # 每个类看 expert 和 union 谁好
    if expert_classes is None:
        expert_classes = []
        for c in range(nc):
            mask = (labels == c)
            e_acc = (e_pred[mask] == c).mean()
            u_acc = (u_pred[mask] == c).mean()
            if e_acc > u_acc:
                expert_classes.append(c)
        print(f"    B4 auto: expert 更优的类={expert_classes}")

    # 对 union 预测的类在 expert_classes 中的样本, 用 expert
    preds = u_pred.copy()
    # 对 expert 预测在 expert_classes 中的, 用 expert 预测
    for n in range(data['N']):
        if e_pred[n] in expert_classes:
            preds[n] = e_pred[n]
    return preds


def route_per_class_oracle(data):
    """B4-Oracle: 用真实标签决定每类用谁（上界）"""
    labels = data['labels']
    e_pred, _, _ = _get_expert_preds_and_margin(data)
    u_pred = data['union_preds']
    nc = data['nc']

    best_classes = []
    for c in range(nc):
        mask = (labels == c)
        e_acc = (e_pred[mask] == c).mean()
        u_acc = (u_pred[mask] == c).mean()
        if e_acc > u_acc:
            best_classes.append(c)

    preds = np.where(np.isin(labels, best_classes), e_pred, u_pred)
    return preds, best_classes


def route_error_threshold(data, thr=0.001):
    """B5: 误差绝对值路由 — expert 预测类误差很低时信 expert"""
    e_pred, _, e_best_err = _get_expert_preds_and_margin(data)
    use_expert = e_best_err < thr
    return np.where(use_expert, e_pred, data['union_preds'])


def route_ensemble_logits(data, alpha=1.0):
    """B6: Ensemble — 用 expert error 转化为 logits, 与 union logits 加权融合
    expert_logits[c] = -error[c] (误差越小 logit 越高)"""
    errors = data['errors'].clone()
    errors[errors == float('inf')] = 1e6
    best_err, _ = errors.min(dim=0)  # (N, nc)

    # Expert error → logits (归一化到与 union logits 可比的范围)
    expert_logits = -best_err  # (N, nc) 越高越好
    # 标准化
    e_mean = expert_logits.mean(dim=1, keepdim=True)
    e_std = expert_logits.std(dim=1, keepdim=True) + 1e-8
    expert_logits_norm = (expert_logits - e_mean) / e_std

    u_logits = data['union_logits']
    u_mean = u_logits.mean(dim=1, keepdim=True)
    u_std = u_logits.std(dim=1, keepdim=True) + 1e-8
    union_logits_norm = (u_logits - u_mean) / u_std

    combined = union_logits_norm + alpha * expert_logits_norm
    return combined.argmax(1).numpy()


def route_ensemble_quality(data, alpha=1.0, min_n=100):
    """B7: Quality Ensemble — 只用高质量 expert 的 error 做 logits"""
    K, N, nc = data['errors'].shape
    sc = data['sample_count']

    best_err = torch.full((N, nc), 1e6)
    for k in range(K):
        for c in range(nc):
            if sc.get((k, c), 0) < min_n: continue
            e = data['errors'][k, :, c].clone()
            e[e == float('inf')] = 1e6
            best_err[:, c] = torch.min(best_err[:, c], e)

    expert_logits = -best_err
    has_expert = (best_err < 1e5)
    # 没有 expert 的类设为 0 (不影响)
    expert_logits[~has_expert] = 0

    e_mean = expert_logits.mean(dim=1, keepdim=True)
    e_std = expert_logits.std(dim=1, keepdim=True) + 1e-8
    expert_logits_norm = (expert_logits - e_mean) / e_std
    expert_logits_norm[~has_expert] = 0  # 无 expert 的类不贡献

    u_logits = data['union_logits']
    u_mean = u_logits.mean(dim=1, keepdim=True)
    u_std = u_logits.std(dim=1, keepdim=True) + 1e-8
    union_logits_norm = (u_logits - u_mean) / u_std

    combined = union_logits_norm + alpha * expert_logits_norm
    return combined.argmax(1).numpy()


# ═══════════════════════════════════════════════════════════
# ★ 策略 C: 跨 Client 交叉验证
# ═══════════════════════════════════════════════════════════

def cross_client_voting(data, min_n=50):
    """C1: 各 client 独立预测, 投票
    每个 client 用自己的 expert 预测 (argmin error), 多数投票"""
    K, N, nc = data['errors'].shape
    sc = data['sample_count']
    votes = torch.zeros(N, nc)  # (N, nc) 投票计数
    for k in range(K):
        ek = data['errors'][k].clone(); ek[ek == float('inf')] = 1e6
        # 该 client 有效的类
        valid_classes = [c for c in range(nc) if sc.get((k, c), 0) >= min_n]
        if not valid_classes: continue
        # 只在有效类上 argmin
        ek_valid = torch.full((N, nc), 1e6)
        for c in valid_classes:
            ek_valid[:, c] = ek[:, c]
        preds_k = ek_valid.argmin(1)  # (N,) 该 client 的预测
        for n in range(N):
            votes[n, preds_k[n]] += 1
    # 投票最多的类; 平票 fallback union
    max_votes, vote_preds = votes.max(dim=1)
    preds = vote_preds.numpy()
    tie = (votes == max_votes.unsqueeze(1)).sum(1) > 1  # 有平票
    preds[tie.numpy()] = data['union_preds'][tie.numpy()]
    return preds


def cross_client_weighted_vote(data, min_n=50):
    """C1b: 加权投票 — 按 log(n+1) 加权"""
    K, N, nc = data['errors'].shape
    sc = data['sample_count']
    votes = torch.zeros(N, nc)
    for k in range(K):
        ek = data['errors'][k].clone(); ek[ek == float('inf')] = 1e6
        valid_classes = [c for c in range(nc) if sc.get((k, c), 0) >= min_n]
        if not valid_classes: continue
        ek_valid = torch.full((N, nc), 1e6)
        for c in valid_classes:
            ek_valid[:, c] = ek[:, c]
        preds_k = ek_valid.argmin(1)
        for n in range(N):
            c_pred = preds_k[n].item()
            w = np.log(sc.get((k, c_pred), 0) + 1)
            votes[n, c_pred] += w
    preds = votes.argmax(1).numpy()
    no_vote = (votes.sum(1) == 0)
    preds[no_vote.numpy()] = data['union_preds'][no_vote.numpy()]
    return preds


def cross_client_agreement(data, min_agree=2, min_n=50):
    """C2: 一致性路由 — ≥min_agree 个 client 同意时信 expert, 否则 union"""
    K, N, nc = data['errors'].shape
    sc = data['sample_count']
    votes = torch.zeros(N, nc)
    for k in range(K):
        ek = data['errors'][k].clone(); ek[ek == float('inf')] = 1e6
        valid_classes = [c for c in range(nc) if sc.get((k, c), 0) >= min_n]
        if not valid_classes: continue
        ek_valid = torch.full((N, nc), 1e6)
        for c in valid_classes:
            ek_valid[:, c] = ek[:, c]
        preds_k = ek_valid.argmin(1)
        for n in range(N):
            votes[n, preds_k[n]] += 1
    max_votes, vote_preds = votes.max(dim=1)
    preds = data['union_preds'].copy()
    agree = (max_votes >= min_agree).numpy()
    preds[agree] = vote_preds.numpy()[agree]
    return preds


def cross_client_exclusion(data, min_n=50):
    """C3: 交叉排斥 — client k 对类 c 的"拒绝度" = 其他类 expert 对该样本的接受度
    如果样本被多个不同类 expert 接受 → 可疑, 降低可信度"""
    K, N, nc = data['errors'].shape
    sc = data['sample_count']
    errors = data['errors'].clone(); errors[errors == float('inf')] = 1e6

    # 每个 (client, class) 的平均误差 → 作为 logit
    # 然后对每个样本, 看它被多少个不同类接受 (低误差)
    best_err, _ = errors.min(dim=0)  # (N, nc)

    # 计算每个类的"专属度": 该类误差相对其他类误差的比例
    # 专属度高 = 只有该类误差低, 其他类误差高
    sorted_err, _ = best_err.sort(dim=1)
    specificity = sorted_err[:, 1] / (sorted_err[:, 0] + 1e-10)  # top2/top1 比值
    # specificity 高 → 更专属 → 更可信

    expert_preds = best_err.argmin(1).numpy()
    # 专属度高时信 expert
    preds = data['union_preds'].copy()
    return expert_preds, specificity.numpy()


def cross_client_per_client_logits(data, alpha=1.0, min_n=50):
    """C4: 各 client 的 expert 误差转 logits → 加权融合 → 与 union 融合"""
    K, N, nc = data['errors'].shape
    sc = data['sample_count']

    # 每个 client 的 expert logits (只算有效类)
    ensemble_logits = torch.zeros(N, nc)
    weight_sum = torch.zeros(N, nc)
    for k in range(K):
        ek = data['errors'][k].clone(); ek[ek == float('inf')] = 1e6
        client_logits = torch.zeros(N, nc)
        for c in range(nc):
            n = sc.get((k, c), 0)
            if n < min_n: continue
            client_logits[:, c] = -ek[:, c]  # 误差取负
            weight_sum[:, c] += np.log(n + 1)
        # 按 client 内 normalize
        cl_valid = (client_logits != 0)
        if cl_valid.any():
            cm = client_logits.mean(dim=1, keepdim=True)
            cs = client_logits.std(dim=1, keepdim=True) + 1e-8
            client_logits_n = (client_logits - cm) / cs
            client_logits_n[~cl_valid] = 0
            ensemble_logits += client_logits_n * weight_sum.clamp(min=0)

    # normalize ensemble
    has_signal = (ensemble_logits.abs().sum(1) > 0)
    em = ensemble_logits.mean(dim=1, keepdim=True)
    es = ensemble_logits.std(dim=1, keepdim=True) + 1e-8
    en = (ensemble_logits - em) / es
    en[~has_signal.unsqueeze(1).expand_as(en)] = 0

    u = data['union_logits']
    um = u.mean(dim=1, keepdim=True); us = u.std(dim=1, keepdim=True) + 1e-8
    un = (u - um) / us

    return (un + alpha * en).argmax(1).numpy()


# ═══════════════════════════════════════════════════════════
# 6. 可视化
# ═══════════════════════════════════════════════════════════

def plot_results(results, save_path):
    fig, ax = plt.subplots(figsize=(16, 14))
    names = list(results.keys())
    accs = [results[n]*100 for n in names]
    colors = []
    for n in names:
        if 'Baseline' in n or 'Oracle' in n: colors.append('#607D8B')
        elif n.startswith('A'): colors.append('#2196F3')
        elif n.startswith('B'): colors.append('#4CAF50')
        elif n.startswith('C'): colors.append('#FF5722')
        else: colors.append('#795548')
    bars = ax.barh(range(len(names)), accs, color=colors, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('v14: 跨 Client 交叉验证', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2.,
                f'{acc:.2f}%', va='center', fontsize=7, fontweight='bold')
    ax.axvline(x=80, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  保存: {save_path}")


# ═══════════════════════════════════════════════════════════
# 7. 主实验
# ═══════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 80)
    print("MoE v14 — 跨 Client 交叉验证")
    print("=" * 80)

    NC=5; NL=10; ALPHA=0.05; FD=256; LD=32; EPB=600; EPE=600

    os.makedirs('outputs', exist_ok=True)
    etf = generate_etf(NL, FD)
    cal, ccl, tl, ccc = prepare_data(NC, ALPHA, NL)

    # Phase 1: 训练
    print(f"\n{'='*60}\n  Phase 1: 训练 (原始, 不改)\n{'='*60}")
    bbs = []; client_exps = []; t0 = time.time()
    for k in range(NC):
        cls = sorted(ccc[k].keys())
        print(f"\n  Client {k}: {len(cls)} cls, {sum(ccc[k].values())} samp")
        bb = Backbone(FD)
        bb = train_bb(bb, cal[k], cls, etf, EPB)
        exps = train_experts(bb, ccl[k], cls, etf, NL, FD, LD, EPE)
        for c in cls:
            mu, _ = compute_stats(bb, exps[c], ccl[k][c], etf[c])
            print(f"    c{c}: n={ccc[k][c]:5d} μ={mu:.6f}")
        bbs.append(bb); client_exps.append(exps)
    tt = time.time() - t0
    print(f"\n  训练: {tt:.1f}s")

    # Phase 2: Union
    print(f"\n{'='*60}\n  Phase 2: Union\n{'='*60}")
    ubb = union_aggregate(bbs, FD, 0.95)

    # Phase 3: 预计算
    print(f"\n{'='*60}\n  Phase 3: 预计算\n{'='*60}")
    data = precompute_all(bbs, client_exps, ubb, tl, etf, ccc, NL)
    labels = data['labels']
    N = data['N']

    # Phase 4: 评估
    print(f"\n{'='*60}\n  Phase 4: 策略评估\n{'='*60}")
    R = {}

    # ── Baselines ──
    print(f"\n  --- Baselines ---")
    R['Baseline: Union'] = (data['union_preds'] == labels).mean()
    print(f"  Union:                 {R['Baseline: Union']:.2%}")

    preds = expert_original(data)
    R['Baseline: Expert(min)'] = (preds == labels).mean()
    print(f"  Expert (original min): {R['Baseline: Expert(min)']:.2%}")

    # Oracle
    e_pred, _, _ = _get_expert_preds_and_margin(data)
    u_correct = (data['union_preds'] == labels)
    e_correct = (e_pred == labels)
    R['Oracle: U∪E'] = (u_correct | e_correct).mean()
    print(f"  Oracle (U∪E):          {R['Oracle: U∪E']:.2%}")

    # ══════════════════════════════════════════════════
    # ★ A: Expert 聚合改进
    # ══════════════════════════════════════════════════
    print(f"\n  --- ★ A: Expert 聚合改进 ---")

    for min_n in [20, 50, 100, 200, 500]:
        preds = expert_quality_filter(data, min_n=min_n)
        R[f'A1 Filter n≥{min_n}'] = (preds == labels).mean()
        print(f"  A1 Filter n≥{min_n:3d}:      {R[f'A1 Filter n≥{min_n}']:.2%}")

    for min_n in [20, 50, 100]:
        preds = expert_weighted_avg(data, min_n=min_n)
        R[f'A2 WeightAvg n≥{min_n}'] = (preds == labels).mean()
        print(f"  A2 WeightAvg n≥{min_n:3d}:   {R[f'A2 WeightAvg n≥{min_n}']:.2%}")

    for min_n in [20, 50, 100, 200]:
        preds = expert_quality_min(data, min_n=min_n)
        R[f'A3 QualMin n≥{min_n}'] = (preds == labels).mean()
        print(f"  A3 QualMin n≥{min_n:3d}:     {R[f'A3 QualMin n≥{min_n}']:.2%}")

    preds = expert_top_quality(data)
    R['A4 TopQuality'] = (preds == labels).mean()
    print(f"  A4 TopQuality:         {R['A4 TopQuality']:.2%}")

    for min_n in [20, 50, 100]:
        preds = expert_weighted_min(data, min_n=min_n)
        R[f'A5 WeightMin n≥{min_n}'] = (preds == labels).mean()
        print(f"  A5 WeightMin n≥{min_n:3d}:   {R[f'A5 WeightMin n≥{min_n}']:.2%}")

    # ══════════════════════════════════════════════════
    # ★ B: Union/Expert 路由
    # ══════════════════════════════════════════════════
    print(f"\n  --- ★ B: Union/Expert 路由 ---")

    # B1: Expert margin
    # 先看 margin 分布
    _, e_margin, e_best_err = _get_expert_preds_and_margin(data)
    print(f"  Expert margin: mean={e_margin.mean():.6f}, "
          f"p10={np.percentile(e_margin,10):.6f}, p50={np.percentile(e_margin,50):.6f}, "
          f"p90={np.percentile(e_margin,90):.6f}")
    for thr in np.percentile(e_margin, [30, 50, 70, 80, 90]):
        preds = route_expert_margin(data, thr=thr)
        acc = (preds == labels).mean()
        tag = f'B1 ExMargin>{thr:.6f}'
        R[tag] = acc
        n_expert = (e_margin > thr).sum()
        print(f"  {tag}: {acc:.2%} (n_expert={n_expert})")

    # B2: Union margin
    print(f"  Union margin: mean={data['union_margin'].mean():.4f}, "
          f"p10={np.percentile(data['union_margin'],10):.4f}, "
          f"p50={np.percentile(data['union_margin'],50):.4f}")
    for thr in np.percentile(data['union_margin'], [10, 20, 30, 40, 50]):
        preds = route_union_margin(data, thr=thr)
        acc = (preds == labels).mean()
        tag = f'B2 UnionMgn<{thr:.3f}'
        R[tag] = acc
        n_expert = (data['union_margin'] < thr).sum()
        print(f"  {tag}: {acc:.2%} (n_expert={n_expert})")

    # B3: Dual
    print(f"\n  B3: Dual routing (扫参数)")
    best_b3 = 0; best_b3_tag = ""
    for e_p in [30, 50, 70]:
        for u_p in [20, 30, 40]:
            e_thr = np.percentile(e_margin, e_p)
            u_thr = np.percentile(data['union_margin'], u_p)
            preds = route_dual(data, e_thr=e_thr, u_thr=u_thr)
            acc = (preds == labels).mean()
            if acc > best_b3:
                best_b3 = acc
                best_b3_tag = f'B3 Dual e>{e_p}%,u<{u_p}%'
    R[best_b3_tag] = best_b3
    print(f"  {best_b3_tag}: {best_b3:.2%}")

    # B4: Per-class (auto)
    preds = route_per_class(data)
    R['B4 PerClass(auto)'] = (preds == labels).mean()
    print(f"  B4 PerClass(auto):     {R['B4 PerClass(auto)']:.2%}")

    # B4-Oracle
    preds_oracle, best_classes = route_per_class_oracle(data)
    R['B4 PerClass(oracle)'] = (preds_oracle == labels).mean()
    print(f"  B4 PerClass(oracle):   {R['B4 PerClass(oracle)']:.2%} classes={best_classes}")

    # B5: Error threshold
    for pct in [50, 60, 70, 80, 90]:
        thr = np.percentile(e_best_err, pct)
        preds = route_error_threshold(data, thr=thr)
        acc = (preds == labels).mean()
        tag = f'B5 ErrThr p{pct}'
        R[tag] = acc
        print(f"  {tag}: {acc:.2%} (thr={thr:.6f})")

    # B6: Ensemble
    for alpha in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
        preds = route_ensemble_logits(data, alpha=alpha)
        R[f'B6 Ensemble α={alpha}'] = (preds == labels).mean()
        print(f"  B6 Ensemble α={alpha}:    {R[f'B6 Ensemble α={alpha}']:.2%}")

    # B7: Quality Ensemble
    for alpha in [0.3, 0.5, 1.0, 2.0]:
        for min_n in [50, 100, 200]:
            preds = route_ensemble_quality(data, alpha=alpha, min_n=min_n)
            tag = f'B7 QualEns α={alpha} n≥{min_n}'
            R[tag] = (preds == labels).mean()
            print(f"  {tag}: {R[tag]:.2%}")

    # ══════════════════════════════════════════════════
    # ★ C: 跨 Client 交叉验证
    # ══════════════════════════════════════════════════
    print(f"\n  --- ★ C: 跨 Client 交叉验证 ---")

    # C1: 投票
    for min_n in [0, 20, 50, 100]:
        preds = cross_client_voting(data, min_n=min_n)
        tag = f'C1 Vote n≥{min_n}'
        R[tag] = (preds == labels).mean()
        print(f"  {tag}: {R[tag]:.2%}")

    # C1b: 加权投票
    for min_n in [0, 50, 100]:
        preds = cross_client_weighted_vote(data, min_n=min_n)
        tag = f'C1b WVote n≥{min_n}'
        R[tag] = (preds == labels).mean()
        print(f"  {tag}: {R[tag]:.2%}")

    # C2: 一致性路由
    for min_agree in [2, 3]:
        for min_n in [0, 50]:
            preds = cross_client_agreement(data, min_agree=min_agree, min_n=min_n)
            tag = f'C2 Agree≥{min_agree} n≥{min_n}'
            R[tag] = (preds == labels).mean()
            print(f"  {tag}: {R[tag]:.2%}")

    # C3: 交叉排斥 (specificity routing)
    e_pred_c3, specificity = cross_client_exclusion(data)
    for pct in [50, 60, 70, 80, 90]:
        thr = np.percentile(specificity, pct)
        preds = np.where(specificity > thr, e_pred_c3, data['union_preds'])
        tag = f'C3 Specificity p{pct}'
        R[tag] = (preds == labels).mean()
        print(f"  {tag}: {R[tag]:.2%} (thr={thr:.2f})")

    # C4: Per-client logits ensemble
    for alpha in [0.3, 0.5, 1.0, 2.0, 5.0]:
        for min_n in [0, 50, 100]:
            preds = cross_client_per_client_logits(data, alpha=alpha, min_n=min_n)
            tag = f'C4 PCEns α={alpha} n≥{min_n}'
            R[tag] = (preds == labels).mean()
            print(f"  {tag}: {R[tag]:.2%}")

    # ══════════════════════════════════════════════════
    # 汇总
    # ══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"★ 最终结果 (Top 30)")
    print(f"{'='*70}")
    sorted_r = sorted(R.items(), key=lambda x: -x[1])
    baseline_u = R['Baseline: Union']
    baseline_e = R['Baseline: Expert(min)']
    for i, (name, acc) in enumerate(sorted_r[:30]):
        diff_u = acc - baseline_u
        diff_e = acc - baseline_e
        cat = ""
        if name.startswith('A'): cat = "[Agg]  "
        elif name.startswith('B'): cat = "[Route]"
        elif name.startswith('C'): cat = "[Cross]"
        marker = " ★" if acc > max(baseline_u, baseline_e) else ""
        print(f"  {i+1:2d}. {cat:8s} {name:<40s} | {acc:>8.2%} "
              f"(vs U {'+' if diff_u>=0 else ''}{diff_u*100:.2f}, "
              f"vs E {'+' if diff_e>=0 else ''}{diff_e*100:.2f}){marker}")

    # 按类别最佳
    print(f"\n  --- 各策略类最佳 ---")
    for prefix, label in [('A', 'Expert聚合'), ('B', 'Union/Expert路由'), ('C', '跨Client交叉')]:
        cat_r = {k:v for k,v in R.items() if k.startswith(prefix)}
        if cat_r:
            best_n, best_a = max(cat_r.items(), key=lambda x: x[1])
            diff_u = best_a - baseline_u
            print(f"  {label:20s}: {best_n:40s} {best_a:.2%} (vs Union {'+' if diff_u>=0 else ''}{diff_u*100:.2f}pp)")

    plot_results(dict(sorted_r), 'outputs/v14_cross.png')
    print(f"\n  训练: {tt:.1f}s\n完成!")


if __name__ == "__main__":
    main()