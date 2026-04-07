"""
ablation_fusion.py
==================
消融实验: Filter Merge vs 各种特征/logits融合方法

训练只做一次, 然后用不同的 union 方式计算 union_logits,
分别测试 union 单独准确率 和 C4 融合准确率.

用法:
  python ablation_fusion.py --alpha 0.05 --gpu 0
  python ablation_fusion.py --alpha 0.1 --gpu 0
  python ablation_fusion.py --alpha 0.3 --gpu 0
  python ablation_fusion.py --alpha 0.5 --gpu 1
  python ablation_fusion.py --alpha 1.0 --gpu 1

结果保存到 results/ablation_fusion_a{alpha}_s{seed}.json
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import warnings
import json
import time
import os
import argparse
from collections import defaultdict

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════
# 复用 rebuild8 的训练组件 (一字不改)
# ═══════════════════════════════════════════════════════════

DL_KWARGS = dict(num_workers=8, pin_memory=True, persistent_workers=True)

def setup_device(gpu):
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42); np.random.seed(42)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device

USE_BF16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

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
    return M

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

def train_bb(bb,loader,classes,etf,device,epochs=600,lr=1e-3):
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
        if (ep+1)%200==0: print(f"      BB {ep+1}/{epochs} loss={el/max(nb,1):.4f}")
    return bb

def preextract(bb,dl,device):
    bb.eval();af=[]
    with torch.no_grad():
        amp=(torch.amp.autocast('cuda',dtype=torch.bfloat16) if USE_BF16
             else torch.amp.autocast('cuda',enabled=False))
        with amp:
            for x,_ in dl: af.append(bb(x.to(device,non_blocking=True)).float())
    return torch.cat(af,0)

def train_exp(exp,cached,eo,ed,others,device,fdim=256,epochs=600,lr=1e-3,margin=0.05):
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

def train_experts(bb,cls_loaders,classes,etf,device,nc=10,fdim=256,ldim=32,epochs=600,lr=1e-3):
    bb.eval();ed=etf.to(device);exps={}
    om={c:torch.tensor([k for k in range(nc) if k!=c],device=device) for c in range(nc)}
    for cls in classes:
        cached=preextract(bb,cls_loaders[cls],device)
        exp=ConditionalExpert(fdim,fdim,128,ldim).to(device)
        exp=train_exp(exp,cached,ed[cls],ed,om[cls],device,fdim,epochs,lr)
        exps[cls]=exp
    return exps


# ═══════════════════════════════════════════════════════════
# Filter merge (完全复用 rebuild8)
# ═══════════════════════════════════════════════════════════

def union_aggregate(bbs, fd=256, thr=0.95, device=None):
    K=len(bbs)
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
    return merged.to(device)


# ═══════════════════════════════════════════════════════════
# ★ 各种 Union Logits 计算方法
# ═══════════════════════════════════════════════════════════

def compute_union_logits_filter_merge(bbs, ubb, x, etf_d):
    """方法 0: Filter Merge (原始方法)"""
    f = F.normalize(ubb(x), dim=1)
    return torch.mm(f, etf_d.T)


def compute_union_logits_avg(bbs, x, etf_d):
    """方法 1: 简单 logits 平均"""
    all_logits = []
    for bb in bbs:
        f = F.normalize(bb(x), dim=1)
        all_logits.append(torch.mm(f, etf_d.T))
    return torch.stack(all_logits).mean(0)


def compute_union_logits_class_weighted(bbs, x, etf_d, ccc, nc):
    """方法 2: 按类加权 logits
    Client k 对类 c 的 logit 按 log(n_kc + 1) 加权"""
    K = len(bbs); bs = x.size(0)
    weighted_logits = torch.zeros(bs, nc, device=x.device)
    weight_sum = torch.zeros(nc, device=x.device)
    for k, bb in enumerate(bbs):
        f = F.normalize(bb(x), dim=1)
        logits_k = torch.mm(f, etf_d.T)  # (bs, nc)
        for c in range(nc):
            n = ccc.get(k, {}).get(c, 0)
            if n == 0: continue
            w = np.log(n + 1)
            weighted_logits[:, c] += logits_k[:, c] * w
            weight_sum[c] += w
    weight_sum = weight_sum.clamp(min=1e-8)
    return weighted_logits / weight_sum.unsqueeze(0)


def compute_union_logits_max(bbs, x, etf_d):
    """方法 3: Max logits — 每个类取所有 client 中的最大 logit"""
    all_logits = []
    for bb in bbs:
        f = F.normalize(bb(x), dim=1)
        all_logits.append(torch.mm(f, etf_d.T))
    return torch.stack(all_logits).max(0)[0]


def compute_union_logits_expert_guided(bbs, x, etf_d, client_exps, ccc, nc):
    """方法 4: Expert-guided — 用 expert 重建误差衡量每个 client 的可信度
    误差低 → 该 client 对该样本理解好 → 权重高"""
    K = len(bbs); bs = x.size(0); ed = etf_d
    weighted_logits = torch.zeros(bs, nc, device=x.device)
    weight_sum = torch.zeros(bs, nc, device=x.device)
    for k, bb in enumerate(bbs):
        f_k = F.normalize(bb(x), dim=1)
        logits_k = torch.mm(f_k, ed.T)  # (bs, nc)
        # 计算该 client 对每个类的重建误差
        for c, exp in client_exps[k].items():
            fr, _ = exp(f_k, ed[c].unsqueeze(0).expand(bs, -1))
            err = ((f_k - fr)**2).mean(1)  # (bs,)
            w = 1.0 / (err + 1e-6)  # 误差越低, 权重越高
            weighted_logits[:, c] += logits_k[:, c] * w
            weight_sum[:, c] += w
    weight_sum = weight_sum.clamp(min=1e-8)
    return weighted_logits / weight_sum


def compute_union_logits_topk(bbs, x, etf_d, client_exps, nc, topk=2):
    """方法 5: Top-K 选择 — 每个样本只用 expert 误差最低的 K 个 client"""
    K = len(bbs); bs = x.size(0); ed = etf_d
    # 计算每个 client 对每个样本的平均重建误差
    client_errors = torch.full((K, bs), 1e6, device=x.device)
    client_feats = []
    for k, bb in enumerate(bbs):
        f_k = F.normalize(bb(x), dim=1)
        client_feats.append(f_k)
        errs = []
        for c, exp in client_exps[k].items():
            fr, _ = exp(f_k, ed[c].unsqueeze(0).expand(bs, -1))
            errs.append(((f_k - fr)**2).mean(1))
        if errs:
            client_errors[k] = torch.stack(errs).mean(0)

    # 选 top-k (误差最低的)
    _, topk_idx = client_errors.topk(min(topk, K), dim=0, largest=False)  # (topk, bs)
    logits_sum = torch.zeros(bs, nc, device=x.device)
    for t in range(topk_idx.size(0)):
        for b in range(bs):
            k = topk_idx[t, b].item()
            logits_sum[b] += torch.mm(client_feats[k][b:b+1], ed.T).squeeze(0)
    return logits_sum / topk_idx.size(0)


def compute_union_logits_attention(bbs, x, etf_d):
    """方法 6: Attention fusion (FAFI 风格)
    用噪声特征作为参考: 离噪声远的特征信息量高, 权重大"""
    K = len(bbs); bs = x.size(0)
    all_feats = []
    all_logits = []
    for bb in bbs:
        f = F.normalize(bb(x), dim=1)
        all_feats.append(f)
        all_logits.append(torch.mm(f, etf_d.T))

    # 噪声参考
    noise = torch.randn(bs, all_feats[0].size(1), device=x.device)
    noise = F.normalize(noise, dim=1)

    # 计算每个 client 的注意力权重
    weights = []
    for f in all_feats:
        sim_to_noise = (f * noise).sum(1)  # (bs,) 与噪声的余弦相似度
        w = 1.0 - sim_to_noise  # 离噪声越远, 权重越高
        weights.append(w)
    weights = torch.stack(weights)  # (K, bs)
    weights = F.softmax(weights, dim=0)  # (K, bs) 归一化

    # 加权融合 logits
    result = torch.zeros_like(all_logits[0])
    for k in range(K):
        result += all_logits[k] * weights[k].unsqueeze(1)
    return result


# ═══════════════════════════════════════════════════════════
# ★ C4 融合 (用任意 union_logits + expert ensemble)
# ═══════════════════════════════════════════════════════════

def c4_with_union_logits(union_logits, errors, sample_count, labels,
                         alpha=0.3, min_n=100):
    """给定 union_logits, 计算 C4 融合结果"""
    N, nc = union_logits.shape
    K = errors.shape[0]
    sc = sample_count

    # Expert ensemble (和原始 C4 完全一致)
    ensemble = torch.zeros(N, nc)
    for k in range(K):
        ek = errors[k].clone(); ek[ek == float('inf')] = 1e6
        cl = torch.zeros(N, nc)
        valid_c = []
        for c in range(nc):
            n = sc.get((k, c), 0)
            if n < min_n: continue
            cl[:, c] = -ek[:, c]
            valid_c.append(c)
        if not valid_c: continue
        cl_mask = torch.zeros(N, nc, dtype=torch.bool)
        for c in valid_c: cl_mask[:, c] = True
        cm = cl.sum(1, keepdim=True) / cl_mask.sum(1, keepdim=True).clamp(min=1)
        diff = (cl - cm) * cl_mask.float()
        cs = ((diff**2).sum(1, keepdim=True)/cl_mask.sum(1, keepdim=True).clamp(min=1)).sqrt() + 1e-8
        cl_n = diff / cs; cl_n[~cl_mask] = 0
        w = np.log(sc.get((k, valid_c[0]), 0) + 1) if valid_c else 1.0
        ensemble += cl_n * w

    # Normalize
    has_signal = (ensemble.abs().sum(1) > 0)
    em = ensemble.mean(1, keepdim=True)
    es = ensemble.std(1, keepdim=True) + 1e-8
    en = (ensemble - em) / es
    en[~has_signal.unsqueeze(1).expand_as(en)] = 0

    u = union_logits
    um = u.mean(1, keepdim=True); us = u.std(1, keepdim=True) + 1e-8
    un = (u - um) / us

    preds = (un + alpha * en).argmax(1).numpy()
    acc = float((preds == labels).mean())
    return acc


# ═══════════════════════════════════════════════════════════
# ★ 主消融实验
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--n_clients', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs_bb', type=int, default=600)
    parser.add_argument('--epochs_exp', type=int, default=600)
    args = parser.parse_args()

    device = setup_device(args.gpu)
    NC = args.n_clients; NL = 10; FD = 256; LD = 32

    os.makedirs('results', exist_ok=True)
    out_path = f"results/ablation_fusion_a{args.alpha}_k{NC}_s{args.seed}.json"

    print(f"\n{'='*70}")
    print(f"  消融实验: Filter Merge vs 特征融合")
    print(f"  α={args.alpha}, K={NC}, seed={args.seed}")
    print(f"{'='*70}")

    # Phase 1: 训练 (只做一次)
    etf = generate_etf(NL, FD)
    cal, ccl, tl, ccc = prepare_data(NC, args.alpha, NL)

    print(f"\n  数据分布:")
    for k in range(NC):
        counts=[ccc[k].get(c,0) for c in range(NL)]
        n_cls = sum(1 for c in counts if c > 0)
        print(f"    Client {k}: {n_cls} cls, {sum(counts)} samp")

    bbs = []; client_exps = []
    t0 = time.time()
    for k in range(NC):
        cls = sorted(ccc[k].keys())
        print(f"\n  Client {k}: {len(cls)} cls, {sum(ccc[k].values())} samp")
        bb = Backbone(FD)
        bb = train_bb(bb, cal[k], cls, etf, device, args.epochs_bb)
        exps = train_experts(bb, ccl[k], cls, etf, device, NL, FD, LD, args.epochs_exp)
        bbs.append(bb); client_exps.append(exps)
    train_time = time.time() - t0
    print(f"\n  训练完成: {train_time:.0f}s")

    # Phase 2: Filter Merge
    print(f"\n{'='*60}")
    print(f"  Phase 2: Filter Merge")
    print(f"{'='*60}")
    ubb = union_aggregate(bbs, FD, 0.95, device)

    # Phase 3: 预计算 — 一次遍历收集所有信号
    print(f"\n{'='*60}")
    print(f"  Phase 3: 预计算所有信号")
    print(f"{'='*60}")

    ed = etf.to(device)
    # 构建 sample_count
    sample_count = {}
    for k in range(NC):
        for c in client_exps[k]:
            sample_count[(k, c)] = ccc[k].get(c, 0)

    all_union_logits_fm = []   # filter merge
    all_union_logits_avg = []  # 简单平均
    all_union_logits_cw = []   # 按类加权
    all_union_logits_max = []  # max
    all_union_logits_eg = []   # expert-guided
    all_union_logits_tk2 = []  # top-2
    all_union_logits_tk3 = []  # top-3
    all_union_logits_att = []  # attention
    all_client_logits = []     # ★ 每个 client 的 ETF logits: (K, bs, nc)
    all_errors = []
    all_labels = []

    ubb.eval()
    for bb in bbs: bb.eval()

    with torch.no_grad():
        for x, y in tl:
            x_d = x.to(device, non_blocking=True); bs = x.size(0)

            # ★ 收集每个 client 的 ETF logits
            batch_client_logits = torch.zeros(NC, bs, NL)
            for k in range(NC):
                f_k = F.normalize(bbs[k](x_d), dim=1)
                batch_client_logits[k] = torch.mm(f_k, ed.T).cpu()
            all_client_logits.append(batch_client_logits)

            # 方法 0: Filter Merge
            all_union_logits_fm.append(
                compute_union_logits_filter_merge(bbs, ubb, x_d, ed).cpu())

            # 方法 1: 简单 logits 平均
            all_union_logits_avg.append(
                compute_union_logits_avg(bbs, x_d, ed).cpu())

            # 方法 2: 按类加权 logits
            all_union_logits_cw.append(
                compute_union_logits_class_weighted(bbs, x_d, ed, ccc, NL).cpu())

            # 方法 3: Max logits
            all_union_logits_max.append(
                compute_union_logits_max(bbs, x_d, ed).cpu())

            # 方法 4: Expert-guided
            all_union_logits_eg.append(
                compute_union_logits_expert_guided(bbs, x_d, ed, client_exps, ccc, NL).cpu())

            # 方法 5a: Top-2
            all_union_logits_tk2.append(
                compute_union_logits_topk(bbs, x_d, ed, client_exps, NL, topk=2).cpu())

            # 方法 5b: Top-3
            all_union_logits_tk3.append(
                compute_union_logits_topk(bbs, x_d, ed, client_exps, NL, topk=3).cpu())

            # 方法 6: Attention (FAFI 风格)
            all_union_logits_att.append(
                compute_union_logits_attention(bbs, x_d, ed).cpu())

            # Expert errors
            batch_errors = torch.full((NC, bs, NL), float('inf'))
            for k in range(NC):
                f_k = bbs[k](x_d)
                for c, exp in client_exps[k].items():
                    fr, _ = exp(f_k, ed[c].unsqueeze(0).expand(bs, -1))
                    batch_errors[k, :, c] = ((f_k - fr)**2).mean(1).cpu()
            all_errors.append(batch_errors)
            all_labels.append(y)

    errors = torch.cat(all_errors, dim=1)
    labels = torch.cat(all_labels).numpy()
    N = len(labels)

    methods = {
        'filter_merge':    torch.cat(all_union_logits_fm, 0),
        'logits_avg':      torch.cat(all_union_logits_avg, 0),
        'class_weighted':  torch.cat(all_union_logits_cw, 0),
        'max_logits':      torch.cat(all_union_logits_max, 0),
        'expert_guided':   torch.cat(all_union_logits_eg, 0),
        'top2_select':     torch.cat(all_union_logits_tk2, 0),
        'top3_select':     torch.cat(all_union_logits_tk3, 0),
        'attention_fafi':  torch.cat(all_union_logits_att, 0),
    }

    # ★ 收集 per-client logits
    client_logits = torch.cat(all_client_logits, dim=1)  # (K, N, nc)

    # Expert min (不依赖 union)
    errs_clean = errors.clone(); errs_clean[errs_clean == float('inf')] = 1e6
    expert_preds = errs_clean.min(0)[0].argmin(1).numpy()
    expert_min_acc = float((expert_preds == labels).mean())

    # ═══════════════════════════════════════════════════════════
    # ★ No-Merge 方法: 不合并模型, 同时利用 ETF logits + Expert error
    # ═══════════════════════════════════════════════════════════

    def znorm(x, dim=1):
        m = x.mean(dim, keepdim=True); s = x.std(dim, keepdim=True) + 1e-8
        return (x - m) / s

    no_merge_methods = {}

    # --- B1: Per-client 内部融合 (ETF logits + expert error), 然后跨 client 加权聚合 ---
    for beta in [0.3, 0.5, 1.0, 2.0]:
        scores = torch.zeros(N, NL)
        for k in range(NC):
            lk = client_logits[k]  # (N, nc)  ETF logits
            ek = errors[k].clone(); ek[ek == float('inf')] = 1e6
            ek_neg = -ek  # (N, nc)  误差取负

            # 该 client 有效的类
            valid_c = [c for c in range(NL) if sample_count.get((k, c), 0) > 0]
            if not valid_c: continue

            # z-score normalize (只在有效类上)
            mask = torch.zeros(N, NL, dtype=torch.bool)
            for c in valid_c: mask[:, c] = True

            lk_n = znorm(lk) * mask.float()
            ek_n = znorm(ek_neg) * mask.float()

            # Client 内融合: ETF + β × Expert
            combined_k = lk_n + beta * ek_n

            # 跨 client 加权
            w = sum(np.log(sample_count.get((k, c), 0) + 1) for c in valid_c)
            scores += combined_k * w

        preds = scores.argmax(1).numpy()
        acc = float((preds == labels).mean())
        no_merge_methods[f'B1_perclient_β={beta}'] = acc

    # --- B2: Per-client 概率乘积 (softmax(logits) × softmax(-error/τ)) ---
    for tau in [0.01, 0.1, 1.0]:
        joint_scores = torch.zeros(N, NL)
        for k in range(NC):
            lk = client_logits[k]  # (N, nc)
            ek = errors[k].clone(); ek[ek == float('inf')] = 1e6

            valid_c = [c for c in range(NL) if sample_count.get((k, c), 0) > 0]
            if not valid_c: continue

            # Softmax on valid classes only
            lk_masked = torch.full((N, NL), -1e6)
            ek_masked = torch.full((N, NL), -1e6)
            for c in valid_c:
                lk_masked[:, c] = lk[:, c]
                ek_masked[:, c] = -ek[:, c] / tau

            prob_etf = F.softmax(lk_masked, dim=1)
            prob_exp = F.softmax(ek_masked, dim=1)
            joint = prob_etf * prob_exp  # (N, nc) 乘性融合

            w = sum(np.log(sample_count.get((k, c), 0) + 1) for c in valid_c)
            joint_scores += joint * w

        preds = joint_scores.argmax(1).numpy()
        acc = float((preds == labels).mean())
        no_merge_methods[f'B2_product_τ={tau}'] = acc

    # --- B3: Expert-gated ETF logits ---
    # 每个 client: 如果 expert error 低于该 client 该类的中位数 → 用 ETF logit, 否则抑制
    for gate_pct in [30, 50, 70]:
        gated_scores = torch.zeros(N, NL)
        gated_weight = torch.zeros(NL)
        for k in range(NC):
            lk = client_logits[k]  # (N, nc)
            ek = errors[k].clone(); ek[ek == float('inf')] = 1e6

            valid_c = [c for c in range(NL) if sample_count.get((k, c), 0) > 0]
            if not valid_c: continue

            for c in valid_c:
                thr = np.percentile(ek[:, c].numpy(), gate_pct)
                gate = (ek[:, c] < thr).float()  # (N,) 低误差的样本通过
                w = np.log(sample_count.get((k, c), 0) + 1)
                gated_scores[:, c] += lk[:, c] * gate * w
                gated_weight[c] += w * gate.mean().item()

        gated_weight = gated_weight.clamp(min=1e-8)
        gated_scores = gated_scores / gated_weight.unsqueeze(0)
        preds = gated_scores.argmax(1).numpy()
        acc = float((preds == labels).mean())
        no_merge_methods[f'B3_gated_p{gate_pct}'] = acc

    # --- B4: Client 最佳类路由 ---
    # 每个类只用训练量最大的那个 client 的 ETF logit
    best_client_logits = torch.zeros(N, NL)
    for c in range(NL):
        best_k = -1; best_n = 0
        for k in range(NC):
            n = sample_count.get((k, c), 0)
            if n > best_n: best_n = n; best_k = k
        if best_k >= 0:
            best_client_logits[:, c] = client_logits[best_k, :, c]
    preds = best_client_logits.argmax(1).numpy()
    no_merge_methods['B4_best_client_per_cls'] = float((preds == labels).mean())

    # --- B5: Expert error 做 per-sample client 权重, 然后加权 ETF logits ---
    # 类似 FAFI 但用 expert error 而非噪声做 informativeness
    for min_n in [0, 50, 100]:
        weighted_logits = torch.zeros(N, NL)
        weight_sum = torch.zeros(N, 1)
        for k in range(NC):
            lk = client_logits[k]
            ek = errors[k].clone(); ek[ek == float('inf')] = 1e6

            valid_c = [c for c in range(NL) if sample_count.get((k, c), 0) >= min_n]
            if not valid_c: continue

            # 该 client 对该样本的整体可信度 = 有效类上平均误差的倒数
            avg_err = torch.stack([ek[:, c] for c in valid_c]).mean(0)  # (N,)
            w = (1.0 / (avg_err + 1e-6)).unsqueeze(1)  # (N, 1)
            weighted_logits += lk * w
            weight_sum += w

        weight_sum = weight_sum.clamp(min=1e-8)
        final_logits = weighted_logits / weight_sum
        preds = final_logits.argmax(1).numpy()
        acc = float((preds == labels).mean())
        no_merge_methods[f'B5_err_weight_n≥{min_n}'] = acc

    # Phase 4: 评估
    print(f"\n{'='*70}")
    print(f"  Phase 4: 消融结果")
    print(f"{'='*70}")
    print(f"  {'方法':<22s} | {'Union 单独':>10s} | {'C4 α=0.2':>10s} | {'C4 α=0.3':>10s} | {'C4 α=0.5':>10s} | {'C4 best':>10s}")
    print(f"  {'-'*88}")

    results = {
        'alpha': args.alpha, 'n_clients': NC, 'seed': args.seed,
        'train_time': train_time,
        'expert_min': expert_min_acc,
        'methods': {},
    }

    # Expert only (无 union, 作为 baseline)
    print(f"  {'expert_only':<22s} | {'—':>10s} | {'—':>10s} | {'—':>10s} | {'—':>10s} | {expert_min_acc:>9.2%}")
    results['methods']['expert_only'] = {
        'union_acc': None,
        'c4_best': expert_min_acc,
    }

    for name, ul in methods.items():
        union_acc = float((ul.argmax(1).numpy() == labels).mean())
        c4_accs = {}
        for af in [0.2, 0.3, 0.5, 1.0]:
            c4_accs[af] = c4_with_union_logits(ul, errors, sample_count, labels,
                                                alpha=af, min_n=100)
        best_af = max(c4_accs, key=c4_accs.get)
        best_c4 = c4_accs[best_af]

        print(f"  {name:<22s} | {union_acc:>9.2%} | {c4_accs.get(0.2,0):>9.2%} | "
              f"{c4_accs.get(0.3,0):>9.2%} | {c4_accs.get(0.5,0):>9.2%} | {best_c4:>9.2%}")

        results['methods'][name] = {
            'union_acc': union_acc,
            'c4_a0.2': c4_accs.get(0.2),
            'c4_a0.3': c4_accs.get(0.3),
            'c4_a0.5': c4_accs.get(0.5),
            'c4_a1.0': c4_accs.get(1.0),
            'c4_best': best_c4,
            'c4_best_alpha': best_af,
        }

    # ── No-Merge 方法 (直接预测, 不经过 C4) ──
    print(f"\n  {'='*70}")
    print(f"  No-Merge 方法 (ETF logits + Expert error, 不合并模型)")
    print(f"  {'='*70}")
    print(f"  {'方法':<28s} | {'准确率':>10s}")
    print(f"  {'-'*42}")

    for name, acc in sorted(no_merge_methods.items(), key=lambda x: -x[1]):
        print(f"  {name:<28s} | {acc:>9.2%}")
        results['methods'][name] = {
            'union_acc': None,
            'c4_best': acc,
            'type': 'no_merge',
        }

    # 排序总结 (所有方法一起排)
    print(f"\n  {'='*70}")
    print(f"  总排名 (所有方法):")
    print(f"  {'='*70}")
    fm_best = results['methods']['filter_merge']['c4_best']
    ranked = sorted(results['methods'].items(), key=lambda x: -x[1]['c4_best'])
    for i, (name, r) in enumerate(ranked):
        u_str = f"{r['union_acc']:.2%}" if r.get('union_acc') is not None else "—"
        is_no_merge = r.get('type') == 'no_merge'
        tag = "[NoMerge]" if is_no_merge else ("[Expert]" if name == 'expert_only' else "[Union+C4]")
        gap = r['c4_best'] - fm_best
        gap_str = f"{gap:+.2%}" if name != 'filter_merge' else "baseline"
        print(f"  {i+1:2d}. {tag:10s} {name:<28s} | {r['c4_best']:.2%} | vs FM: {gap_str}")

    # 保存
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"  训练: {train_time:.0f}s, 总计: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()