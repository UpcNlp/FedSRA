"""
MoE One-Shot FL v2-Union+ — 4 方案验证

基于 v2-Union (78.65% ETF-Proto) 的改进:
  方向 1: Per-client FC 通道选路 — 从 Union conv 中选出各 client 的通道, 用原始 FC
  方向 2: Union ETF-Proto + Expert 联合评分 — 两组信号加权组合
  方向 3: 跨 client Expert 利用 — 所有 expert 在所有 client 的路由特征上评估
  方向 4: 改进投票策略 — 加权投票 / 过滤不可信组合

运行: python moe_etf_v2_union_plus.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import copy
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
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"BF16: {'ON' if USE_BF16 else 'OFF'}")
print("=" * 80)

DL_KWARGS = dict(num_workers=8, pin_memory=True, persistent_workers=True)


# ═══════════════════════════════════════════════════════════
# 1. 数据准备 / ETF / Backbone / Expert / Loss / 训练
#    (与 v2 完全相同, 压缩展示)
# ═══════════════════════════════════════════════════════════

def dirichlet_split(dataset, n_clients, alpha, n_classes=10):
    targets = np.array(dataset.targets)
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)
    client_indices = defaultdict(list)
    client_class_counts = defaultdict(lambda: defaultdict(int))
    for class_id in range(n_classes):
        indices = np.array(class_indices[class_id])
        np.random.shuffle(indices)
        proportions = np.random.dirichlet([alpha] * n_clients)
        proportions = (proportions * len(indices)).astype(int)
        proportions[-1] = len(indices) - proportions[:-1].sum()
        start = 0
        for client_id in range(n_clients):
            end = start + proportions[client_id]
            if end > start:
                client_indices[client_id].extend(indices[start:end].tolist())
                client_class_counts[client_id][class_id] = proportions[client_id]
            start = end
    return dict(client_indices), dict(client_class_counts)


def prepare_data(n_clients=5, alpha=0.05, n_classes=10):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
        transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2), transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
        transforms.RandomErasing(p=0.25, scale=(0.02,0.2)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    client_indices, client_class_counts = dirichlet_split(train_dataset, n_clients, alpha, n_classes)

    print(f"\n数据分布 (α={alpha}):")
    print("-" * 80)
    for cid in range(n_clients):
        counts = [client_class_counts[cid].get(c,0) for c in range(n_classes)]
        total = sum(counts)
        ncls = sum(1 for c in counts if c > 0)
        print(f"  Client {cid}: {ncls} cls, {total} samples")
    print("-" * 80)

    targets = np.array(train_dataset.targets)
    client_all_loaders = {}
    for cid in range(n_clients):
        subset = Subset(train_dataset, client_indices[cid])
        client_all_loaders[cid] = DataLoader(subset, batch_size=128, shuffle=True, drop_last=True, **DL_KWARGS)

    client_class_loaders = {}
    for cid in range(n_clients):
        client_class_loaders[cid] = {}
        class_idx_map = defaultdict(list)
        for idx in client_indices[cid]:
            class_idx_map[targets[idx]].append(idx)
        for cls_id, indices in class_idx_map.items():
            subset = Subset(train_dataset, indices)
            dl_kw = dict(num_workers=4, pin_memory=True,
                         persistent_workers=True if len(indices)>=64 else False)
            client_class_loaders[cid][cls_id] = DataLoader(
                subset, batch_size=64, shuffle=True, drop_last=False, **dl_kw)

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, **DL_KWARGS)
    return client_all_loaders, client_class_loaders, test_loader, client_class_counts


def generate_etf(n_classes, feature_dim, seed=42):
    rng = torch.Generator(); rng.manual_seed(seed)
    M = np.sqrt(n_classes/(n_classes-1)) * (torch.eye(n_classes) - torch.ones(n_classes,n_classes)/n_classes)
    if feature_dim > n_classes:
        Q, _ = torch.linalg.qr(torch.randn(feature_dim, n_classes, generator=rng))
        M = M @ Q.T
    print(f"  ETF: norm={torch.norm(M,dim=1).mean():.4f}")
    return M


class Backbone(nn.Module):
    def __init__(self, feature_dim=256, channels=None):
        super().__init__()
        if channels is None: channels = [64,128,256,256]
        c1,c2,c3,c4 = channels
        self.channels = channels
        self.features = nn.Sequential(
            nn.Conv2d(3,c1,3,padding=1), nn.BatchNorm2d(c1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(c1,c2,3,padding=1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(c2,c3,3,padding=1), nn.BatchNorm2d(c3), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(c3,c4,3,padding=1), nn.BatchNorm2d(c4), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(c4*2*2, feature_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, dim=1)

    def forward_conv(self, x):
        """只过 conv 层, 返回 feature map (B, C4, 2, 2)"""
        return self.features(x)


class ConditionalExpert(nn.Module):
    def __init__(self, feature_dim=256, etf_dim=256, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.enc1 = nn.Linear(feature_dim+etf_dim, hidden_dim)
        self.enc_bn1 = nn.LayerNorm(hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, latent_dim)
        self.dec1 = nn.Linear(latent_dim+etf_dim, hidden_dim)
        self.dec_bn1 = nn.LayerNorm(hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, feature_dim)

    def encode(self, f, etf_cond):
        x = torch.cat([f, etf_cond], dim=1)
        return self.enc2(F.relu(self.enc_bn1(self.enc1(x))))

    def decode(self, z, etf_cond):
        x = torch.cat([z, etf_cond], dim=1)
        return self.dec2(F.relu(self.dec_bn1(self.dec1(x))))

    def forward(self, f, etf_cond):
        z = self.encode(f, etf_cond)
        return self.decode(z, etf_cond), z


def etf_contrastive_loss(features, labels, etf_targets, temperature=0.1,
                         lambda_proto=1.0, lambda_sample=1.0):
    features = F.normalize(features, dim=1)
    batch_size = features.size(0)
    loss_proto = F.cross_entropy(torch.mm(features, etf_targets.T)/temperature, labels)
    loss_sample = torch.tensor(0.0, device=features.device)
    if lambda_sample > 0 and batch_size > 1:
        self_mask = torch.eye(batch_size, device=features.device, dtype=torch.bool)
        not_self = ~self_mask
        sim = torch.mm(features, features.T)/temperature
        mask_pos = (labels.unsqueeze(0)==labels.unsqueeze(1)).float()*not_self.float()
        pos_count = mask_pos.sum(dim=1); valid = pos_count > 0
        if valid.sum() > 0:
            sim_stable = sim - sim.max(dim=1,keepdim=True)[0].detach()
            exp_sim = torch.exp(sim_stable)*not_self.float()
            log_prob = sim_stable - torch.log(exp_sim.sum(dim=1)+1e-8).unsqueeze(1)
            loss_sample = -(mask_pos*log_prob).sum(dim=1)[valid]/(pos_count[valid]+1e-8)
            loss_sample = loss_sample.mean()
    return lambda_proto*loss_proto + lambda_sample*loss_sample, loss_proto.item(), loss_sample.item()


def etf_alignment_loss(features, labels, etf_targets):
    features = F.normalize(features, dim=1)
    return (1-(features*etf_targets[labels]).sum(dim=1)).mean()


def train_backbone(backbone, loader, classes, etf_targets,
                   epochs=600, lr=1e-3, temp=0.1, lp=1.0, ls=0.5, la=0.5):
    backbone = backbone.to(device); etf_dev = etf_targets.to(device)
    ncl = len(classes); backbone.train()
    opt = torch.optim.Adam(backbone.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    amp_ctx = (torch.amp.autocast('cuda', dtype=torch.bfloat16) if USE_BF16
               else torch.amp.autocast('cuda', enabled=False))
    for epoch in range(epochs):
        el=0; nb=0
        for x,y in loader:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            with amp_ctx:
                f=backbone(x)
                if ncl>=2:
                    lc,_,_=etf_contrastive_loss(f,y,etf_dev,temp,lp,ls)
                    loss=lc+la*etf_alignment_loss(f,y,etf_dev)
                else:
                    loss=lp*etf_alignment_loss(f,y,etf_dev)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            el+=loss.item(); nb+=1
        sched.step()
        if (epoch+1)%100==0 or epoch==0:
            print(f"      BB epoch {epoch+1}/{epochs}, loss={el/max(nb,1):.4f}")
    return backbone


def preextract_features(backbone, dataloader):
    backbone.eval(); all_f=[]
    with torch.no_grad():
        amp_ctx = (torch.amp.autocast('cuda', dtype=torch.bfloat16) if USE_BF16
                   else torch.amp.autocast('cuda', enabled=False))
        with amp_ctx:
            for x,_ in dataloader:
                all_f.append(backbone(x.to(device,non_blocking=True)).float())
    return torch.cat(all_f, dim=0)


def train_expert_single(expert, cached, etf_own, etf_dev, others_idx,
                        fdim=256, epochs=600, lr=1e-3, margin=0.05,
                        ns_min=0.05, ns_max=0.3, n_neg=64, l_neg=1.0):
    expert=expert.to(device); N=cached.size(0); no=others_idx.size(0)
    opt=torch.optim.Adam(expert.parameters(),lr=lr)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    bs=min(64,N)
    for epoch in range(epochs):
        expert.train(); perm=torch.randperm(N,device=device)
        for i in range(0,N,bs):
            idx=perm[i:i+bs]; fp=cached[idx]; B=fp.size(0)
            co=etf_own.unsqueeze(0).expand(B,-1)
            fr1,_=expert(fp,co); l1=F.mse_loss(fr1,fp)
            nc=others_idx[torch.randint(0,no,(n_neg,),device=device)]
            sc=ns_min+(ns_max-ns_min)*torch.rand(n_neg,1,device=device)
            ff=F.normalize(etf_dev[nc]+torch.randn(n_neg,fdim,device=device)*sc,dim=1)
            fc=others_idx[torch.randint(0,no,(B,),device=device)]
            fr2,_=expert(fp,etf_dev[fc]); l2=F.relu(margin-((fp-fr2)**2).mean(dim=1)).mean()
            fr3,_=expert(ff,etf_dev[nc]); l3=F.relu(margin-((ff-fr3)**2).mean(dim=1)).mean()
            fr4,_=expert(ff,etf_own.unsqueeze(0).expand(n_neg,-1))
            l4=F.relu(margin-((ff-fr4)**2).mean(dim=1)).mean()
            loss=l1+l_neg*(l2+l3+l4)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sched.step()
    expert.eval(); return expert


def train_experts(backbone, class_loaders, classes, etf_targets,
                  nc=10, fdim=256, ldim=32, epochs=600, lr=1e-3,
                  margin=0.05, ns_min=0.05, ns_max=0.3, n_neg=64, l_neg=1.0):
    backbone.eval(); etf_dev=etf_targets.to(device); experts={}
    om={c:torch.tensor([k for k in range(nc) if k!=c],device=device) for c in range(nc)}
    for cls in classes:
        print(f"      Expert {cls}...",end=" ",flush=True); t0=time.time()
        cached=preextract_features(backbone, class_loaders[cls])
        expert=ConditionalExpert(fdim,fdim,128,ldim).to(device)
        expert=train_expert_single(expert,cached,etf_dev[cls],etf_dev,om[cls],
                                   fdim,epochs,lr,margin,ns_min,ns_max,n_neg,l_neg)
        experts[cls]=expert
        with torch.no_grad():
            ne=min(256,cached.size(0))
            fr,_=expert(cached[:ne],etf_dev[cls].unsqueeze(0).expand(ne,-1))
            mse=((cached[:ne]-fr)**2).mean().item()
        print(f"done ({time.time()-t0:.1f}s) MSE={mse:.6f}")
    return experts


def compute_error_stats(backbone, expert, dataloader, etf_vertex):
    backbone.eval(); expert.eval(); etf_v=etf_vertex.to(device); errors=[]
    with torch.no_grad():
        for x,_ in dataloader:
            f=backbone(x.to(device,non_blocking=True)); B=f.size(0)
            fr,_=expert(f,etf_v.unsqueeze(0).expand(B,-1))
            errors.append(((f-fr)**2).mean(dim=1))
    errors=torch.cat(errors); return errors.mean().item(), errors.std().item()


class ClientModel:
    def __init__(self, cid, backbone):
        self.client_id=cid; self.backbone=backbone
        self.experts={}; self.stats={}
    def add_expert(self, cls_id, expert, mu, sigma, n):
        self.experts[cls_id]=expert; self.stats[cls_id]=(mu,sigma,n)


# ═══════════════════════════════════════════════════════════
# 2. ★ Union 聚合 (返回 channel maps)
# ═══════════════════════════════════════════════════════════

def cosine_sim_matrix(a, b):
    a_f=F.normalize(a.view(a.size(0),-1).float(),dim=1)
    b_f=F.normalize(b.view(b.size(0),-1).float(),dim=1)
    return torch.mm(a_f, b_f.T)


def greedy_union_match(all_filters, all_sources, threshold=0.95):
    if not all_filters: return [],[]
    stacked=torch.stack(all_filters); N=stacked.size(0)
    sim=cosine_sim_matrix(stacked,stacked)
    assigned=[False]*N; global_filters=[]; mapping=[]
    norms=stacked.view(N,-1).float().norm(dim=1)
    order=norms.argsort(descending=True).tolist()
    for seed in order:
        if assigned[seed]: continue
        cluster=[seed]; assigned[seed]=True
        for j in order:
            if assigned[j]: continue
            if all(sim[j,ci]>threshold for ci in cluster):
                cluster.append(j); assigned[j]=True
        cf=stacked[cluster]; cn=norms[cluster]
        w=cn/(cn.sum()+1e-8)
        merged=(cf.float()*w.view(-1,*([1]*(cf.dim()-1)))).sum(0)
        global_filters.append(merged); mapping.append([all_sources[i] for i in cluster])
    return global_filters, mapping


def union_aggregate_cnn(client_backbones, feature_dim=256, threshold=0.95):
    """Union 聚合, 返回 (merged_backbone, per_layer_channel_maps)"""
    K=len(client_backbones)
    print(f"\n  [Union] 聚合 {K} 个 CNN backbone (thr={threshold})")

    conv_idx=[0,4,8,12]; bn_idx=[1,5,9,13]
    layer_params=[]; all_layer_maps=[]
    prev_maps=None; prev_n=3

    for li in range(4):
        ci=conv_idx[li]; bi=bn_idx[li]
        all_f=[]; all_s=[]; all_b=[]; all_bn=[]
        for k,bb in enumerate(client_backbones):
            conv=bb.features[ci]; bn=bb.features[bi]
            w=conv.weight.data.cpu(); b_=conv.bias.data.cpu()
            Co,Ci=w.size(0),w.size(1)
            if li==0:
                wr=w
            else:
                wr=torch.zeros(Co,prev_n,3,3)
                for loc_in in range(Ci):
                    if loc_in in prev_maps[k]:
                        wr[:,prev_maps[k][loc_in],:,:]=w[:,loc_in,:,:]
            for i in range(Co):
                all_f.append(wr[i]); all_s.append((k,i)); all_b.append(b_[i])
                all_bn.append({'w':bn.weight.data.cpu()[i],'b':bn.bias.data.cpu()[i],
                               'm':bn.running_mean.cpu()[i],'v':bn.running_var.cpu()[i]})

        gf,fm=greedy_union_match(all_f,all_s,threshold)
        No=len(gf)
        mb=[]; mbn={'w':[],'b':[],'m':[],'v':[]}
        for g,group in enumerate(fm):
            idxs=[]
            for(ck,ci_)in group:
                flat=sum(client_backbones[kk].features[ci].weight.size(0) for kk in range(ck))+ci_
                idxs.append(flat)
            mb.append(torch.stack([all_b[i] for i in idxs]).mean())
            for key in 'wbmv':
                full_key={'w':'w','b':'b','m':'m','v':'v'}[key]
                mbn[full_key].append(torch.stack([all_bn[i][full_key] for i in idxs]).mean())

        new_maps=[{} for _ in range(K)]
        for g,group in enumerate(fm):
            for(ck,ci_)in group:
                new_maps[ck][ci_]=g

        orig_co=client_backbones[0].features[ci].weight.size(0)
        print(f"    Conv{li+1}: {prev_n}→{No} (from {K*orig_co})")

        layer_params.append({
            'N_in':prev_n,'N_out':No,
            'filters':torch.stack(gf),'biases':torch.stack(mb),
            'bn_w':torch.stack(mbn['w']),'bn_b':torch.stack(mbn['b']),
            'bn_m':torch.stack(mbn['m']),'bn_v':torch.stack(mbn['v']),
        })
        all_layer_maps.append(new_maps)
        prev_maps=new_maps; prev_n=No

    # FC
    Nf=prev_n; fc_in=Nf*2*2
    mfw=torch.zeros(feature_dim,fc_in); mfb=torch.zeros(feature_dim)
    fcc=torch.zeros(fc_in)
    for k,bb in enumerate(client_backbones):
        fw=bb.fc.weight.data.cpu(); fb=bb.fc.bias.data.cpu()
        c4l=bb.channels[3]
        for loc_i in range(c4l):
            if loc_i not in prev_maps[k]: continue
            g=prev_maps[k][loc_i]
            mfw[:,g*4:(g+1)*4]+=fw[:,loc_i*4:(loc_i+1)*4]
            fcc[g*4:(g+1)*4]+=1
        mfb+=fb/K
    fcc=fcc.clamp(min=1); mfw/=fcc.unsqueeze(0)

    channels=[lp['N_out'] for lp in layer_params]
    merged=Backbone(feature_dim,channels)
    with torch.no_grad():
        for li in range(4):
            lp=layer_params[li]; ci_=conv_idx[li]; bi_=bn_idx[li]
            Ni=lp['N_in']; No=lp['N_out']
            merged.features[ci_]=nn.Conv2d(Ni,No,3,padding=1)
            merged.features[ci_].weight.copy_(lp['filters'][:,:Ni])
            merged.features[ci_].bias.copy_(lp['biases'])
            merged.features[bi_]=nn.BatchNorm2d(No)
            merged.features[bi_].weight.copy_(lp['bn_w'])
            merged.features[bi_].bias.copy_(lp['bn_b'])
            merged.features[bi_].running_mean.copy_(lp['bn_m'])
            merged.features[bi_].running_var.copy_(lp['bn_v'])
        merged.fc=nn.Linear(fc_in,feature_dim)
        merged.fc.weight.copy_(mfw); merged.fc.bias.copy_(mfb)

    n_params=sum(p.numel() for p in merged.parameters())
    print(f"    合并: {n_params:,} params, ch={channels}")

    # final_channel_maps = all_layer_maps[-1] (Conv4 的映射)
    return merged.to(device), all_layer_maps


# ═══════════════════════════════════════════════════════════
# 3. ★ 方向 1: Per-client FC 通道选路
# ═══════════════════════════════════════════════════════════

def inference_perclient_fc_routing(union_bb, client_backbones, client_experts_list,
                                   conv4_channel_maps, test_loader, etf_targets,
                                   n_classes=10):
    """
    方向 1: Union conv 提取宽 feature map → 按 channel map 选出各 client 通道 → 原始 FC → Expert
    """
    all_preds, all_labels = [], []
    etf_dev = etf_targets.to(device)
    K = len(client_backbones)
    union_bb.eval()

    # 预计算每个 client 的 Conv4 通道索引 (按 local 顺序排列)
    client_channel_indices = []
    for k in range(K):
        mapping = conv4_channel_maps[k]  # {local_i: global_g}
        sorted_pairs = sorted(mapping.items(), key=lambda x: x[0])
        global_indices = [g for _, g in sorted_pairs]
        client_channel_indices.append(global_indices)

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            batch_size = x.size(0)

            # Union conv forward
            feat_map = union_bb.forward_conv(x)  # (B, C4_union, 2, 2)

            class_errors = defaultdict(list)

            for k in range(K):
                # 选出 client k 的通道
                indices = client_channel_indices[k]
                feat_k = feat_map[:, indices, :, :]   # (B, c4_orig, 2, 2)
                feat_k_flat = feat_k.reshape(batch_size, -1)  # (B, c4_orig*4)

                # 用 client k 的原始 FC
                f_k = F.normalize(client_backbones[k].fc(feat_k_flat), dim=1)  # (B, 256)

                # 用 client k 的 expert
                for cls_id, expert in client_experts_list[k].items():
                    etf_cond = etf_dev[cls_id].unsqueeze(0).expand(batch_size, -1)
                    f_recon, _ = expert(f_k, etf_cond)
                    error = ((f_k - f_recon)**2).mean(dim=1)
                    class_errors[cls_id].append(error)

            class_scores = []
            for c in range(n_classes):
                if c not in class_errors:
                    class_scores.append(torch.full((batch_size,), float('inf'), device=device))
                else:
                    class_scores.append(torch.stack(class_errors[c], dim=1).min(dim=1)[0])
            preds = torch.stack(class_scores, dim=1).argmin(dim=1)
            all_preds.append(preds.cpu()); all_labels.append(y)

    all_preds=torch.cat(all_preds).numpy(); all_labels=torch.cat(all_labels).numpy()
    return (all_preds==all_labels).mean()


def inference_perclient_fc_etfproto(union_bb, client_backbones, conv4_channel_maps,
                                     test_loader, etf_targets, n_classes=10):
    """方向 1 变体: 通道选路 + ETF-Proto (不用 expert)"""
    all_preds, all_labels = [], []
    etf_dev = etf_targets.to(device)
    K = len(client_backbones)
    union_bb.eval()

    client_channel_indices = []
    for k in range(K):
        mapping = conv4_channel_maps[k]
        sorted_pairs = sorted(mapping.items(), key=lambda x: x[0])
        client_channel_indices.append([g for _, g in sorted_pairs])

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            feat_map = union_bb.forward_conv(x)
            all_sims = []
            for k in range(K):
                indices = client_channel_indices[k]
                feat_k = feat_map[:, indices, :, :].reshape(x.size(0), -1)
                f_k = F.normalize(client_backbones[k].fc(feat_k), dim=1)
                all_sims.append(torch.mm(f_k, etf_dev.T))
            best_sim = torch.stack(all_sims, dim=0).max(dim=0)[0]
            all_preds.append(best_sim.argmax(dim=1).cpu())
            all_labels.append(y)
    return (torch.cat(all_preds).numpy()==torch.cat(all_labels).numpy()).mean()


# ═══════════════════════════════════════════════════════════
# 4. ★ 方向 2: Union ETF-Proto + Expert 联合评分
# ═══════════════════════════════════════════════════════════

def inference_joint_scoring(union_bb, client_backbones, client_experts_list,
                            conv4_channel_maps, test_loader, etf_targets,
                            alpha=0.5, temperature=1.0, n_classes=10):
    """
    方向 2: 联合评分

    信号 A: Union backbone ETF-Proto → softmax(cosine_sim / T)
    信号 B: Per-client routing Expert → softmax(-error / T)
    联合: alpha * A + (1-alpha) * B
    """
    all_preds, all_labels = [], []
    etf_dev = etf_targets.to(device)
    K = len(client_backbones)
    union_bb.eval()

    client_channel_indices = []
    for k in range(K):
        mapping = conv4_channel_maps[k]
        sorted_pairs = sorted(mapping.items(), key=lambda x: x[0])
        client_channel_indices.append([g for _, g in sorted_pairs])

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            batch_size = x.size(0)

            # 信号 A: Union ETF-Proto
            f_union = union_bb(x)   # (B, 256)
            sim_union = torch.mm(f_union, etf_dev.T)   # (B, 10)
            score_a = F.softmax(sim_union / temperature, dim=1)  # (B, 10)

            # 信号 B: Per-client routing Expert
            feat_map = union_bb.forward_conv(x)
            class_min_errors = torch.full((batch_size, n_classes), float('inf'), device=device)

            for k in range(K):
                indices = client_channel_indices[k]
                feat_k = feat_map[:, indices, :, :].reshape(batch_size, -1)
                f_k = F.normalize(client_backbones[k].fc(feat_k), dim=1)
                for cls_id, expert in client_experts_list[k].items():
                    etf_cond = etf_dev[cls_id].unsqueeze(0).expand(batch_size, -1)
                    f_recon, _ = expert(f_k, etf_cond)
                    error = ((f_k - f_recon)**2).mean(dim=1)  # (B,)
                    class_min_errors[:, cls_id] = torch.min(
                        class_min_errors[:, cls_id], error)

            # 将 error 转为概率 (error 越小越好 → 取负)
            # 对 inf 值处理: 替换为最大有限值
            finite_mask = class_min_errors < float('inf')
            max_finite = class_min_errors[finite_mask].max() if finite_mask.any() else 1.0
            class_min_errors = torch.where(finite_mask, class_min_errors,
                                           torch.full_like(class_min_errors, max_finite * 2))
            score_b = F.softmax(-class_min_errors / temperature, dim=1)  # (B, 10)

            # 联合
            combined = alpha * score_a + (1 - alpha) * score_b
            preds = combined.argmax(dim=1)
            all_preds.append(preds.cpu()); all_labels.append(y)

    return (torch.cat(all_preds).numpy()==torch.cat(all_labels).numpy()).mean()


# ═══════════════════════════════════════════════════════════
# 5. ★ 方向 3: 跨 client Expert 利用
# ═══════════════════════════════════════════════════════════

def inference_cross_client_expert(union_bb, client_backbones, client_experts_list,
                                  conv4_channel_maps, test_loader, etf_targets,
                                  n_classes=10):
    """
    方向 3: 每个 expert 在所有 client 的路由特征上评估

    对 class c 的 expert (来自 client j):
      不仅用 client j 的路由特征, 还用 client 0,1,2,... 的路由特征
      取所有组合中误差最小值
    """
    all_preds, all_labels = [], []
    etf_dev = etf_targets.to(device)
    K = len(client_backbones)
    union_bb.eval()

    client_channel_indices = []
    for k in range(K):
        mapping = conv4_channel_maps[k]
        sorted_pairs = sorted(mapping.items(), key=lambda x: x[0])
        client_channel_indices.append([g for _, g in sorted_pairs])

    # 收集所有 expert
    all_experts = defaultdict(list)  # cls_id → [(expert, source_client)]
    for k in range(K):
        for cls_id, expert in client_experts_list[k].items():
            all_experts[cls_id].append((expert, k))

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            batch_size = x.size(0)
            feat_map = union_bb.forward_conv(x)

            # 预计算所有 client 的路由特征
            client_features = []
            for k in range(K):
                indices = client_channel_indices[k]
                feat_k = feat_map[:, indices, :, :].reshape(batch_size, -1)
                f_k = F.normalize(client_backbones[k].fc(feat_k), dim=1)
                client_features.append(f_k)

            class_errors = defaultdict(list)
            for cls_id, expert_list in all_experts.items():
                for expert, src_k in expert_list:
                    etf_cond = etf_dev[cls_id].unsqueeze(0).expand(batch_size, -1)

                    # 用 source client 的路由特征 (标准)
                    f_src = client_features[src_k]
                    fr, _ = expert(f_src, etf_cond)
                    class_errors[cls_id].append(((f_src - fr)**2).mean(dim=1))

                    # ★ 也用其他 client 的路由特征 (跨 client)
                    for other_k in range(K):
                        if other_k == src_k:
                            continue
                        f_other = client_features[other_k]
                        fr_other, _ = expert(f_other, etf_cond)
                        class_errors[cls_id].append(((f_other - fr_other)**2).mean(dim=1))

            class_scores = []
            for c in range(n_classes):
                if c not in class_errors:
                    class_scores.append(torch.full((batch_size,), float('inf'), device=device))
                else:
                    class_scores.append(torch.stack(class_errors[c], dim=1).min(dim=1)[0])
            preds = torch.stack(class_scores, dim=1).argmin(dim=1)
            all_preds.append(preds.cpu()); all_labels.append(y)

    return (torch.cat(all_preds).numpy()==torch.cat(all_labels).numpy()).mean()


# ═══════════════════════════════════════════════════════════
# 6. ★ 方向 4: 改进投票策略
# ═══════════════════════════════════════════════════════════

def inference_weighted_voting(union_bb, client_backbones, client_experts_list,
                              client_stats_list, conv4_channel_maps,
                              test_loader, etf_targets, n_classes=10):
    """
    方向 4a: 加权投票 — expert 的 vote 按训练样本数加权
    """
    all_preds, all_labels = [], []
    etf_dev = etf_targets.to(device)
    K = len(client_backbones)
    union_bb.eval()

    client_channel_indices = []
    for k in range(K):
        mapping = conv4_channel_maps[k]
        sorted_pairs = sorted(mapping.items(), key=lambda x: x[0])
        client_channel_indices.append([g for _, g in sorted_pairs])

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            batch_size = x.size(0)
            feat_map = union_bb.forward_conv(x)

            # 加权评分: score = Σ (n_samples * (-error))
            class_weighted_scores = torch.zeros(batch_size, n_classes, device=device)

            for k in range(K):
                indices = client_channel_indices[k]
                feat_k = feat_map[:, indices, :, :].reshape(batch_size, -1)
                f_k = F.normalize(client_backbones[k].fc(feat_k), dim=1)

                for cls_id, expert in client_experts_list[k].items():
                    mu, sigma, n_samples = client_stats_list[k][cls_id]
                    etf_cond = etf_dev[cls_id].unsqueeze(0).expand(batch_size, -1)
                    fr, _ = expert(f_k, etf_cond)
                    error = ((f_k - fr)**2).mean(dim=1)  # (B,)
                    # 权重 = log(n_samples+1), 避免极端值
                    weight = np.log(n_samples + 1)
                    class_weighted_scores[:, cls_id] += weight * (-error)

            preds = class_weighted_scores.argmax(dim=1)
            all_preds.append(preds.cpu()); all_labels.append(y)

    return (torch.cat(all_preds).numpy()==torch.cat(all_labels).numpy()).mean()


def inference_filtered_voting(union_bb, client_backbones, client_experts_list,
                              client_stats_list, conv4_channel_maps,
                              test_loader, etf_targets,
                              min_samples=50, n_classes=10):
    """
    方向 4b: 过滤投票 — 只用训练样本 >= min_samples 的 expert
    """
    all_preds, all_labels = [], []
    etf_dev = etf_targets.to(device)
    K = len(client_backbones)
    union_bb.eval()

    client_channel_indices = []
    for k in range(K):
        mapping = conv4_channel_maps[k]
        sorted_pairs = sorted(mapping.items(), key=lambda x: x[0])
        client_channel_indices.append([g for _, g in sorted_pairs])

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            batch_size = x.size(0)
            feat_map = union_bb.forward_conv(x)

            class_errors = defaultdict(list)
            for k in range(K):
                indices = client_channel_indices[k]
                feat_k = feat_map[:, indices, :, :].reshape(batch_size, -1)
                f_k = F.normalize(client_backbones[k].fc(feat_k), dim=1)
                for cls_id, expert in client_experts_list[k].items():
                    mu, sigma, n_samples = client_stats_list[k][cls_id]
                    if n_samples < min_samples:
                        continue  # ★ 过滤!
                    etf_cond = etf_dev[cls_id].unsqueeze(0).expand(batch_size, -1)
                    fr, _ = expert(f_k, etf_cond)
                    class_errors[cls_id].append(((f_k - fr)**2).mean(dim=1))

            class_scores = []
            for c in range(n_classes):
                if c not in class_errors:
                    class_scores.append(torch.full((batch_size,), float('inf'), device=device))
                else:
                    class_scores.append(torch.stack(class_errors[c], dim=1).min(dim=1)[0])
            preds = torch.stack(class_scores, dim=1).argmin(dim=1)
            all_preds.append(preds.cpu()); all_labels.append(y)

    return (torch.cat(all_preds).numpy()==torch.cat(all_labels).numpy()).mean()


# ═══════════════════════════════════════════════════════════
# 7. Baseline 推理
# ═══════════════════════════════════════════════════════════

def inference_orig_ensemble_expert(client_models, test_loader, etf_targets, n_classes=10):
    all_preds, all_labels = [], []
    etf_dev = etf_targets.to(device)
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True); bs = x.size(0)
            class_errors = defaultdict(list)
            for cm in client_models:
                f = cm.backbone(x)
                for cls_id, expert in cm.experts.items():
                    cond = etf_dev[cls_id].unsqueeze(0).expand(bs, -1)
                    fr, _ = expert(f, cond)
                    class_errors[cls_id].append(((f - fr)**2).mean(dim=1))
            cs = []
            for c in range(n_classes):
                if c not in class_errors:
                    cs.append(torch.full((bs,), float('inf'), device=device))
                else:
                    cs.append(torch.stack(class_errors[c], dim=1).min(dim=1)[0])
            all_preds.append(torch.stack(cs, dim=1).argmin(dim=1).cpu())
            all_labels.append(y)
    return (torch.cat(all_preds).numpy()==torch.cat(all_labels).numpy()).mean()


def inference_union_etf_proto(union_bb, test_loader, etf_targets):
    all_preds, all_labels = [], []
    etf_dev = etf_targets.to(device); union_bb.eval()
    with torch.no_grad():
        for x, y in test_loader:
            f = F.normalize(union_bb(x.to(device, non_blocking=True)), dim=1)
            all_preds.append(torch.mm(f, etf_dev.T).argmax(dim=1).cpu())
            all_labels.append(y)
    return (torch.cat(all_preds).numpy()==torch.cat(all_labels).numpy()).mean()


# ═══════════════════════════════════════════════════════════
# 8. 可视化
# ═══════════════════════════════════════════════════════════

def plot_results(results, save_path):
    fig, ax = plt.subplots(figsize=(16, 10))
    names = list(results.keys())
    accs = [results[n]*100 for n in names]
    colors = []
    for n in names:
        if 'A0' in n: colors.append('#607D8B')
        elif 'Union ETF' in n: colors.append('#4CAF50')
        elif 'D1' in n: colors.append('#2196F3')
        elif 'D2' in n: colors.append('#E91E63')
        elif 'D3' in n: colors.append('#9C27B0')
        elif 'D4' in n: colors.append('#FF9800')
        else: colors.append('#795548')
    bars = ax.barh(range(len(names)), accs, color=colors, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('v2-Union+: 4 方案验证', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2.,
                f'{acc:.2f}%', va='center', fontsize=9, fontweight='bold')
    ax.axvline(x=80, color='red', linestyle='--', alpha=0.5, label='80% target')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save_path}")


# ═══════════════════════════════════════════════════════════
# 9. 主实验
# ═══════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 80)
    print("MoE One-Shot FL v2-Union+ — 4 方案验证")
    print("=" * 80)

    N_CLIENTS=5; N_CLASSES=10; ALPHA=0.05
    FEATURE_DIM=256; LATENT_DIM=32
    EPOCHS_BB=600; EPOCHS_EXP=600
    LR_BB=1e-3; LR_EXP=1e-3; TEMP=0.1
    LP=1.0; LS=0.5; LA=0.5
    MARGIN=0.05; NS_MIN=0.05; NS_MAX=0.3; N_NEG=64; L_NEG=1.0
    UNION_THR=0.95

    os.makedirs('outputs', exist_ok=True)
    print(f"\n生成 ETF:")
    etf_targets = generate_etf(N_CLASSES, FEATURE_DIM)
    client_all_loaders, client_class_loaders, test_loader, client_class_counts = \
        prepare_data(N_CLIENTS, ALPHA, N_CLASSES)

    # ════════════════════════════════════════════
    # Phase 1: 训练
    # ════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  Phase 1: 训练")
    print(f"{'='*60}")

    client_models = []
    client_backbones = []
    client_experts_list = []
    client_stats_list = []
    t_start = time.time()

    for cid in range(N_CLIENTS):
        classes = sorted(client_class_counts[cid].keys())
        n_total = sum(client_class_counts[cid].values())
        print(f"\n  Client {cid}: {len(classes)} cls, {n_total} samples, classes={classes}")

        bb = Backbone(FEATURE_DIM)
        bb = train_backbone(bb, client_all_loaders[cid], classes, etf_targets,
                            EPOCHS_BB, LR_BB, TEMP, LP, LS, LA)
        experts = train_experts(bb, client_class_loaders[cid], classes, etf_targets,
                                N_CLASSES, FEATURE_DIM, LATENT_DIM, EPOCHS_EXP, LR_EXP,
                                MARGIN, NS_MIN, NS_MAX, N_NEG, L_NEG)

        cm = ClientModel(cid, bb)
        stats = {}
        for cls_id in classes:
            mu, sigma = compute_error_stats(bb, experts[cls_id],
                                            client_class_loaders[cid][cls_id], etf_targets[cls_id])
            n = client_class_counts[cid][cls_id]
            cm.add_expert(cls_id, experts[cls_id], mu, sigma, n)
            stats[cls_id] = (mu, sigma, n)
            print(f"    Class {cls_id}: n={n:5d}, μ={mu:.6f}")
        client_models.append(cm)
        client_backbones.append(bb)
        client_experts_list.append(experts)
        client_stats_list.append(stats)

    train_time = time.time() - t_start
    print(f"\n  训练时间: {train_time:.1f}s")

    # ════════════════════════════════════════════
    # Phase 2: Union 聚合
    # ════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  Phase 2: Union 聚合")
    print(f"{'='*60}")

    union_bb, all_layer_maps = union_aggregate_cnn(client_backbones, FEATURE_DIM, UNION_THR)
    conv4_maps = all_layer_maps[-1]  # Conv4 的 channel maps

    # ════════════════════════════════════════════
    # Phase 3: 评估所有方案
    # ════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  Phase 3: 评估")
    print(f"{'='*60}")

    results = {}

    # --- Baselines ---
    print(f"\n  --- Baselines ---")

    acc = inference_orig_ensemble_expert(client_models, test_loader, etf_targets, N_CLASSES)
    results['A0 Orig Ensemble Expert'] = acc
    print(f"  A0 Orig Ensemble Expert:        {acc:.2%}")

    acc = inference_union_etf_proto(union_bb, test_loader, etf_targets)
    results['Union ETF-Proto'] = acc
    print(f"  Union ETF-Proto:                {acc:.2%}")

    # --- 方向 1: Per-client FC routing ---
    print(f"\n  --- 方向 1: Per-client FC Routing ---")

    acc = inference_perclient_fc_etfproto(union_bb, client_backbones, conv4_maps,
                                          test_loader, etf_targets, N_CLASSES)
    results['D1 PC-Routing ETF-Proto'] = acc
    print(f"  D1 PC-Routing ETF-Proto:        {acc:.2%}")

    acc = inference_perclient_fc_routing(union_bb, client_backbones, client_experts_list,
                                         conv4_maps, test_loader, etf_targets, N_CLASSES)
    results['D1 PC-Routing Expert'] = acc
    print(f"  D1 PC-Routing Expert:           {acc:.2%}")

    # --- 方向 2: 联合评分 ---
    print(f"\n  --- 方向 2: Joint Scoring ---")

    for alpha in [0.3, 0.5, 0.7]:
        for temp in [0.1, 0.5, 1.0]:
            acc = inference_joint_scoring(union_bb, client_backbones, client_experts_list,
                                          conv4_maps, test_loader, etf_targets,
                                          alpha=alpha, temperature=temp, n_classes=N_CLASSES)
            key = f'D2 Joint α={alpha} T={temp}'
            results[key] = acc
            print(f"  {key}: {acc:.2%}")

    # --- 方向 3: 跨 client Expert ---
    print(f"\n  --- 方向 3: Cross-client Expert ---")

    acc = inference_cross_client_expert(union_bb, client_backbones, client_experts_list,
                                        conv4_maps, test_loader, etf_targets, N_CLASSES)
    results['D3 Cross-client Expert'] = acc
    print(f"  D3 Cross-client Expert:         {acc:.2%}")

    # --- 方向 4: 改进投票 ---
    print(f"\n  --- 方向 4: Improved Voting ---")

    acc = inference_weighted_voting(union_bb, client_backbones, client_experts_list,
                                    client_stats_list, conv4_maps,
                                    test_loader, etf_targets, N_CLASSES)
    results['D4a Weighted Voting'] = acc
    print(f"  D4a Weighted Voting:            {acc:.2%}")

    for min_n in [20, 50, 100, 200]:
        acc = inference_filtered_voting(union_bb, client_backbones, client_experts_list,
                                        client_stats_list, conv4_maps,
                                        test_loader, etf_targets,
                                        min_samples=min_n, n_classes=N_CLASSES)
        key = f'D4b Filtered(min={min_n})'
        results[key] = acc
        print(f"  {key}: {acc:.2%}")

    # ════════════════════════════════════════════
    # 结果汇总
    # ════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"★ 最终结果汇总")
    print(f"{'='*70}")
    print(f"  {'Method':<45} | {'Accuracy':>10}")
    print(f"  {'-'*60}")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        marker = " ★" if acc >= 0.80 else ""
        print(f"  {name:<45} | {acc:>8.2%}{marker}")

    # 关键对比
    baseline = results['A0 Orig Ensemble Expert']
    print(f"\n  vs Baseline ({baseline:.2%}):")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        diff = acc - baseline
        sign = '+' if diff >= 0 else ''
        print(f"    {name:<45} {sign}{diff*100:.2f}pp")

    plot_results(results, 'outputs/v2_union_plus_results.png')

    print(f"\n{'='*60}")
    print(f"  训练时间: {train_time:.1f}s")
    print(f"  Union threshold: {UNION_THR}")
    print(f"{'='*60}")
    print(f"\n实验完成!")


if __name__ == "__main__":
    main()