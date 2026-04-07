"""
resnet18_filter_merge.py
========================
ResNet-18 的 Filter Merge 实现.

核心挑战: ResNet 有 residual connection, conv2 的输出必须和 shortcut 的输出 channel 数一致.
解法: 
  - 在 stage 转换点(channel 变化处) 做 filter merge 决定新的 channel 数
  - block 内部的 conv1 也做 filter merge (扩展内部宽度)
  - conv2 输出必须匹配 stage 输出 → 用 permutation map 对齐后平均

用法:
  from resnet18_filter_merge import ResNet18Backbone, BasicBlock, union_aggregate_resnet18
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ═══════════════════════════════════════════════════════════
# ResNet-18 Backbone (CIFAR-adapted)
# ═══════════════════════════════════════════════════════════

class BasicBlock(nn.Module):
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


# ═══════════════════════════════════════════════════════════
# 合并后的 ResNet (支持每层不同 channel 数)
# ═══════════════════════════════════════════════════════════

class MergedBlock(nn.Module):
    """支持任意 channel 数的 BasicBlock"""
    def __init__(self, ic, internal_c, oc, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(ic, internal_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(internal_c)
        self.conv2 = nn.Conv2d(internal_c, oc, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(oc)
        self.has_shortcut = (stride != 1 or ic != oc)
        if self.has_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(ic, oc, 1, stride=stride, bias=False),
                nn.BatchNorm2d(oc))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class MergedResNet18(nn.Module):
    """合并后的 ResNet-18, 各层 channel 数可能不同"""
    def __init__(self, fd, dims):
        """dims: dict with keys C0, A10, A11, C1, A20, A21, C2, A30, A31, C3, A40, A41"""
        super().__init__()
        d = dims
        self.conv1 = nn.Conv2d(3, d['C0'], 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(d['C0'])
        self.layer1 = nn.Sequential(
            MergedBlock(d['C0'], d['A10'], d['C0'], stride=1),
            MergedBlock(d['C0'], d['A11'], d['C0'], stride=1))
        self.layer2 = nn.Sequential(
            MergedBlock(d['C0'], d['A20'], d['C1'], stride=2),
            MergedBlock(d['C1'], d['A21'], d['C1'], stride=1))
        self.layer3 = nn.Sequential(
            MergedBlock(d['C1'], d['A30'], d['C2'], stride=2),
            MergedBlock(d['C2'], d['A31'], d['C2'], stride=1))
        self.layer4 = nn.Sequential(
            MergedBlock(d['C2'], d['A40'], d['C3'], stride=2),
            MergedBlock(d['C3'], d['A41'], d['C3'], stride=1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(d['C3'], fd)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return F.normalize(self.fc(x), dim=1)


# ═══════════════════════════════════════════════════════════
# Filter Merge 核心工具函数
# ═══════════════════════════════════════════════════════════

def _get_bn_stats(bn, idx):
    return {
        'w': bn.weight.data.cpu()[idx],
        'b': bn.bias.data.cpu()[idx],
        'm': bn.running_mean.data.cpu()[idx],
        'v': bn.running_var.data.cpu()[idx],
    }


def _cluster_and_merge(all_filters, all_sources, thr):
    """聚类并合并 filters.
    all_filters: list of tensors (each is one filter)
    all_sources: list of (client_k, filter_idx)
    Returns: merged_filters, groups, perm_maps_per_client, n_merged
    """
    N = len(all_filters); K_max = max(s[0] for s in all_sources) + 1
    st = torch.stack(all_filters)
    sf = F.normalize(st.view(N, -1).float(), dim=1)
    sim = sf @ sf.T
    norms = st.view(N, -1).float().norm(dim=1)
    order = norms.argsort(descending=True).tolist()

    assigned = [False] * N; merged_f = []; groups = []
    for seed in order:
        if assigned[seed]: continue
        cluster = [seed]; assigned[seed] = True
        for j in order:
            if assigned[j]: continue
            if all(sim[j, c] > thr for c in cluster):
                cluster.append(j); assigned[j] = True
        cf = st[cluster]; cn = norms[cluster]
        ww = cn / (cn.sum() + 1e-8)
        merged_f.append((cf.float() * ww.view(-1, *([1]*(cf.dim()-1)))).sum(0))
        groups.append([all_sources[i] for i in cluster])

    pm = [{} for _ in range(K_max)]
    for g, grp in enumerate(groups):
        for ck, ci in grp:
            pm[ck][ci] = g
    return torch.stack(merged_f), groups, pm, len(merged_f)


def _remap_input(w, pm_k, n_in):
    """重映射 weight 的 input channels. w: (Co, Ci, kh, kw)"""
    Co = w.size(0); rest = w.shape[2:]
    wr = torch.zeros(Co, n_in, *rest)
    for old in range(w.size(1)):
        if old in pm_k:
            wr[:, pm_k[old]] = w[:, old]  # assignment, 和 CNN filter merge 一致
    return wr


def _merge_conv_bn(bbs, conv_fn, bn_fn, prev_pm, n_in, thr):
    """对一个 conv+bn 层做 filter merge.
    conv_fn(bb) → conv module, bn_fn(bb) → bn module
    Returns: merged_w, merged_bn_dict, new_pm, n_out
    """
    K = len(bbs); af = []; ass = []; abn = []
    for k, bb in enumerate(bbs):
        conv = conv_fn(bb); bn = bn_fn(bb)
        w = conv.weight.data.cpu(); Co = w.size(0)
        wr = w if prev_pm is None else _remap_input(w, prev_pm[k], n_in)
        for i in range(Co):
            af.append(wr[i]); ass.append((k, i)); abn.append(_get_bn_stats(bn, i))
    merged_w, groups, pm, No = _cluster_and_merge(af, ass, thr)
    # 合并 BN
    bn_merged = {key: [] for key in 'wbmv'}
    for grp in groups:
        for key in 'wbmv':
            idx_list = []
            for ck, ci in grp:
                offset = sum(conv_fn(bbs[kk]).weight.size(0) for kk in range(ck))
                idx_list.append(offset + ci)
            bn_merged[key].append(torch.stack([abn[i][key] for i in idx_list]).mean())
    bn_merged = {k: torch.stack(v) for k, v in bn_merged.items()}
    return merged_w, bn_merged, pm, No


def _avg_conv_bn(bbs, conv_fn, bn_fn, in_pm, n_in, out_pm, n_out):
    """对一个 conv+bn 层做对齐后平均 (output channel 固定).
    用于 identity shortcut block 中的 conv2.
    """
    K = len(bbs)
    ksize = conv_fn(bbs[0]).weight.shape[2:]
    avg_w = torch.zeros(n_out, n_in, *ksize)
    avg_bn = {k: torch.zeros(n_out) for k in 'wbmv'}
    cnt_w = torch.zeros(n_out, n_in, *([1]*len(ksize)))
    cnt_bn = torch.zeros(n_out)
    bn_key_map = {'w': 'weight', 'b': 'bias', 'm': 'running_mean', 'v': 'running_var'}

    for k, bb in enumerate(bbs):
        w = conv_fn(bb).weight.data.cpu(); bn = bn_fn(bb)
        for o_old in range(w.size(0)):
            if o_old not in out_pm[k]: continue
            o_new = out_pm[k][o_old]
            for i_old in range(w.size(1)):
                if i_old not in in_pm[k]: continue
                i_new = in_pm[k][i_old]
                avg_w[o_new, i_new] += w[o_old, i_old]
                cnt_w[o_new, i_new] += 1
            for key in 'wbmv':
                avg_bn[key][o_new] += getattr(bn, bn_key_map[key]).data.cpu()[o_old]
            cnt_bn[o_new] += 1

    avg_w /= cnt_w.clamp(min=1)
    for key in 'wbmv':
        avg_bn[key] /= cnt_bn.clamp(min=1)
    return avg_w, avg_bn


def _set_conv_bn(conv, bn, w, bn_dict):
    """将 merged weights 写入 conv 和 bn 模块"""
    with torch.no_grad():
        assert w.shape == conv.weight.shape, \
            f"Weight shape mismatch: merged={w.shape} vs conv={conv.weight.shape}"
        conv.weight.copy_(w)
        bn.weight.copy_(bn_dict['w'])
        bn.bias.copy_(bn_dict['b'])
        bn.running_mean.copy_(bn_dict['m'])
        bn.running_var.copy_(bn_dict['v'])


# ═══════════════════════════════════════════════════════════
# ★ ResNet-18 Filter Merge 主函数
# ═══════════════════════════════════════════════════════════

def union_aggregate_resnet18(bbs, fd=256, thr=0.95, device=None):
    """对 K 个 ResNet-18 backbone 做 filter merge.
    
    思路: 和 CNN filter merge 一样逐层聚类合并,
    但需要处理 residual connection 的约束:
    - 转换层 (stride=2, 有shortcut): conv2 输出决定新 stage channel 数
    - 非转换层 (identity shortcut): conv2 输出 channel 必须匹配 stage output
    - block 内部 conv1: 自由扩展
    """
    K = len(bbs)
    print(f"\n  [ResNet-18 Filter Merge] {K} clients, thr={thr}")

    # ── Stage 0: conv1 (3→64) ──
    w0, bn0, pm0, C0 = _merge_conv_bn(
        bbs, lambda b: b.conv1, lambda b: b.bn1, None, 3, thr)
    print(f"    conv1: 3→{C0} (from {K*64})")

    # ── layer1: 2 blocks, identity shortcut, output stays C0 ──
    # block0.conv1: merge
    w_l1b0c1, bn_l1b0c1, pm_l1b0c1, A10 = _merge_conv_bn(
        bbs, lambda b: b.layer1[0].conv1, lambda b: b.layer1[0].bn1, pm0, C0, thr)
    # block0.conv2: output must be C0, average with pm0
    w_l1b0c2, bn_l1b0c2 = _avg_conv_bn(
        bbs, lambda b: b.layer1[0].conv2, lambda b: b.layer1[0].bn2,
        pm_l1b0c1, A10, pm0, C0)
    print(f"    layer1[0]: {C0}→{A10}→{C0}")

    # block1.conv1: merge
    w_l1b1c1, bn_l1b1c1, pm_l1b1c1, A11 = _merge_conv_bn(
        bbs, lambda b: b.layer1[1].conv1, lambda b: b.layer1[1].bn1, pm0, C0, thr)
    # block1.conv2: output must be C0
    w_l1b1c2, bn_l1b1c2 = _avg_conv_bn(
        bbs, lambda b: b.layer1[1].conv2, lambda b: b.layer1[1].bn2,
        pm_l1b1c1, A11, pm0, C0)
    print(f"    layer1[1]: {C0}→{A11}→{C0}")

    # ── layer2: transition block (stride=2) + identity block ──
    # block0.conv1: merge (C0→A20)
    w_l2b0c1, bn_l2b0c1, pm_l2b0c1, A20 = _merge_conv_bn(
        bbs, lambda b: b.layer2[0].conv1, lambda b: b.layer2[0].bn1, pm0, C0, thr)
    # block0.conv2: merge → 决定 C1 (新 stage output)
    w_l2b0c2, bn_l2b0c2, pm1, C1 = _merge_conv_bn(
        bbs, lambda b: b.layer2[0].conv2, lambda b: b.layer2[0].bn2,
        pm_l2b0c1, A20, thr)
    # block0.shortcut: output must match C1/pm1
    w_l2sc, bn_l2sc = _avg_conv_bn(
        bbs, lambda b: b.layer2[0].shortcut[0], lambda b: b.layer2[0].shortcut[1],
        pm0, C0, pm1, C1)
    print(f"    layer2[0]: {C0}→{A20}→{C1} (from {K*128})")

    # block1: identity, output stays C1
    w_l2b1c1, bn_l2b1c1, pm_l2b1c1, A21 = _merge_conv_bn(
        bbs, lambda b: b.layer2[1].conv1, lambda b: b.layer2[1].bn1, pm1, C1, thr)
    w_l2b1c2, bn_l2b1c2 = _avg_conv_bn(
        bbs, lambda b: b.layer2[1].conv2, lambda b: b.layer2[1].bn2,
        pm_l2b1c1, A21, pm1, C1)
    print(f"    layer2[1]: {C1}→{A21}→{C1}")

    # ── layer3 ──
    w_l3b0c1, bn_l3b0c1, pm_l3b0c1, A30 = _merge_conv_bn(
        bbs, lambda b: b.layer3[0].conv1, lambda b: b.layer3[0].bn1, pm1, C1, thr)
    w_l3b0c2, bn_l3b0c2, pm2, C2 = _merge_conv_bn(
        bbs, lambda b: b.layer3[0].conv2, lambda b: b.layer3[0].bn2,
        pm_l3b0c1, A30, thr)
    w_l3sc, bn_l3sc = _avg_conv_bn(
        bbs, lambda b: b.layer3[0].shortcut[0], lambda b: b.layer3[0].shortcut[1],
        pm1, C1, pm2, C2)
    print(f"    layer3[0]: {C1}→{A30}→{C2} (from {K*256})")

    w_l3b1c1, bn_l3b1c1, pm_l3b1c1, A31 = _merge_conv_bn(
        bbs, lambda b: b.layer3[1].conv1, lambda b: b.layer3[1].bn1, pm2, C2, thr)
    w_l3b1c2, bn_l3b1c2 = _avg_conv_bn(
        bbs, lambda b: b.layer3[1].conv2, lambda b: b.layer3[1].bn2,
        pm_l3b1c1, A31, pm2, C2)
    print(f"    layer3[1]: {C2}→{A31}→{C2}")

    # ── layer4 ──
    w_l4b0c1, bn_l4b0c1, pm_l4b0c1, A40 = _merge_conv_bn(
        bbs, lambda b: b.layer4[0].conv1, lambda b: b.layer4[0].bn1, pm2, C2, thr)
    w_l4b0c2, bn_l4b0c2, pm3, C3 = _merge_conv_bn(
        bbs, lambda b: b.layer4[0].conv2, lambda b: b.layer4[0].bn2,
        pm_l4b0c1, A40, thr)
    w_l4sc, bn_l4sc = _avg_conv_bn(
        bbs, lambda b: b.layer4[0].shortcut[0], lambda b: b.layer4[0].shortcut[1],
        pm2, C2, pm3, C3)
    print(f"    layer4[0]: {C2}→{A40}→{C3} (from {K*512})")

    w_l4b1c1, bn_l4b1c1, pm_l4b1c1, A41 = _merge_conv_bn(
        bbs, lambda b: b.layer4[1].conv1, lambda b: b.layer4[1].bn1, pm3, C3, thr)
    w_l4b1c2, bn_l4b1c2 = _avg_conv_bn(
        bbs, lambda b: b.layer4[1].conv2, lambda b: b.layer4[1].bn2,
        pm_l4b1c1, A41, pm3, C3)
    print(f"    layer4[1]: {C3}→{A41}→{C3}")

    # ── FC ──
    mfw = torch.zeros(fd, C3); mfb = torch.zeros(fd); fcc = torch.zeros(C3)
    for k, bb in enumerate(bbs):
        fw = bb.fc.weight.data.cpu(); fb = bb.fc.bias.data.cpu()
        for old in range(fw.size(1)):
            if old in pm3[k]:
                g = pm3[k][old]; mfw[:, g] += fw[:, old]; fcc[g] += 1
        mfb += fb / K
    mfw /= fcc.clamp(min=1).unsqueeze(0)

    # ── 组装合并模型 ──
    dims = {'C0':C0, 'A10':A10, 'A11':A11,
            'C1':C1, 'A20':A20, 'A21':A21,
            'C2':C2, 'A30':A30, 'A31':A31,
            'C3':C3, 'A40':A40, 'A41':A41}
    merged = MergedResNet18(fd, dims)

    with torch.no_grad():
        _set_conv_bn(merged.conv1, merged.bn1, w0, bn0)

        # layer1
        _set_conv_bn(merged.layer1[0].conv1, merged.layer1[0].bn1, w_l1b0c1, bn_l1b0c1)
        _set_conv_bn(merged.layer1[0].conv2, merged.layer1[0].bn2, w_l1b0c2, bn_l1b0c2)
        _set_conv_bn(merged.layer1[1].conv1, merged.layer1[1].bn1, w_l1b1c1, bn_l1b1c1)
        _set_conv_bn(merged.layer1[1].conv2, merged.layer1[1].bn2, w_l1b1c2, bn_l1b1c2)

        # layer2
        _set_conv_bn(merged.layer2[0].conv1, merged.layer2[0].bn1, w_l2b0c1, bn_l2b0c1)
        _set_conv_bn(merged.layer2[0].conv2, merged.layer2[0].bn2, w_l2b0c2, bn_l2b0c2)
        _set_conv_bn(merged.layer2[0].shortcut[0], merged.layer2[0].shortcut[1], w_l2sc, bn_l2sc)
        _set_conv_bn(merged.layer2[1].conv1, merged.layer2[1].bn1, w_l2b1c1, bn_l2b1c1)
        _set_conv_bn(merged.layer2[1].conv2, merged.layer2[1].bn2, w_l2b1c2, bn_l2b1c2)

        # layer3
        _set_conv_bn(merged.layer3[0].conv1, merged.layer3[0].bn1, w_l3b0c1, bn_l3b0c1)
        _set_conv_bn(merged.layer3[0].conv2, merged.layer3[0].bn2, w_l3b0c2, bn_l3b0c2)
        _set_conv_bn(merged.layer3[0].shortcut[0], merged.layer3[0].shortcut[1], w_l3sc, bn_l3sc)
        _set_conv_bn(merged.layer3[1].conv1, merged.layer3[1].bn1, w_l3b1c1, bn_l3b1c1)
        _set_conv_bn(merged.layer3[1].conv2, merged.layer3[1].bn2, w_l3b1c2, bn_l3b1c2)

        # layer4
        _set_conv_bn(merged.layer4[0].conv1, merged.layer4[0].bn1, w_l4b0c1, bn_l4b0c1)
        _set_conv_bn(merged.layer4[0].conv2, merged.layer4[0].bn2, w_l4b0c2, bn_l4b0c2)
        _set_conv_bn(merged.layer4[0].shortcut[0], merged.layer4[0].shortcut[1], w_l4sc, bn_l4sc)
        _set_conv_bn(merged.layer4[1].conv1, merged.layer4[1].bn1, w_l4b1c1, bn_l4b1c1)
        _set_conv_bn(merged.layer4[1].conv2, merged.layer4[1].bn2, w_l4b1c2, bn_l4b1c2)

        # FC
        merged.fc = nn.Linear(C3, fd)
        merged.fc.weight.copy_(mfw); merged.fc.bias.copy_(mfb)

    n_params = sum(p.numel() for p in merged.parameters())
    print(f"    合并完成: {n_params:,} params, dims={dims}")
    return merged.to(device) if device else merged


# ═══════════════════════════════════════════════════════════
# 快速测试
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Testing ResNet-18 Filter Merge...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建 5 个随机 backbone
    bbs = [ResNet18Backbone(256) for _ in range(5)]
    for bb in bbs:
        bb.eval()

    # 合并
    merged = union_aggregate_resnet18(bbs, fd=256, thr=0.95, device=device)

    # 测试 forward
    x = torch.randn(4, 3, 32, 32).to(device)
    with torch.no_grad():
        out = merged(x)
    print(f"  Input: {x.shape} → Output: {out.shape}")
    print(f"  Output norm: {out.norm(dim=1).mean():.4f}")
    print("  OK!")