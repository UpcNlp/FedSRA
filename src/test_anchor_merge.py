"""
Anchor-based Filter Alignment for ResNet-18
对比: Filter Merge vs Anchor Alignment vs Simple Avg
"""
import torch, torch.nn as nn, torch.nn.functional as F
import copy, time, numpy as np
from resnet18_filter_merge import ResNet18Backbone, union_aggregate_resnet18
from rebuild8 import (prepare_data, generate_etf, train_bb, train_experts,
                       compute_stats, precompute_all, device, USE_BF16)

def anchor_aggregate_resnet18(bbs, fd=256, device_=None):
    """Anchor-based alignment: O(K) 复杂度"""
    K = len(bbs)
    anchor = bbs[0]
    
    # 收集所有需要对齐的 conv 层 (按顺序)
    conv_layers = [
        ('conv1',),
        ('layer1', 0, 'conv1'), ('layer1', 0, 'conv2'),
        ('layer1', 1, 'conv1'), ('layer1', 1, 'conv2'),
        ('layer2', 0, 'conv1'), ('layer2', 0, 'conv2'),
        ('layer2', 1, 'conv1'), ('layer2', 1, 'conv2'),
        ('layer3', 0, 'conv1'), ('layer3', 0, 'conv2'),
        ('layer3', 1, 'conv1'), ('layer3', 1, 'conv2'),
        ('layer4', 0, 'conv1'), ('layer4', 0, 'conv2'),
        ('layer4', 1, 'conv1'), ('layer4', 1, 'conv2'),
    ]
    
    # shortcut conv 层
    shortcut_layers = [
        ('layer2', 0), ('layer3', 0), ('layer4', 0),
    ]
    
    def get_module(model, path):
        m = model
        for p in path:
            if isinstance(p, int):
                m = m[p]
            else:
                m = getattr(m, p)
        return m
    
    def greedy_permutation(w_anchor, w_other):
        """输出 channel 维度的贪心匹配"""
        Co = w_anchor.size(0)
        a = F.normalize(w_anchor.reshape(Co, -1).float(), dim=1)
        o = F.normalize(w_other.reshape(Co, -1).float(), dim=1)
        sim = a @ o.T
        perm = []
        used = set()
        for i in range(Co):
            row = sim[i].clone()
            for u in used: row[u] = -2
            j = row.argmax().item()
            perm.append(j)
            used.add(j)
        return perm  # perm[anchor_idx] = other_idx
    
    def apply_output_perm(param, perm):
        """重排 output channel (dim=0)"""
        return param[perm]
    
    def apply_input_perm(param, perm):
        """重排 input channel (dim=1)"""
        return param[:, perm]
    
    print(f"\n  [Anchor Alignment] K={K} models")
    t0 = time.time()
    
    # 逐层计算每个 client 相对于 anchor 的 permutation
    # stage_perm[k] = 当前 stage 输出 channel 的 permutation
    
    aligned_sds = [bbs[0].state_dict()]  # anchor 不需要对齐
    
    for k in range(1, K):
        sd_k = bbs[k].state_dict()
        sd_anchor = bbs[0].state_dict()
        sd_aligned = {}
        
        # 逐 stage 对齐
        # Stage 0: conv1 (3→64)
        perm_conv1 = greedy_permutation(
            sd_anchor['conv1.weight'], sd_k['conv1.weight'])
        sd_aligned['conv1.weight'] = sd_k['conv1.weight'][perm_conv1]
        sd_aligned['bn1.weight'] = sd_k['bn1.weight'][perm_conv1]
        sd_aligned['bn1.bias'] = sd_k['bn1.bias'][perm_conv1]
        sd_aligned['bn1.running_mean'] = sd_k['bn1.running_mean'][perm_conv1]
        sd_aligned['bn1.running_var'] = sd_k['bn1.running_var'][perm_conv1]
        if 'bn1.num_batches_tracked' in sd_k:
            sd_aligned['bn1.num_batches_tracked'] = sd_k['bn1.num_batches_tracked']
        
        prev_perm = perm_conv1  # 64 channels
        
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            for block_idx in [0, 1]:
                prefix = f"{layer_name}.{block_idx}"
                
                # conv1: input from prev stage, output is internal
                w_a = sd_anchor[f"{prefix}.conv1.weight"]
                w_k = sd_k[f"{prefix}.conv1.weight"]
                # 重排 input channels
                w_k_in = w_k[:, prev_perm]
                # 对齐 output channels
                perm_internal = greedy_permutation(w_a, w_k_in)
                sd_aligned[f"{prefix}.conv1.weight"] = w_k_in[perm_internal]
                sd_aligned[f"{prefix}.bn1.weight"] = sd_k[f"{prefix}.bn1.weight"][perm_internal]
                sd_aligned[f"{prefix}.bn1.bias"] = sd_k[f"{prefix}.bn1.bias"][perm_internal]
                sd_aligned[f"{prefix}.bn1.running_mean"] = sd_k[f"{prefix}.bn1.running_mean"][perm_internal]
                sd_aligned[f"{prefix}.bn1.running_var"] = sd_k[f"{prefix}.bn1.running_var"][perm_internal]
                if f"{prefix}.bn1.num_batches_tracked" in sd_k:
                    sd_aligned[f"{prefix}.bn1.num_batches_tracked"] = sd_k[f"{prefix}.bn1.num_batches_tracked"]
                
                # conv2: input from internal, output must match stage output
                w_a2 = sd_anchor[f"{prefix}.conv2.weight"]
                w_k2 = sd_k[f"{prefix}.conv2.weight"]
                # 重排 input (internal perm)
                w_k2_in = w_k2[:, perm_internal]
                # conv2 的输出是 stage output，需要和 stage 一致
                # 对于 block 0 的 stage 转换，需要新的 stage perm
                # 对于 block 1 或者非 stride 块，output = prev stage output
                
                if block_idx == 0:
                    # 可能有 stride/channel 变化
                    perm_stage = greedy_permutation(w_a2, w_k2_in)
                    sd_aligned[f"{prefix}.conv2.weight"] = w_k2_in[perm_stage]
                    
                    # shortcut
                    sc_key = f"{prefix}.shortcut.0.weight"
                    if sc_key in sd_k:
                        w_sc = sd_k[sc_key]
                        w_sc_in = w_sc[:, prev_perm]
                        sd_aligned[sc_key] = w_sc_in[perm_stage]
                        for bn_attr in ['weight', 'bias', 'running_mean', 'running_var']:
                            bn_key = f"{prefix}.shortcut.1.{bn_attr}"
                            if bn_key in sd_k:
                                sd_aligned[bn_key] = sd_k[bn_key][perm_stage]
                        nbt_key = f"{prefix}.shortcut.1.num_batches_tracked"
                        if nbt_key in sd_k:
                            sd_aligned[nbt_key] = sd_k[nbt_key]
                    
                    prev_perm = perm_stage
                else:
                    # block 1: output 必须和 block 0 的 stage output 一致
                    perm_block1 = greedy_permutation(w_a2, w_k2_in)
                    sd_aligned[f"{prefix}.conv2.weight"] = w_k2_in[perm_block1]
                    prev_perm = perm_block1
                
                sd_aligned[f"{prefix}.bn2.weight"] = sd_k[f"{prefix}.bn2.weight"][prev_perm]
                sd_aligned[f"{prefix}.bn2.bias"] = sd_k[f"{prefix}.bn2.bias"][prev_perm]
                sd_aligned[f"{prefix}.bn2.running_mean"] = sd_k[f"{prefix}.bn2.running_mean"][prev_perm]
                sd_aligned[f"{prefix}.bn2.running_var"] = sd_k[f"{prefix}.bn2.running_var"][prev_perm]
                if f"{prefix}.bn2.num_batches_tracked" in sd_k:
                    sd_aligned[f"{prefix}.bn2.num_batches_tracked"] = sd_k[f"{prefix}.bn2.num_batches_tracked"]
        
        # FC: input is last stage output
        sd_aligned['fc.weight'] = sd_k['fc.weight'][:, prev_perm]
        sd_aligned['fc.bias'] = sd_k['fc.bias']
        
        aligned_sds.append(sd_aligned)
    
    # 平均所有对齐后的权重
    merged_sd = {}
    keys = aligned_sds[0].keys()
    for key in keys:
        merged_sd[key] = sum(sd[key].float() for sd in aligned_sds) / K
    
    merged = ResNet18Backbone(fd)
    merged.load_state_dict({k: v for k, v in merged_sd.items()})
    
    print(f"    对齐完成: {time.time()-t0:.1f}s")
    print(f"    参数量: {sum(p.numel() for p in merged.parameters()):,}")
    
    return merged.to(device_) if device_ else merged


# ═══════════════════════════════════════════════════════════
# 实验
# ═══════════════════════════════════════════════════════════

ALPHA = 0.5; NC = 5; NL = 10; FD = 256; LD = 32; EPB = 600; EPE = 600
torch.manual_seed(42); np.random.seed(42)
etf = generate_etf(NL, FD)
cal, ccl, tl, ccc = prepare_data(NC, ALPHA, NL)

# 训练
print("\n=== 训练 ===")
bbs = []; client_exps = []
for k in range(NC):
    cls = sorted(ccc[k].keys())
    print(f"Client {k}: {len(cls)} cls, {sum(ccc[k].values())} samp")
    bb = ResNet18Backbone(FD)
    bb = train_bb(bb, cal[k], cls, etf, EPB)
    exps = train_experts(bb, ccl[k], cls, etf, NL, FD, LD, EPE)
    bbs.append(bb.cpu())
    client_exps.append({c: exp.cpu() for c, exp in exps.items()})
    torch.cuda.empty_cache()

# A: Filter Merge
print("\n=== A: Filter Merge ===")
t0 = time.time()
ubb_fm = union_aggregate_resnet18([copy.deepcopy(bb) for bb in bbs], FD, 0.95, device)
t_fm = time.time() - t0
data_fm = precompute_all(bbs, client_exps, ubb_fm, tl, etf, ccc, NL)
acc_fm = (data_fm['union_preds'] == data_fm['labels']).mean()
print(f"  Union acc: {acc_fm:.4f}, time: {t_fm:.1f}s")

# B: Anchor Alignment
print("\n=== B: Anchor Alignment ===")
t0 = time.time()
ubb_aa = anchor_aggregate_resnet18(bbs, FD, device)
t_aa = time.time() - t0
data_aa = precompute_all(bbs, client_exps, ubb_aa, tl, etf, ccc, NL)
acc_aa = (data_aa['union_preds'] == data_aa['labels']).mean()
print(f"  Union acc: {acc_aa:.4f}, time: {t_aa:.1f}s")

# 最终对比
print(f"\n{'='*50}")
print(f"  Filter Merge:      {acc_fm:.4f}  ({t_fm:.1f}s)")
print(f"  Anchor Alignment:  {acc_aa:.4f}  ({t_aa:.1f}s)")
print(f"  差距:              {(acc_fm - acc_aa)*100:+.2f} pp")
print(f"  加速:              {t_fm/t_aa:.1f}x")
print(f"{'='*50}")
