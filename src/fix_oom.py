#!/usr/bin/env python3
"""
OOM 修复脚本 — 直接修改 rebuild8.py, rebuild8_resnet18.py, rebuild8_cifar100.py
运行: python fix_oom.py
"""
import re

def patch_file(filepath, replacements):
    with open(filepath, 'r') as f:
        content = f.read()
    for old, new in replacements:
        if old not in content:
            print(f"  ⚠️  未找到待替换内容: {old[:60]}...")
            continue
        content = content.replace(old, new)
        print(f"  ✅ 已替换: {old[:60]}...")
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"  已保存: {filepath}\n")


# ============================================================
# 1. rebuild8.py — 替换 precompute_all + 修改 bbs.append
# ============================================================
print("=" * 60)
print("修改 rebuild8.py")
print("=" * 60)

OLD_PRECOMPUTE = '''def precompute_all(bbs, client_exps, union_bb, tl, etf, ccc, nc=10):
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
    }'''

NEW_PRECOMPUTE = '''def precompute_all(bbs, client_exps, union_bb, tl, etf, ccc, nc=10):
    """内存安全版: 逐 client 计算, GPU 上只放 1 个 client 的模型"""
    K = len(bbs); ed = etf.to(device)

    # 构建样本量表
    sample_count = {}
    for k in range(K):
        for c in client_exps[k]:
            sample_count[(k, c)] = ccc[k].get(c, 0)

    # --- Pass 1: Union logits ---
    union_bb = union_bb.to(device); union_bb.eval()
    all_union_logits = []; all_labels = []
    with torch.no_grad():
        amp = (torch.amp.autocast('cuda', dtype=torch.bfloat16) if USE_BF16
               else torch.amp.autocast('cuda', enabled=False))
        for x, y in tl:
            x_dev = x.to(device, non_blocking=True)
            with amp:
                f_union = F.normalize(union_bb(x_dev), dim=1)
                all_union_logits.append(torch.mm(f_union, ed.T).float().cpu())
            all_labels.append(y)
    union_bb = union_bb.cpu(); torch.cuda.empty_cache()

    union_logits = torch.cat(all_union_logits, 0)
    labels = torch.cat(all_labels).numpy()
    N = len(labels)

    # --- Pass 2: Expert errors (逐 client, 1 个 client 在 GPU) ---
    errors = torch.full((K, N, nc), float('inf'))
    with torch.no_grad():
        amp = (torch.amp.autocast('cuda', dtype=torch.bfloat16) if USE_BF16
               else torch.amp.autocast('cuda', enabled=False))
        for k in range(K):
            bbs[k] = bbs[k].to(device); bbs[k].eval()
            for c in client_exps[k]:
                client_exps[k][c] = client_exps[k][c].to(device)
                client_exps[k][c].eval()

            offset = 0
            for x, y in tl:
                x_dev = x.to(device, non_blocking=True); bs = x.size(0)
                with amp:
                    f_k = bbs[k](x_dev)
                    for c, exp in client_exps[k].items():
                        fr, _ = exp(f_k, ed[c].unsqueeze(0).expand(bs, -1))
                        errors[k, offset:offset+bs, c] = ((f_k - fr)**2).mean(1).float().cpu()
                offset += bs

            bbs[k] = bbs[k].cpu()
            for c in client_exps[k]:
                client_exps[k][c] = client_exps[k][c].cpu()
            torch.cuda.empty_cache()

            if (k+1) % 10 == 0 or k == K-1:
                print(f"    预计算 client {k+1}/{K} done")

    # Union 预测
    union_preds = union_logits.argmax(1).numpy()
    sorted_ul, _ = union_logits.sort(dim=1, descending=True)
    union_margin = (sorted_ul[:, 0] - sorted_ul[:, 1]).numpy()

    print(f"  预计算完成: {K} clients × {N} samples")
    print(f"  Union acc: {(union_preds == labels).mean():.2%}")

    return {
        'errors': errors,           # (K, N, nc)
        'union_logits': union_logits,  # (N, nc)
        'union_preds': union_preds,  # (N,)
        'union_margin': union_margin,  # (N,)
        'labels': labels,            # (N,)
        'sample_count': sample_count,  # {(k,c): int}
        'K': K, 'N': N, 'nc': nc,
    }'''

patch_file('rebuild8.py', [
    (OLD_PRECOMPUTE, NEW_PRECOMPUTE),
    ('        bbs.append(bb); client_exps.append(exps)\n    tt = time.time() - t0\n    print(f"\\n  训练: {tt:.1f}s")',
     '        bbs.append(bb.cpu()); client_exps.append({c: exp.cpu() for c, exp in exps.items()})\n        torch.cuda.empty_cache()\n    tt = time.time() - t0\n    print(f"\\n  训练: {tt:.1f}s")'),
])


# ============================================================
# 2. rebuild8_resnet18.py — 修改 bbs.append
# ============================================================
print("=" * 60)
print("修改 rebuild8_resnet18.py")
print("=" * 60)

patch_file('rebuild8_resnet18.py', [
    ('        bbs.append(bb); client_exps.append(exps)\n    tt = time.time() - t0\n    print(f"\\n  训练: {tt:.1f}s")',
     '        bbs.append(bb.cpu()); client_exps.append({c: exp.cpu() for c, exp in exps.items()})\n        torch.cuda.empty_cache()\n    tt = time.time() - t0\n    print(f"\\n  训练: {tt:.1f}s")'),
])


# ============================================================
# 3. rebuild8_cifar100.py — 修改 bbs.append
# ============================================================
print("=" * 60)
print("修改 rebuild8_cifar100.py")
print("=" * 60)

patch_file('rebuild8_cifar100.py', [
    ('        bbs.append(bb); client_exps.append(exps)\n        print(f"    训练了 {len(exps)} 个 expert")',
     '        bbs.append(bb.cpu()); client_exps.append({c: exp.cpu() for c, exp in exps.items()})\n        print(f"    训练了 {len(exps)} 个 expert")'),
])

print("=" * 60)
print("全部修改完成！")
print("=" * 60)
