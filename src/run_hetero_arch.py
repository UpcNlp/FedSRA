"""
run_hetero_arch.py  (NEW — does not modify any existing file)
=============================================================
模型异构实验 (只跑 Ours). K=5 固定, 每个 client 不同架构,
服务器端用主方法 znorm+sqrt(n) 关系信号 + per-class expert 内在信号融合
(与 run_znorm_scalability.py 完全同口径, 仅把同构 ResNet18 换成异构 backbone).

  --dataset cifar10|cifar100
  --tier    mild|strong        (见 hetero_backbones.TIER_POOLS)
  --alpha   0.05|0.1|0.3|0.5
  --seed    42 (default)
  --resume  若 checkpoint 已存在则跳过训练直接评估
  --smoke   2-epoch 冒烟测试 (验证端到端管线)

数据划分只由 np.random(seed) 决定 (dirichlet_split), 与 tier/架构无关 →
同一 (dataset, alpha, seed) 下 mild/strong 共享完全相同的数据划分, 是受控对比.

输出: results/hetero_<ds>_<tier>_a<α>_s<seed>.json
"""
import torch
import torch.nn.functional as F
import numpy as np
import json
import time
import os
import argparse

from rebuild8 import (prepare_data, generate_etf, train_bb, train_experts,
                      cross_client_per_client_logits, expert_original,
                      ConditionalExpert, device)
from rebuild8_cifar100 import prepare_data_cifar100, train_experts_cifar100
from hetero_backbones import make_backbone, TIER_POOLS


def save_models(bbs, client_exps, ccc, arch_names, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for k in range(len(bbs)):
        if bbs[k] is None:
            continue
        cd = os.path.join(save_dir, f"client_{k}")
        os.makedirs(cd, exist_ok=True)
        torch.save(bbs[k].state_dict(), os.path.join(cd, "backbone.pt"))
        for c, exp in client_exps[k].items():
            torch.save(exp.state_dict(), os.path.join(cd, f"expert_{c}.pt"))
    torch.save(dict(ccc), os.path.join(save_dir, "ccc.pt"))
    json.dump(arch_names, open(os.path.join(save_dir, "arch_map.json"), "w"))
    print(f"  saved -> {save_dir}")


def _atomic_torch_save(obj, path):
    """Write a checkpoint atomically so an interrupted job cannot look complete."""
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_client(k, bb, exps, save_dir):
    """Persist one fully trained client immediately (backbone + every expert)."""
    cd = os.path.join(save_dir, f"client_{k}")
    os.makedirs(cd, exist_ok=True)
    _atomic_torch_save(bb.state_dict(), os.path.join(cd, "backbone.pt"))
    for c, exp in exps.items():
        _atomic_torch_save(exp.state_dict(), os.path.join(cd, f"expert_{c}.pt"))
    marker = os.path.join(cd, "COMPLETE.json")
    tmp = f"{marker}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump({'client': k, 'n_experts': len(exps)}, f, indent=2)
    os.replace(tmp, marker)


def load_client(k, arch_name, save_dir, NL, FD=256):
    cd = os.path.join(save_dir, f"client_{k}")
    bb = make_backbone(arch_name, FD)
    bb.load_state_dict(torch.load(os.path.join(cd, "backbone.pt"),
                                  map_location='cpu', weights_only=True))
    exps = {}
    for c in range(NL):
        ep = os.path.join(cd, f"expert_{c}.pt")
        if os.path.exists(ep):
            exp = ConditionalExpert(FD, FD, 128, 32)
            exp.load_state_dict(torch.load(ep, map_location='cpu', weights_only=True))
            exps[c] = exp
    return bb, exps


def all_ckpts_exist(save_dir, NC):
    if not os.path.exists(os.path.join(save_dir, "arch_map.json")):
        return False
    # New resumable runs use per-client terminal markers.  Legacy completed runs
    # have no markers, so retain backward compatibility only when none exist.
    has_markers = any(os.path.exists(os.path.join(save_dir, f"client_{k}", "COMPLETE.json"))
                      for k in range(NC))
    for k in range(NC):
        if not os.path.exists(os.path.join(save_dir, f"client_{k}", "backbone.pt")):
            return False
        if has_markers and not os.path.exists(
                os.path.join(save_dir, f"client_{k}", "COMPLETE.json")):
            return False
    return True


def load_models(save_dir, NC, NL, FD=256):
    arch_names = json.load(open(os.path.join(save_dir, "arch_map.json")))
    bbs = []; client_exps = []
    for k in range(NC):
        cd = os.path.join(save_dir, f"client_{k}")
        if not os.path.exists(os.path.join(cd, "backbone.pt")):
            bbs.append(None); client_exps.append({}); continue
        bb = make_backbone(arch_names[k], FD)
        bb.load_state_dict(torch.load(f"{cd}/backbone.pt", map_location='cpu', weights_only=True))
        bbs.append(bb)
        exps = {}
        for c in range(NL):
            ep = f"{cd}/expert_{c}.pt"
            if os.path.exists(ep):
                exp = ConditionalExpert(FD, FD, 128, 32)
                exp.load_state_dict(torch.load(ep, map_location='cpu', weights_only=True))
                exps[c] = exp
        client_exps.append(exps)
    ccc = torch.load(os.path.join(save_dir, "ccc.pt"), weights_only=False)
    return bbs, client_exps, ccc, arch_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['cifar10', 'cifar100'], required=True)
    ap.add_argument('--tier', choices=['mild', 'strong'], required=True)
    ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--save_dir', default=None,
                    help='checkpoint directory override (recommended for versioned runs)')
    ap.add_argument('--out', default=None,
                    help='result JSON override (recommended for versioned runs)')
    args = ap.parse_args()

    DS = args.dataset; TIER = args.tier; ALPHA = args.alpha; SEED = args.seed
    NC = 5; FD = 256; LD = 32
    NL = 10 if DS == 'cifar10' else 100
    if args.smoke:
        EPB = EPE = 2
    else:
        EPB, EPE = (600, 600) if DS == 'cifar10' else (300, 200)

    torch.manual_seed(SEED); np.random.seed(SEED)
    arch_names = list(TIER_POOLS[TIER])
    assert len(arch_names) == NC, f"tier pool must have {NC} archs"

    tag = f"hetero_{DS}_{TIER}_a{ALPHA}_s{SEED}"
    suffix = "_smoke" if args.smoke else ""
    save_dir = args.save_dir or f"saved_models/{tag}{suffix}"
    print(f"\n{'='*64}\n  {tag}{suffix}  archs={arch_names}\n  NL={NL} EPB={EPB} EPE={EPE}\n{'='*64}")

    etf = generate_etf(NL, FD)
    if DS == 'cifar10':
        cal, ccl, tl, ccc = prepare_data(NC, ALPHA, NL)
    else:
        cal, ccl, tl, ccc = prepare_data_cifar100(NC, ALPHA, NL)

    # Save immutable run structure before any expensive training.  COMPLETE.json is
    # written per client only after its backbone and all available experts are safe.
    os.makedirs(save_dir, exist_ok=True)
    arch_path = os.path.join(save_dir, "arch_map.json")
    if os.path.exists(arch_path):
        old_arch = json.load(open(arch_path))
        if old_arch != arch_names:
            raise RuntimeError(f"architecture mismatch in {arch_path}: {old_arch} != {arch_names}")
    else:
        json.dump(arch_names, open(arch_path, "w"), indent=2)
    ccc_path = os.path.join(save_dir, "ccc.pt")
    if not os.path.exists(ccc_path):
        _atomic_torch_save(dict(ccc), ccc_path)

    if args.resume and all_ckpts_exist(save_dir, NC):
        bbs, client_exps, ccc, arch_names = load_models(save_dir, NC, NL, FD)
        print(f"  [resume] loaded {sum(1 for b in bbs if b is not None)}/{NC} clients")
        train_time = 0.0
    else:
        bbs = []; client_exps = []
        t0 = time.time()
        for k in range(NC):
            cls = sorted(ccc.get(k, {}).keys())
            if not cls:
                print(f"  Client {k}: EMPTY"); bbs.append(None); client_exps.append({}); continue
            marker = os.path.join(save_dir, f"client_{k}", "COMPLETE.json")
            if args.resume and os.path.exists(marker):
                bb, exps = load_client(k, arch_names[k], save_dir, NL, FD)
                bbs.append(bb); client_exps.append(exps)
                print(f"  [resume] Client {k}: loaded backbone + {len(exps)} experts")
                continue
            print(f"\n  Client {k} [{arch_names[k]}]: {len(cls)} cls, {sum(ccc[k].values())} samp")
            bb = make_backbone(arch_names[k], FD)
            bb = train_bb(bb, cal[k], cls, etf, EPB)
            if DS == 'cifar10':
                exps = train_experts(bb, ccl[k], cls, etf, NL, FD, LD, EPE)
            else:
                exps = train_experts_cifar100(bb, ccl[k], cls, etf, ccc[k],
                                               nc=NL, fdim=FD, ldim=LD, epochs=EPE)
            bbs.append(bb.cpu())
            client_exps.append({c: e.cpu() for c, e in exps.items()})
            save_client(k, bbs[-1], client_exps[-1], save_dir)
            print(f"  [checkpoint] Client {k}: backbone + {len(exps)} experts")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        train_time = time.time() - t0
        save_models(bbs, client_exps, ccc, arch_names, save_dir)

    # ───── 评估: 主方法 znorm+sqrt(n) union + per-class expert 融合 ─────
    ed = etf.to(device)
    N = 10000
    all_raw = []; all_labels = []
    errors = torch.full((NC, N, NL), float('inf'))
    for k in range(NC):
        if bbs[k] is None:
            all_raw.append(None); continue
        bbs[k] = bbs[k].to(device); bbs[k].eval()
        for c in client_exps[k]:
            client_exps[k][c] = client_exps[k][c].to(device)
        feats = []; offset = 0
        with torch.no_grad():
            for x, y in tl:
                x = x.to(device); bs = x.size(0)
                feat = bbs[k]._feat(x)                 # RAW pre-normalization 特征
                feats.append(feat.cpu())
                f_norm = F.normalize(feat, dim=1)
                for c, exp in client_exps[k].items():
                    fr, _ = exp(f_norm, ed[c].unsqueeze(0).expand(bs, -1))
                    errors[k, offset:offset + bs, c] = ((f_norm - fr) ** 2).mean(1).cpu()
                if k == 0:
                    all_labels.append(y)
                offset += bs
        all_raw.append(torch.cat(feats, 0))
        bbs[k] = bbs[k].cpu()
        for c in client_exps[k]:
            client_exps[k][c] = client_exps[k][c].cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    labels = torch.cat(all_labels).numpy()

    sample_count = {}
    for k in range(NC):
        for c in client_exps[k]:
            sample_count[(k, c)] = ccc.get(k, {}).get(c, 0)

    # znorm + sqrt(n) 关系聚合 (与 run_znorm_scalability 一致)
    feat = torch.zeros(N, FD); w_sum = 0.0
    for k in range(NC):
        if all_raw[k] is None:
            continue
        f = all_raw[k]
        f_z = (f - f.mean(0, keepdim=True)) / (f.std(0, keepdim=True) + 1e-8)
        w = np.sqrt(sum(ccc.get(k, {}).values()))
        feat += f_z * w; w_sum += w
    feat_n = F.normalize(feat / w_sum, dim=1)
    union_logits = feat_n @ etf.T
    union_preds = union_logits.argmax(1).numpy()
    acc_union = (union_preds == labels).mean()

    sl, _ = union_logits.sort(dim=1, descending=True)
    data = {
        'errors': errors, 'union_logits': union_logits, 'union_preds': union_preds,
        'union_margin': (sl[:, 0] - sl[:, 1]).numpy(), 'labels': labels,
        'sample_count': sample_count, 'K': NC, 'N': N, 'nc': NL,
    }
    acc_expert = (expert_original(data) == labels).mean()
    acc_full = {}
    for a in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        acc_full[a] = (cross_client_per_client_logits(data, alpha=a, min_n=10) == labels).mean()
    best_a = max(acc_full, key=acc_full.get)
    best_full = acc_full[best_a]

    coverage = np.zeros(NL)
    for c in range(NL):
        coverage[c] = sum(1 for k in range(NC) if ccc.get(k, {}).get(c, 0) >= 10) / NC
    af_dyn = 0.2 * coverage.mean()
    acc_dyn = (cross_client_per_client_logits(data, alpha=af_dyn, min_n=10) == labels).mean()

    # 异构程度量化: 各 client backbone 参数量 + 变异系数 CV
    pcounts = [int(sum(p.numel() for p in make_backbone(a, FD).parameters())) for a in arch_names]
    pmean = float(np.mean(pcounts))
    pcv = float(np.std(pcounts) / pmean) if pmean > 0 else 0.0

    print(f"\n{'='*64}")
    print(f"  {tag}{suffix}")
    print(f"  union={acc_union*100:.2f}  expert={acc_expert*100:.2f}  "
          f"best={best_full*100:.2f}(a={best_a})  dynamic={acc_dyn*100:.2f}")
    print(f"  param_counts={pcounts}  CV={pcv:.4f}")
    print(f"{'='*64}")

    os.makedirs('results', exist_ok=True)
    out = {
        'experiment': 'hetero_arch', 'dataset': DS, 'tier': TIER,
        'alpha': ALPHA, 'seed': SEED, 'n_clients': NC, 'n_classes': NL,
        'architectures': arch_names, 'param_counts': pcounts, 'param_cv': round(pcv, 4),
        'epochs_bb': EPB, 'epochs_exp': EPE, 'train_time': train_time,
        'acc_union': float(acc_union), 'acc_expert': float(acc_expert),
        'acc_full': {str(k): float(v) for k, v in acc_full.items()},
        'best_alpha': float(best_a), 'best_acc': float(best_full),
        'acc_dynamic': float(acc_dyn), 'af_dynamic': round(float(af_dyn), 4),
        'avg_coverage': round(float(coverage.mean()), 4), 'smoke': args.smoke,
    }
    path = args.out or f"results/{tag}{suffix}.json"
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    json.dump(out, open(path, 'w'), indent=2)
    print(f"  saved: {path}")


if __name__ == '__main__':
    main()
