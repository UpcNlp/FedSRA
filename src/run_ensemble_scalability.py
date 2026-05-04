"""
Scalability Baseline: Ensemble (平均所有客户端的预测 logits)
对每个 (α, K) 组合: 训练 K 个独立 ResNet-18 (CE loss), 测试时平均 logits
"""
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np, json, time, os, argparse
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

from resnet18_filter_merge import ResNet18Backbone

def dirichlet_split(dataset, n_clients, alpha, n_classes=10):
    targets = np.array(dataset.targets)
    ci = defaultdict(list)
    for idx, l in enumerate(targets): ci[l].append(idx)
    client_idx = defaultdict(list)
    for c in range(n_classes):
        idxs = np.array(ci[c]); np.random.shuffle(idxs)
        props = np.random.dirichlet([alpha]*n_clients)
        props = (props*len(idxs)).astype(int); props[-1] = len(idxs)-props[:-1].sum()
        s = 0
        for k in range(n_clients):
            e = s+props[k]
            if e > s: client_idx[k].extend(idxs[s:e].tolist())
            s = e
    return dict(client_idx)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--n_clients', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--n_classes', type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    NC = args.n_clients; NL = args.n_classes; ALPHA = args.alpha

    # Data
    tt = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    te = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=tt)
    test_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=te)
    cidx = dirichlet_split(train_ds, NC, ALPHA, NL)
    tl = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Ensemble Baseline: α={ALPHA}, K={NC}, seed={args.seed}")

    # Train K independent models with CE
    models = []
    t0 = time.time()
    for k in range(NC):
        print(f"  Client {k}/{NC}: {len(cidx[k])} samples")
        model = ResNet18Backbone(256).to(device)
        head = nn.Linear(256, NL).to(device)
        dl = DataLoader(Subset(train_ds, cidx[k]), batch_size=128, shuffle=True,
                       num_workers=4, pin_memory=True, drop_last=True)
        opt = torch.optim.SGD(list(model.parameters()) + list(head.parameters()),
                              lr=0.05, momentum=0.9, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        model.train(); head.train()
        for ep in range(args.epochs):
            for x, y in dl:
                x, y = x.to(device), y.to(device)
                feat = model(x)
                logits = head(feat)
                loss = F.cross_entropy(logits, y)
                opt.zero_grad(); loss.backward(); opt.step()
            sch.step()
            if (ep+1) % 100 == 0:
                print(f"    Epoch {ep+1}/{args.epochs}")

        models.append((model.cpu(), head.cpu()))
        torch.cuda.empty_cache()

    train_time = time.time() - t0

    # Ensemble inference: average logits
    all_preds = []; all_labels = []
    with torch.no_grad():
        for x, y in tl:
            x = x.to(device)
            logits_sum = torch.zeros(x.size(0), NL, device=device)
            for model, head in models:
                model = model.to(device); head = head.to(device)
                model.eval(); head.eval()
                feat = model(x)
                logits_sum += head(feat)
                model.cpu(); head.cpu()
            preds = logits_sum.argmax(1).cpu()
            all_preds.append(preds); all_labels.append(y)
            torch.cuda.empty_cache()

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    acc = (preds == labels).mean()

    # Also compute FedAvg baseline
    avg_sd_model = {}; avg_sd_head = {}
    for i, (m, h) in enumerate(models):
        sd_m = m.state_dict(); sd_h = h.state_dict()
        if i == 0:
            avg_sd_model = {k: v.float().clone() for k, v in sd_m.items()}
            avg_sd_head = {k: v.float().clone() for k, v in sd_h.items()}
        else:
            for k in avg_sd_model: avg_sd_model[k] += sd_m[k].float()
            for k in avg_sd_head: avg_sd_head[k] += sd_h[k].float()
    for k in avg_sd_model: avg_sd_model[k] /= NC
    for k in avg_sd_head: avg_sd_head[k] /= NC

    fedavg_model = ResNet18Backbone(256); fedavg_model.load_state_dict(avg_sd_model)
    fedavg_head = nn.Linear(256, NL); fedavg_head.load_state_dict(avg_sd_head)
    fedavg_model = fedavg_model.to(device); fedavg_head = fedavg_head.to(device)
    fedavg_model.eval(); fedavg_head.eval()

    all_preds_fa = []
    with torch.no_grad():
        for x, y in tl:
            x = x.to(device)
            logits = fedavg_head(fedavg_model(x))
            all_preds_fa.append(logits.argmax(1).cpu())
    fa_acc = (torch.cat(all_preds_fa).numpy() == labels).mean()

    print(f"\n  Results: Ensemble={acc*100:.2f}%, FedAvg={fa_acc*100:.2f}%, time={train_time:.0f}s")

    # Save
    os.makedirs('results', exist_ok=True)
    out = {
        'method': 'baseline_scalability',
        'alpha': ALPHA, 'n_clients': NC, 'seed': args.seed,
        'ensemble_acc': float(acc),
        'fedavg_acc': float(fa_acc),
        'train_time': train_time,
    }
    path = f"results/baseline_scale_a{ALPHA}_k{NC}_s{args.seed}.json"
    json.dump(out, open(path, 'w'), indent=2)
    print(f"  Saved: {path}")

if __name__ == '__main__':
    main()
