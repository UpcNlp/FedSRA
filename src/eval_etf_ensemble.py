"""Step 2 of build-up: ETF backbone + LOGIT ENSEMBLE inference (no feature aggregation).
Per client: logits_k = features_k @ ETF^T (per-sample); softmax; average across clients; argmax.
"""
import torch, torch.nn.functional as F, sys, json, argparse, os
sys.path.insert(0, "/public/home/dongshou/fedETF/ETF-pesuade")
from rebuild8 import prepare_data, generate_etf, device
from resnet18_filter_merge import ResNet18Backbone

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="cifar10")
ap.add_argument("--alpha", type=float, required=True)
ap.add_argument("--n_clients", type=int, default=5)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--temp", type=float, default=0.1)
a = ap.parse_args()

NL = 10 if a.dataset == "cifar10" else 100
FD = 256
candidates = [
    f"saved_models/ablation_{a.dataset}_a{a.alpha}_k{a.n_clients}_s{a.seed}",
    f"saved_models/ablation_{a.dataset}/J_a{a.alpha}_k{a.n_clients}_s{a.seed}",
]
sd = next((d for d in candidates if os.path.exists(d)), None)
print(f"using save_dir={sd}")

torch.manual_seed(a.seed)
etf = generate_etf(NL, FD).to(device)

if a.dataset == "cifar10":
    _, _, tl, _ = prepare_data(a.n_clients, a.alpha, NL)
else:
    from rebuild8_cifar100 import prepare_data_cifar100
    _, _, tl, _ = prepare_data_cifar100(a.n_clients, a.alpha, NL)

per_client_softmax = []; labels = None
for k in range(a.n_clients):
    bb = ResNet18Backbone(FD)
    bb.load_state_dict(torch.load(f"{sd}/client_{k}/backbone.pt", map_location='cpu', weights_only=True))
    bb = bb.to(device).eval()
    ls = []; lab = []
    with torch.no_grad():
        for x, y in tl:
            x = x.to(device)
            xx = F.relu(bb.bn1(bb.conv1(x)))
            xx = bb.layer1(xx); xx = bb.layer2(xx)
            xx = bb.layer3(xx); xx = bb.layer4(xx)
            xx = bb.pool(xx).flatten(1)
            feat = bb.fc(xx)
            feat_n = F.normalize(feat, dim=1)        # ETF logits use unit features (standard)
            logits = feat_n @ etf.T / a.temp
            ls.append(F.softmax(logits, dim=1).cpu())
            if labels is None: lab.append(y)
    per_client_softmax.append(torch.cat(ls, 0))
    if labels is None: labels = torch.cat(lab, 0).numpy()
    bb = bb.cpu(); torch.cuda.empty_cache()

avg = torch.stack(per_client_softmax, 0).mean(0)
acc = float((avg.argmax(1).numpy() == labels).mean())
print(f"RESULT  {a.dataset}  a{a.alpha}  K{a.n_clients}  ETF+ENSEMBLE  acc={acc*100:.2f}")
out = {"dataset": a.dataset, "alpha": a.alpha, "K": a.n_clients, "method": "ETF+Ensemble", "acc": acc}
os.makedirs("results", exist_ok=True)
tag = "" if a.dataset == "cifar10" else f"{a.dataset}_"
json.dump(out, open(f"results/etf_ensemble_{tag}a{a.alpha}_k{a.n_clients}_s{a.seed}.json", "w"), indent=2)
