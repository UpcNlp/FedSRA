"""Ensemble + Expert fusion: per-client softmax average + per-class expert reconstruction, fused via alpha_f."""
import torch, torch.nn.functional as F, sys, json, argparse, os
sys.path.insert(0, "/public/home/dongshou/fedETF/ETF-pesuade")
from rebuild8 import prepare_data, generate_etf, device, ConditionalExpert
from resnet18_filter_merge import ResNet18Backbone

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="cifar10")
ap.add_argument("--alpha", type=float, required=True)
ap.add_argument("--n_clients", type=int, default=5)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--temp", type=float, default=0.1)
ap.add_argument("--alpha_f", type=float, default=0.3)
a = ap.parse_args()
NL = 10 if a.dataset == "cifar10" else 100; FD = 256
cands = [f"saved_models/ablation_{a.dataset}/J_a{a.alpha}_k{a.n_clients}_s{a.seed}",
         f"saved_models/ablation_{a.dataset}_a{a.alpha}_k{a.n_clients}_s{a.seed}"]
sd = next((d for d in cands if os.path.exists(d)), None)
print(f"using save_dir={sd}")
torch.manual_seed(a.seed); etf = generate_etf(NL, FD).to(device)
if a.dataset == "cifar10":
    _, _, tl, _ = prepare_data(a.n_clients, a.alpha, NL)
else:
    from rebuild8_cifar100 import prepare_data_cifar100
    _, _, tl, _ = prepare_data_cifar100(a.n_clients, a.alpha, NL)
N = 10000
per_client_softmax = []
per_client_errors = torch.full((a.n_clients, N, NL), float('inf'))
labels = None
for k in range(a.n_clients):
    bb = ResNet18Backbone(FD)
    bb.load_state_dict(torch.load(f"{sd}/client_{k}/backbone.pt", map_location='cpu', weights_only=True))
    bb = bb.to(device).eval()
    experts = {}
    for c in range(NL):
        ep = f"{sd}/client_{k}/expert_{c}.pt"
        if os.path.exists(ep):
            ex = ConditionalExpert(FD, FD, 128, 32)
            ex.load_state_dict(torch.load(ep, map_location='cpu', weights_only=True))
            experts[c] = ex.to(device).eval()
    sm = []; lab = []; offset = 0
    with torch.no_grad():
        for x, y in tl:
            x = x.to(device); bs = x.size(0)
            xx = F.relu(bb.bn1(bb.conv1(x))); xx = bb.layer1(xx); xx = bb.layer2(xx)
            xx = bb.layer3(xx); xx = bb.layer4(xx); xx = bb.pool(xx).flatten(1)
            feat = bb.fc(xx); feat_n = F.normalize(feat, dim=1)
            logits = feat_n @ etf.T / a.temp
            sm.append(F.softmax(logits, dim=1).cpu())
            if labels is None: lab.append(y)
            for c, exp in experts.items():
                fr, _ = exp(feat_n, etf[c].unsqueeze(0).expand(bs, -1))
                per_client_errors[k, offset:offset+bs, c] = ((feat_n - fr)**2).mean(1).cpu()
            offset += bs
    per_client_softmax.append(torch.cat(sm, 0))
    if labels is None: labels = torch.cat(lab, 0).numpy()
    bb = bb.cpu()
    for c in experts: experts[c] = experts[c].cpu()
    torch.cuda.empty_cache()

ens_logits = torch.stack(per_client_softmax, 0).mean(0)        # (N,NL) ensemble softmax
ens_acc = float((ens_logits.argmax(1).numpy() == labels).mean())
expert_signal = -per_client_errors.min(dim=0).values            # (N,NL); -inf for unseen classes
mask_inf = torch.isinf(expert_signal)
# Replace -inf with row min for stable z-norm
safe_expert = expert_signal.clone()
safe_expert[mask_inf] = safe_expert[~mask_inf].min() if (~mask_inf).any() else 0
def znorm(x):
    return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)
ens_z = znorm(ens_logits); exp_z = znorm(safe_expert)
fused = ens_z + a.alpha_f * exp_z
fused_acc = float((fused.argmax(1).numpy() == labels).mean())
print(f"RESULT  {a.dataset}  a{a.alpha}  K{a.n_clients}  ENS={ens_acc*100:.2f}  ENS+EXPERT={fused_acc*100:.2f}  delta={(fused_acc-ens_acc)*100:+.2f}")
os.makedirs("results", exist_ok=True)
tag = "" if a.dataset == "cifar10" else f"{a.dataset}_"
json.dump({"dataset":a.dataset,"alpha":a.alpha,"K":a.n_clients,"ensemble_acc":ens_acc,"ensemble_expert_acc":fused_acc,"alpha_f":a.alpha_f},
          open(f"results/ens_expert_{tag}a{a.alpha}_k{a.n_clients}_s{a.seed}.json","w"), indent=2)
