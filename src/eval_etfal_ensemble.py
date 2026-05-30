"""Build-up step ②: ETF backbone trained with etf_al ONLY + LOGIT ENSEMBLE inference."""
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
NL = 10 if a.dataset == "cifar10" else 100; FD = 256
sd = f"saved_models/ablation_{a.dataset}/I_a{a.alpha}_k{a.n_clients}_s{a.seed}"
if not os.path.exists(sd):
    sd = f"saved_models/ablation_{a.dataset}_a{a.alpha}_k{a.n_clients}_s{a.seed}_I"
print(f"using save_dir={sd}")
torch.manual_seed(a.seed); etf = generate_etf(NL, FD).to(device)
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
            xx = F.relu(bb.bn1(bb.conv1(x))); xx = bb.layer1(xx); xx = bb.layer2(xx)
            xx = bb.layer3(xx); xx = bb.layer4(xx); xx = bb.pool(xx).flatten(1)
            feat = bb.fc(xx); feat_n = F.normalize(feat, dim=1)
            logits = feat_n @ etf.T / a.temp
            ls.append(F.softmax(logits, dim=1).cpu())
            if labels is None: lab.append(y)
    per_client_softmax.append(torch.cat(ls, 0))
    if labels is None: labels = torch.cat(lab, 0).numpy()
    bb = bb.cpu(); torch.cuda.empty_cache()
avg = torch.stack(per_client_softmax, 0).mean(0)
acc = float((avg.argmax(1).numpy() == labels).mean())
print(f"RESULT  {a.dataset}  a{a.alpha}  K{a.n_clients}  ETFal+ENSEMBLE  acc={acc*100:.2f}")
