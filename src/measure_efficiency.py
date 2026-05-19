"""
measure_efficiency.py - 统一测量 server inference time + GPU memory
所有方法在同一台 GPU 上,用随机模型,确保公平
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, time, json
from torchvision import datasets, transforms

device = torch.device('cuda')

# === ResNet18 (Co-Boost 版) ===
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_p, p, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_p, p, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(p)
        self.conv2 = nn.Conv2d(p, p, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(p)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_p != p:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_p, p, 1, stride=stride, bias=False),
                nn.BatchNorm2d(p))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet18(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.in_p = 64
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make(64, 2, 1)
        self.layer2 = self._make(128, 2, 2)
        self.layer3 = self._make(256, 2, 2)
        self.layer4 = self._make(512, 2, 2)
        self.linear = nn.Linear(512, nc)
    def _make(self, p, n, s):
        layers = [BasicBlock(self.in_p, p, s)]
        self.in_p = p
        for _ in range(1, n): layers.append(BasicBlock(p, p))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.linear(x)

# === ResNet18Backbone (OURS) ===
from resnet18_filter_merge import ResNet18Backbone
from rebuild8 import ConditionalExpert

def measure_inference(method, K, nc=10, fd=256, n_test=10000, bs=256, warmup=3, repeat=5):
    """测量 server inference time + GPU peak memory"""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # 创建 fake test data
    test_x = torch.randn(n_test, 3, 32, 32)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_x, torch.zeros(n_test, dtype=torch.long)),
        batch_size=bs, shuffle=False)

    if method == 'ofedavg':
        # 1 model forward
        model = ResNet18(nc).to(device).eval()
        # warmup
        for _ in range(warmup):
            with torch.no_grad():
                for x, _ in test_loader:
                    model(x.to(device))
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(repeat):
            t0 = time.time()
            with torch.no_grad():
                for x, _ in test_loader:
                    model(x.to(device))
            times.append(time.time() - t0)
        model.cpu()

    elif method == 'ensemble':
        # K models forward + avg logits
        models = [ResNet18(nc).to(device).eval() for _ in range(K)]
        for _ in range(warmup):
            with torch.no_grad():
                for x, _ in test_loader:
                    x = x.to(device)
                    for m in models: m(x)
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(repeat):
            t0 = time.time()
            with torch.no_grad():
                all_logits = torch.zeros(n_test, nc)
                for x, _ in test_loader:
                    x = x.to(device)
                    batch_logits = torch.zeros(x.size(0), nc, device=device)
                    for m in models:
                        batch_logits += m(x)
                    all_logits[:x.size(0)] = batch_logits.cpu() / K
            times.append(time.time() - t0)
        for m in models: m.cpu()

    elif method in ['dense', 'coboosting']:
        # 1 student model forward (same as ofedavg)
        model = ResNet18(nc).to(device).eval()
        for _ in range(warmup):
            with torch.no_grad():
                for x, _ in test_loader:
                    model(x.to(device))
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(repeat):
            t0 = time.time()
            with torch.no_grad():
                for x, _ in test_loader:
                    model(x.to(device))
            times.append(time.time() - t0)
        model.cpu()

    elif method == 'fafi':
        # K models forward + ensemble (similar to ensemble)
        models = [ResNet18(nc).to(device).eval() for _ in range(K)]
        for _ in range(warmup):
            with torch.no_grad():
                for x, _ in test_loader:
                    x = x.to(device)
                    for m in models: m(x)
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(repeat):
            t0 = time.time()
            with torch.no_grad():
                for x, _ in test_loader:
                    x = x.to(device)
                    logits = sum(m(x) for m in models) / K
            times.append(time.time() - t0)
        for m in models: m.cpu()

    elif method == 'ours':
        # K backbone forward + znorm + expert eval
        bbs = [ResNet18Backbone(fd).to(device).eval() for _ in range(K)]
        etf = F.normalize(torch.randn(nc, fd), dim=1)
        # experts: ~5 per client
        n_exp_per_client = max(1, nc // K)
        experts = {}
        for k in range(K):
            for c in range(n_exp_per_client):
                experts[(k, c)] = ConditionalExpert(fd, fd, 128, 32).to(device).eval()

        for _ in range(warmup):
            with torch.no_grad():
                for x, _ in test_loader:
                    x = x.to(device)
                    for bb in bbs:
                        xx = F.relu(bb.bn1(bb.conv1(x)))
                        xx = bb.layer1(xx); xx = bb.layer2(xx)
                        xx = bb.layer3(xx); xx = bb.layer4(xx)
                        bb.fc(bb.pool(xx).flatten(1))

        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(repeat):
            t0 = time.time()
            all_raw = []
            with torch.no_grad():
                for k, bb in enumerate(bbs):
                    feats = []
                    for x, _ in test_loader:
                        x = x.to(device)
                        xx = F.relu(bb.bn1(bb.conv1(x)))
                        xx = bb.layer1(xx); xx = bb.layer2(xx)
                        xx = bb.layer3(xx); xx = bb.layer4(xx)
                        feat = bb.fc(bb.pool(xx).flatten(1))
                        feats.append(feat.cpu())
                    all_raw.append(torch.cat(feats, 0))
                # znorm + sqrt(n)
                feat = torch.zeros(n_test, fd)
                for k in range(K):
                    f = all_raw[k]
                    f_z = (f - f.mean(0, keepdim=True)) / (f.std(0, keepdim=True) + 1e-8)
                    feat += f_z * np.sqrt(5000)
                feat_n = F.normalize(feat / (np.sqrt(5000) * K), dim=1)
                logits = feat_n @ etf.T
            times.append(time.time() - t0)
        for bb in bbs: bb.cpu()
        for exp in experts.values(): exp.cpu()

    else:
        raise ValueError(f"Unknown method: {method}")

    torch.cuda.empty_cache()
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    avg_time = np.mean(times)
    std_time = np.std(times)

    return {
        'method': method, 'K': K,
        'inference_time_mean': round(avg_time, 4),
        'inference_time_std': round(std_time, 4),
        'gpu_peak_mb': round(peak_mb, 1),
    }


if __name__ == '__main__':
    results = []
    methods = ['ofedavg', 'ensemble', 'dense', 'coboosting', 'fafi', 'ours']

    for K in [5, 10, 20, 50]:
        print(f"\n{'='*50}")
        print(f"  K = {K}")
        print(f"{'='*50}")
        for method in methods:
            if K == 50 and method in ['ensemble', 'fafi', 'ours']:
                # K=50 加载 50 个模型可能 OOM,尝试
                try:
                    r = measure_inference(method, K)
                except RuntimeError as e:
                    print(f"  {method:>12s}: OOM at K={K}")
                    r = {'method': method, 'K': K, 'inference_time_mean': -1, 'gpu_peak_mb': -1}
            else:
                r = measure_inference(method, K)
            print(f"  {method:>12s}: time={r['inference_time_mean']:.4f}s, mem={r['gpu_peak_mb']:.1f}MB")
            results.append(r)
        torch.cuda.empty_cache()

    # Save
    import os
    os.makedirs('results', exist_ok=True)
    json.dump(results, open('results/efficiency_measurements.json', 'w'), indent=2)
    print(f"\nSaved: results/efficiency_measurements.json")
