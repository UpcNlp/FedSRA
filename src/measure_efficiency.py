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

def measure_inference(method, K, nc=10, fd=256, n_test=10000, bs=256, warmup=3, repeat=5, stream=False):
    """测量 server inference time + GPU peak memory.
    stream=True (仅对 'ours' 有意义): backbone 逐个搬上 GPU -> 前向 -> 搬回 CPU,
    使 GPU 同一时刻只驻留 1 个 backbone, 峰值显存与 K 无关 (O(1) vs O(K))。"""
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
        # K backbone forward -> per-client znorm -> coverage-weighted sum -> L2 -> ETF sim.
        # NO conditional experts: the deployed union path is backbone -> znorm -> etf only,
        # so we must not allocate experts (they would inflate peak memory unfairly).
        etf = F.normalize(torch.randn(nc, fd), dim=1)
        if stream:
            bbs = [ResNet18Backbone(fd).eval() for _ in range(K)]        # stay on CPU
        else:
            bbs = [ResNet18Backbone(fd).to(device).eval() for _ in range(K)]

        def bb_feat(bb, x):
            xx = F.relu(bb.bn1(bb.conv1(x)))
            xx = bb.layer1(xx); xx = bb.layer2(xx)
            xx = bb.layer3(xx); xx = bb.layer4(xx)
            return bb.fc(bb.pool(xx).flatten(1))

        def run_once():
            all_raw = []
            for bb in bbs:
                if stream: bb.to(device)
                feats = []
                for x, _ in test_loader:
                    feats.append(bb_feat(bb, x.to(device)).cpu())
                all_raw.append(torch.cat(feats, 0))
                if stream:
                    bb.cpu(); torch.cuda.empty_cache()
            # znorm + sqrt(n) aggregation (mirrors run_znorm_scalability.py:137-148)
            feat = torch.zeros(n_test, fd)
            for f in all_raw:
                f_z = (f - f.mean(0, keepdim=True)) / (f.std(0, keepdim=True) + 1e-8)
                feat += f_z * np.sqrt(5000)
            feat_n = F.normalize(feat / (np.sqrt(5000) * K), dim=1)
            return feat_n @ etf.T

        for _ in range(warmup):
            with torch.no_grad(): run_once()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(repeat):
            t0 = time.time()
            with torch.no_grad(): run_once()
            times.append(time.time() - t0)
        for bb in bbs: bb.cpu()

    else:
        raise ValueError(f"Unknown method: {method}")

    torch.cuda.empty_cache()
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    avg_time = np.mean(times)
    std_time = np.std(times)

    # upload / communication cost (fp32 param bytes): single-model methods send 1 model,
    # ensemble/fafi/ours send all K client models/backbones.
    def _mb(m): return sum(p.numel() for p in m.parameters()) * 4 / 1024**2
    if method in ('ofedavg', 'dense', 'coboosting'):
        upload_mb = _mb(ResNet18(nc))
    elif method in ('ensemble', 'fafi'):
        upload_mb = K * _mb(ResNet18(nc))
    else:  # ours
        upload_mb = K * _mb(ResNet18Backbone(fd))

    return {
        'method': method, 'K': K, 'bs': bs, 'n_test': n_test, 'stream': stream,
        'time_total_s_mean': round(avg_time, 4),
        'time_total_s_std': round(std_time, 4),
        'ms_per_img': round(avg_time / n_test * 1000, 4),
        'throughput_img_s': round(n_test / avg_time, 1),
        'gpu_peak_mb': round(peak_mb, 1),
        'upload_mb': round(upload_mb, 1),
    }


if __name__ == '__main__':
    import os
    os.makedirs('results', exist_ok=True)
    results = []
    methods = ['ofedavg', 'ensemble', 'dense', 'coboosting', 'fafi', 'ours']

    # (bs, n_test, repeat): batched throughput at bs=256; per-sample latency at bs=1
    # (bs=1 uses a smaller n_test/repeat so K=50 stays tractable; everything is reported per-image).
    configs = [(256, 10000, 5), (1, 2000, 3)]

    for K in [5, 10, 20, 50]:
        print(f"\n{'='*60}\n  K = {K}\n{'='*60}")
        for bs, n_test, repeat in configs:
            for method in methods:
                # 'ours' also gets a streaming-backbone variant -> O(1) peak memory
                variants = [False, True] if method == 'ours' else [False]
                for stream in variants:
                    tag = method + ('-stream' if stream else '')
                    try:
                        r = measure_inference(method, K, n_test=n_test, bs=bs,
                                              repeat=repeat, stream=stream)
                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower():
                            print(f"  {tag:>14s} [bs={bs:>3d}]: OOM at K={K}")
                            torch.cuda.empty_cache()
                            r = {'method': method, 'K': K, 'bs': bs, 'stream': stream,
                                 'ms_per_img': -1, 'throughput_img_s': -1,
                                 'gpu_peak_mb': -1, 'upload_mb': -1}
                        else:
                            raise
                    if r.get('ms_per_img', -1) >= 0:
                        print(f"  {tag:>14s} [bs={bs:>3d}]: "
                              f"{r['ms_per_img']:7.3f} ms/img, {r['throughput_img_s']:8.0f} img/s, "
                              f"mem={r['gpu_peak_mb']:7.1f}MB, up={r['upload_mb']:7.1f}MB")
                    results.append(r)
            torch.cuda.empty_cache()

    json.dump(results, open('results/efficiency_measurements.json', 'w'), indent=2)
    print(f"\nSaved: results/efficiency_measurements.json ({len(results)} rows)")
