"""
hetero_backbones.py  (NEW — does not modify any existing file)
================================================================
模型异构实验用的 CIFAR-native backbone 池. 每个 backbone:
  - 输出特征维度统一为 fd=256 (投影到共享 ETF 空间)
  - 提供 ._feat(x): 返回投影头 fc 的 *原始* (pre-normalization) 特征,
    供主方法的 znorm+sqrt(n) 聚合使用 (与 run_znorm_scalability 一致)
  - forward(x) = F.normalize(_feat(x)): 供 train_bb / train_experts 使用

两档异构 (K=5, 每个 client 一个架构):
  mild   = ResNet 家族不同深度 (intra-family, 容量异构)
  strong = 跨族 CNN + ResNet     (inter-family, 对齐 Co-Boosting 的混合池思路)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────── ResNet 家族 (BasicBlock, CIFAR-adapted) ─────────────────────────
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
                nn.Conv2d(ic, oc, 1, stride=stride, bias=False), nn.BatchNorm2d(oc))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class HeteroResNet(nn.Module):
    """CIFAR-adapted ResNet, depth 由 blocks 控制 (与 resnet18_filter_merge.ResNet18Backbone 同构)."""
    def __init__(self, blocks, fd=256):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make(64, 64, blocks[0], 1)
        self.layer2 = self._make(64, 128, blocks[1], 2)
        self.layer3 = self._make(128, 256, blocks[2], 2)
        self.layer4 = self._make(256, 512, blocks[3], 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, fd)

    def _make(self, ic, oc, n, stride):
        layers = [BasicBlock(ic, oc, stride)]
        for _ in range(1, n):
            layers.append(BasicBlock(oc, oc))
        return nn.Sequential(*layers)

    def _feat(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

    def forward(self, x):
        return F.normalize(self._feat(x), dim=1)


# ───────────────────────── 跨族 CNN ─────────────────────────
class SmallCNN(nn.Module):
    """LeNet-style 浅层 CNN"""
    def __init__(self, fd=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(128 * 4 * 4, fd)

    def _feat(self, x):
        x = self.features(x)
        return self.fc(x.flatten(1))

    def forward(self, x):
        return F.normalize(self._feat(x), dim=1)


class MediumCNN(nn.Module):
    """VGG-style 中等 CNN"""
    def __init__(self, fd=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(256 * 4 * 4, fd)

    def _feat(self, x):
        x = self.features(x)
        return self.fc(x.flatten(1))

    def forward(self, x):
        return F.normalize(self._feat(x), dim=1)


class WideCNN(nn.Module):
    """宽而浅的 CNN"""
    def __init__(self, fd=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(512 * 4 * 4, fd)

    def _feat(self, x):
        x = self.features(x)
        return self.fc(x.flatten(1))

    def forward(self, x):
        return F.normalize(self._feat(x), dim=1)


class PlainCNN(nn.Module):
    """4-layer plain CNN (与 rebuild8.Backbone 同构, 但暴露 _feat)"""
    def __init__(self, fd=256):
        super().__init__()
        ch = [64, 128, 256, 256]
        self.features = nn.Sequential(
            nn.Conv2d(3, ch[0], 3, padding=1), nn.BatchNorm2d(ch[0]), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(ch[0], ch[1], 3, padding=1), nn.BatchNorm2d(ch[1]), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(ch[1], ch[2], 3, padding=1), nn.BatchNorm2d(ch[2]), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(ch[2], ch[3], 3, padding=1), nn.BatchNorm2d(ch[3]), nn.ReLU(True), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(ch[3] * 2 * 2, fd)

    def _feat(self, x):
        x = self.features(x)
        return self.fc(x.flatten(1))

    def forward(self, x):
        return F.normalize(self._feat(x), dim=1)


# ───────────────────────── 工厂 + 异构池 ─────────────────────────
_RESNET_BLOCKS = {
    'resnet10': [1, 1, 1, 1],
    'resnet14': [1, 2, 2, 1],
    'resnet18': [2, 2, 2, 2],
    'resnet26': [2, 3, 4, 2],
    'resnet34': [3, 4, 6, 3],
}
_CNN = {'smallcnn': SmallCNN, 'mediumcnn': MediumCNN, 'widecnn': WideCNN, 'plaincnn': PlainCNN}


def make_backbone(name, fd=256):
    if name in _RESNET_BLOCKS:
        return HeteroResNet(_RESNET_BLOCKS[name], fd)
    if name in _CNN:
        return _CNN[name](fd)
    raise ValueError(f"unknown backbone: {name}")


# K=5 固定: 每个 client 一个架构
TIER_POOLS = {
    'mild':   ['resnet10', 'resnet14', 'resnet18', 'resnet26', 'resnet34'],
    'strong': ['smallcnn', 'mediumcnn', 'plaincnn', 'widecnn', 'resnet18'],
}
