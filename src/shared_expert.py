"""
shared_expert.py - 共享条件专家网络 (替代 per-class ConditionalExpert)
======================================================================
单个网络代替 K × |classes| 套独立 expert, 通过 AdaLN 在每个 block 注入类条件。

架构: 残差 MLP autoencoder, 编码器/解码器各 n_blocks 个 CondResBlock。
默认 hd=384, ld=64, n_blocks=3 → ~5M params。

输入约定: feat 应已 L2-normalized (匹配 ResNet18Backbone.forward 输出)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CondAdaLN(nn.Module):
    """Class-conditioned adaptive LayerNorm: c → (γ, β) modulating LN output.

    γ 与 β 的 modulator 用 zero-init, 启动时等价于 identity LN, 训练稳定。
    """
    def __init__(self, dim, ed):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.mod = nn.Linear(ed, 2 * dim)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x, c):
        g, b = self.mod(c).chunk(2, dim=-1)
        return self.ln(x) * (1 + g) + b


class CondResBlock(nn.Module):
    """Residual MLP block, AdaLN-modulated, GELU, expansion ratio `mult`."""
    def __init__(self, dim, ed, mult=2, dropout=0.0):
        super().__init__()
        self.adaln = CondAdaLN(dim, ed)
        self.fc1 = nn.Linear(dim, dim * mult)
        self.fc2 = nn.Linear(dim * mult, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, c):
        h = self.adaln(x, c)
        return x + self.fc2(self.drop(self.act(self.fc1(h))))


class SharedConditionalExpert(nn.Module):
    """Single network shared across all classes of a client.

    Args:
        fd: feature dim
        ed: class-embedding dim (same as ETF dim)
        hd: hidden width
        ld: latent bottleneck dim
        n_blocks: residual blocks per side (encoder = decoder)
    """
    def __init__(self, fd=256, ed=256, hd=384, ld=64, n_blocks=3, dropout=0.0):
        super().__init__()
        self.fd, self.ed = fd, ed
        self.hd, self.ld = hd, ld
        self.n_blocks = n_blocks

        self.enc_in = nn.Linear(fd, hd)
        self.enc_blocks = nn.ModuleList(
            [CondResBlock(hd, ed, mult=2, dropout=dropout) for _ in range(n_blocks)]
        )
        self.enc_norm = nn.LayerNorm(hd)
        self.enc_out = nn.Linear(hd, ld)

        self.dec_in = nn.Linear(ld, hd)
        self.dec_blocks = nn.ModuleList(
            [CondResBlock(hd, ed, mult=2, dropout=dropout) for _ in range(n_blocks)]
        )
        self.dec_norm = nn.LayerNorm(hd)
        self.dec_out = nn.Linear(hd, fd)

    def encode(self, f, c):
        h = self.enc_in(f)
        for blk in self.enc_blocks:
            h = blk(h, c)
        return self.enc_out(self.enc_norm(h))

    def decode(self, z, c):
        h = self.dec_in(z)
        for blk in self.dec_blocks:
            h = blk(h, c)
        return self.dec_out(self.dec_norm(h))

    def forward(self, f, c):
        z = self.encode(f, c)
        return self.decode(z, c), z


def shared_expert_loss(feat, y, expert, ed, n_classes, fdim, margin=0.05):
    """Batch-wise reconstruction + repulsion loss for shared conditional expert.

    feat:   (B, fd)  already-normalized features
    y:      (B,)     class labels
    ed:     (NL, ed) ETF class embeddings
    Returns total loss and dict of components.
    """
    B = feat.size(0)
    dev = feat.device

    # l1: positive reconstruction
    rec_pos, _ = expert(feat, ed[y])
    l1 = F.mse_loss(rec_pos, feat)

    # sample wrong classes (uniform over y' ≠ y)
    off1 = torch.randint(1, n_classes, (B,), device=dev)
    y_neg = (y + off1) % n_classes
    off2 = torch.randint(1, n_classes, (B,), device=dev)
    y_neg2 = (y + off2) % n_classes

    # l2: real feat with wrong condition → repel
    rec_neg, _ = expert(feat, ed[y_neg])
    l2 = F.relu(margin - ((feat - rec_neg) ** 2).mean(1)).mean()

    # synthetic features around other-class ETF anchors
    scale = 0.05 + 0.25 * torch.rand(B, 1, device=dev)
    synth = F.normalize(ed[y_neg2] + torch.randn(B, fdim, device=dev) * scale, dim=1)

    # l3: synth feat with its own neg condition → repel
    rec_s1, _ = expert(synth, ed[y_neg2])
    l3 = F.relu(margin - ((synth - rec_s1) ** 2).mean(1)).mean()

    # l4: synth feat with positive-y condition → repel
    rec_s2, _ = expert(synth, ed[y])
    l4 = F.relu(margin - ((synth - rec_s2) ** 2).mean(1)).mean()

    return l1 + l2 + l3 + l4, dict(l1=l1.item(), l2=l2.item(), l3=l3.item(), l4=l4.item())


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


if __name__ == '__main__':
    for cfg in [(256, 64, 3), (384, 64, 3), (512, 128, 4)]:
        hd, ld, nb = cfg
        m = SharedConditionalExpert(256, 256, hd, ld, nb)
        print(f"hd={hd} ld={ld} n_blocks={nb}: {count_params(m)/1e6:.2f}M params")
