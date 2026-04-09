CUDA_VISIBLE_DEVICES=0 python3 -c "
from rebuild8_tinyimagenet import *
from rebuild8 import generate_etf

# 1. 检查标签映射
train_dir = './data/tiny-imagenet-200/train'
val_dir = './data/tiny-imagenet-200/val'
from torchvision import datasets
train_ds = datasets.ImageFolder(train_dir)
print(f'训练集 class_to_idx 前5: {dict(list(train_ds.class_to_idx.items())[:5])}')

te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(TINY_MEAN, TINY_STD)])
val_ds = TinyImageNetValDataset(val_dir, transform=te, class_to_idx=train_ds.class_to_idx)
print(f'验证集: {len(val_ds)} samples')
print(f'验证集前10个标签: {val_ds.targets[:10]}')

# 2. 检查标签范围
import numpy as np
targets = np.array(val_ds.targets)
print(f'标签范围: [{targets.min()}, {targets.max()}], 类数: {len(set(targets))}')

# 3. 快速训练1个client + 测试
NC=5; NL=200; FD=256
etf = generate_etf(NL, FD)
cal, ccl, tl, ccc = prepare_data_tinyimagenet(NC, 0.5, NL)

# 训练1个client, 只跑10 epoch
bb = ResNet18Backbone64(FD)
cls = sorted(ccc[0].keys())
bb = train_bb(bb, cal[0], cls, etf, epochs=10)

# 测试 ETF logits
bb.eval(); ed = etf.to(device)
correct = 0; total = 0
with torch.no_grad():
    for x, y in tl:
        f = F.normalize(bb(x.to(device)), dim=1)
        pred = torch.mm(f, ed.T).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += y.size(0)
print(f'\n单 client (10ep, α=0.5) ETF acc: {correct/total:.2%} (随机=0.50%)')
print('OK!' if correct/total > 0.01 else 'BUG! 仍然接近随机')
"
