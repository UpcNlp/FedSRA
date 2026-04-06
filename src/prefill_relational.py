"""
prefill_relational.py
=====================
把日志里已知的 relational 结果写入 json，
这样后续只跑 --pipeline intrinsic 即可，不重复跑 CE。
"""
import os, json

os.makedirs("results", exist_ok=True)

# 从日志中读到的 relational 结果
# 格式: (alpha, k, acc_relational)
known = [
    (0.1,  50,  0.2741),
    (0.1,  100, 0.1802),
    (0.2,  100, 0.1837),
    (0.3,  100, 0.2098),
]

for alpha, k, acc_rel in known:
    path = f"results/grid_a{alpha}_k{k}_s42.json"
    existing = {}
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)

    # 只在没有 relational 结果时写入
    if existing.get("acc_relational") is None:
        existing["alpha"]          = alpha
        existing["n_clients"]      = k
        existing["seed"]           = 42
        existing["acc_relational"] = acc_rel
        existing["acc_intrinsic"]  = existing.get("acc_intrinsic", None)
        existing["gap"]            = None
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"Written: {path}  rel={acc_rel}")
    else:
        print(f"Skip (already has rel={existing['acc_relational']}): {path}")
