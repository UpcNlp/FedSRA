import re, os, json
import sys

logs = []
for d in ["logs/overnight_89984", "logs/overnight_50527"]:
    for f in sorted(os.listdir(d)):
        if f.startswith("sweep_c100_") and f.endswith(".log"):
            logs.append(os.path.join(d, f))

rows = {}
for log in logs:
    m = re.match(r".*sweep_c100_a([\d.]+)_k(\d+)\.log", log)
    if not m: continue
    a, K = m.group(1), int(m.group(2))
    txt = open(log).read()
    blocks = re.findall(r"--- α_f=([\d.]+) ---\s*\n(.*?)(?=--- α_f|\Z)", txt, re.DOTALL)
    r_acc = i_acc = j_acc = None
    sweeps = []
    for af, blk in blocks:
        af = float(af)
        if i_acc is None:
            mm = re.search(r"row I\s+:\s*([\d.]+)%", blk)
            i_acc = float(mm.group(1)) if mm else None
        if j_acc is None:
            mm = re.search(r"row J\s+:\s*([\d.]+)%", blk)
            j_acc = float(mm.group(1)) if mm else None
        if r_acc is None:
            mm = re.search(r"row R\s+:\s*([\d.]+)%", blk)
            r_acc = float(mm.group(1)) if mm else None
        mm = re.search(r"row I\+Expert:\s*([\d.]+)%", blk)
        ie = float(mm.group(1)) if mm else None
        mm = re.search(r"row J\+Expert:\s*([\d.]+)%", blk)
        je = float(mm.group(1)) if mm else None
        sweeps.append((af, ie, je))
    ie_best = max(sweeps, key=lambda x: x[1] if x[1] is not None else -1)
    je_best = max(sweeps, key=lambda x: x[2] if x[2] is not None else -1)
    rows[(a, K)] = dict(R=r_acc, I=i_acc, J=j_acc,
                        IE_best=ie_best[1], IE_af=ie_best[0],
                        JE_best=je_best[2], JE_af=je_best[0])

print(f'{"α":>5} {"K":>3} | {"R":>6} {"I":>6} {"J":>6} | {"I+E*":>6} {"α_f":>5} | {"ΔI→IE":>7} | {"J+E*":>6} {"α_f":>5}')
print("-" * 90)
for a in ["0.05", "0.1", "0.3", "0.5"]:
    for K in [5, 10, 20, 50]:
        r = rows.get((a, K))
        if r is None:
            print(f'{a:>5} {K:>3} | MISSING')
            continue
        def ds(v): return f'{v:>6.2f}' if v is not None else f'{"---":>6}'
        delta = (r["IE_best"] - r["I"]) if r["IE_best"] is not None and r["I"] is not None else None
        ds_delta = f'{delta:>+7.2f}' if delta is not None else f'{"---":>7}'
        print(f'{a:>5} {K:>3} | {ds(r["R"])} {ds(r["I"])} {ds(r["J"])} | {ds(r["IE_best"])} {r["IE_af"]:>5.1f} | {ds_delta} | {ds(r["JE_best"])} {r["JE_af"]:>5.1f}')

json.dump({f"{a}_{K}": v for (a, K), v in rows.items()}, open("logs/overnight_summary.json", "w"), indent=2)
print("\nSaved to logs/overnight_summary.json")
