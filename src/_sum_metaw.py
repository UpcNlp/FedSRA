import json, glob
def keyf(f):
    d=json.load(open(f)); return (d['dataset'], d['K'], d['alpha'])
hdr=f"{'dataset':9s}{'a':>5s}{'K':>4s}{'unif':>7s}{'count':>7s}{'sqrtN*':>8s}{'cover':>7s}{'noisy10':>8s}{'noisy20':>8s}{'oracle':>8s}"
print(hdr); print('-'*len(hdr))
for f in sorted(glob.glob("results/metaw_*.json"), key=keyf):
    d=json.load(open(f)); a=d['acc']; n=d['noisy']
    print(f"{d['dataset']:9s}{d['alpha']:>5}{d['K']:>4}{a['uniform']*100:7.1f}{a['count']*100:7.1f}{a['sqrt_count']*100:8.1f}{a['coverage']*100:7.1f}{n['coverage_noisy10']['mean']*100:8.1f}{n['coverage_noisy20']['mean']*100:8.1f}{d['oracle']*100:8.1f}")
print("\n* sqrtN = deployed method;  cover = paper's stated prior")
