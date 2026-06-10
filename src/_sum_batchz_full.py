import json, glob
hdr=f"{'dataset':9s}{'alpha':>6s}{'K':>4s}{'B=1':>7s}{'B=8':>7s}{'B=32':>7s}{'B=128':>7s}{'B=512':>7s}{'full':>7s}{'frozen':>8s}{'eval_n':>8s}"
print(hdr); print('-'*len(hdr))
def keyf(f):
    d=json.load(open(f)); return (d['dataset'], d['K'], d['alpha'])
for f in sorted(glob.glob("results/batchz_*.json"), key=keyf):
    d=json.load(open(f)); b=d['batch_acc']
    g=lambda k: b[k]*100 if k in b else float('nan')
    print(f"{d['dataset']:9s}{d['alpha']:>6}{d['K']:>4}{g('1'):7.1f}{g('8'):7.1f}{g('32'):7.1f}{g('128'):7.1f}{g('512'):7.1f}{d['global_acc']*100:7.1f}{d['frozen_acc']*100:8.1f}{d['eval_n']:>8d}")
