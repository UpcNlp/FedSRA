import json, glob
print(f"{'data':9s}{'a':>5s}{'K':>3s}{'B1':>7s}{'B8':>7s}{'B32':>7s}{'B128':>7s}{'full':>7s}{'frozen':>8s}")
for f in sorted(glob.glob("results/batchz_*.json")):
    d=json.load(open(f)); b=d["batch_acc"]
    g=lambda k: b.get(k,-0.01)*100
    print(f"{d['dataset']:9s}{d['alpha']:>5}{d['K']:>3}{g('1'):7.1f}{g('8'):7.1f}{g('32'):7.1f}{g('128'):7.1f}{d['global_acc']*100:7.1f}{d['frozen_acc']*100:8.1f}")
