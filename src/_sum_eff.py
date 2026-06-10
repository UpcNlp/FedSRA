import json
d=json.load(open("results/efficiency_measurements.json"))
order=['ofedavg','dense','coboosting','ensemble','fafi','ours','ours-stream']
def name(r): return r['method']+('-stream' if r.get('stream') else '')
for bs in (256,1):
    rows=[r for r in d if r.get('bs')==bs and r.get('ms_per_img',-1)>=0]
    if not rows: continue
    print(f"\n=== bs={bs} ===")
    print(f"{'method':14s}{'K':>4s}{'ms/img':>9s}{'img/s':>9s}{'mem MB':>9s}{'up MB':>8s}")
    for K in (5,10,20,50):
        for r in sorted([x for x in rows if x['K']==K], key=lambda x: order.index(name(x)) if name(x) in order else 99):
            print(f"{name(r):14s}{K:>4d}{r['ms_per_img']:>9.3f}{r['throughput_img_s']:>9.0f}{r['gpu_peak_mb']:>9.1f}{r['upload_mb']:>8.1f}")
