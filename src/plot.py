"""
plot_intro_figure_beautified.py
===============================
Beautified intro figure for ICDE paper.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Experimental data ─────────────────────────────────────────────
RESULTS = {
    0.05: {5:(42.3,37.6), 10:(36.2,42.3), 20:(20.7,48.0), 50:(10.8,39.5)},
    0.1:  {5:(53.8,44.0), 10:(42.7,44.1), 20:(36.5,45.5)},
    0.2:  {5:(66.1,55.1), 10:(48.7,48.9), 20:(42.2,48.8), 50:(26.7,51.5)},
    0.3:  {5:(67.1,59.6), 10:(53.1,49.1), 20:(43.1,45.5), 50:(36.2,46.4)},
    0.5:  {5:(75.4,39.4), 10:(61.0,48.4), 20:(52.9,41.5), 50:(34.5,39.4)},
    1.0:  {5:(78.5,42.1), 10:(69.4,47.9), 20:(58.2,42.6), 50:(39.6,40.7)},
}

Ks_ALL = [5, 10, 20, 50]

# ── Academic Style Config ─────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.edgecolor': '#333333',
    'text.color': '#111111',
    'axes.labelcolor': '#111111',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

C_REL   = '#2874A6'
C_INT   = '#239B56'
C_CROSS = '#C0392B'

OUT_DIR = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────
def find_crossover(x, y_rel, y_int):
    diff = np.array(y_rel) - np.array(y_int)
    for i in range(len(diff) - 1):
        if np.isnan(diff[i]) or np.isnan(diff[i+1]):
            continue
        if diff[i] * diff[i+1] < 0:
            frac = diff[i] / (diff[i] - diff[i+1])
            xc = x[i] + frac * (x[i+1] - x[i])
            yc = y_rel[i] + frac * (y_rel[i+1] - y_rel[i])
            return xc, yc
    return None

def setup_academic_axes(ax):
    ax.yaxis.grid(True, linestyle='--', color='#E5E8E8', alpha=0.8, zorder=0)
    ax.set_axisbelow(True)

# ── Figure ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(8.0, 3.2))
gs = fig.add_gridspec(
    1, 2,
    left=0.08, right=0.98,
    top=0.88,  bottom=0.18,
    wspace=0.28,
)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])

setup_academic_axes(ax_a)
setup_academic_axes(ax_b)

x_pos = np.arange(len(Ks_ALL))
x_labels = [str(k) for k in Ks_ALL]

# ═════════════════════════════════════════════════════════════════
# Panel (a):  α = 0.3, Relational vs Intrinsic over K
# ═════════════════════════════════════════════════════════════════
ALPHA_SHOW = 0.3
d = RESULTS[ALPHA_SHOW]
ks_a  = sorted([k for k in d.keys() if k in Ks_ALL])
idx_a = [Ks_ALL.index(k) for k in ks_a]
rel_a = [d[k][0] for k in ks_a]
int_a = [d[k][1] for k in ks_a]

ax_a.plot(idx_a, rel_a,
          color=C_REL, lw=2.0, marker='o', markersize=6.5,
          markerfacecolor='white', markeredgecolor=C_REL,
          markeredgewidth=1.8, label='Relational (CE)', zorder=4)
ax_a.plot(idx_a, int_a,
          color=C_INT, lw=2.0, marker='s', markersize=6.0,
          markerfacecolor='white', markeredgecolor=C_INT,
          markeredgewidth=1.8, label='Intrinsic (SSL)', zorder=4)

ax_a.fill_between(idx_a, rel_a, int_a,
                  where=np.array(rel_a) >= np.array(int_a),
                  alpha=0.08, color=C_REL, interpolate=True, zorder=1)
ax_a.fill_between(idx_a, rel_a, int_a,
                  where=np.array(int_a) >= np.array(rel_a),
                  alpha=0.08, color=C_INT, interpolate=True, zorder=1)

cr = find_crossover(np.array(idx_a, dtype=float), rel_a, int_a)
if cr:
    xc, yc = cr
    ax_a.axvline(xc, color=C_CROSS, lw=1.2, ls=(0, (5, 5)), alpha=0.7, zorder=2)
    ax_a.scatter([xc], [yc], color=C_CROSS, s=100, marker='*',
                 linewidths=1.0, edgecolors='white', zorder=5)

    k_star = 10 ** np.interp(xc, np.arange(len(Ks_ALL)), np.log10(Ks_ALL))
    ax_a.text(xc + 0.08, yc + 3.0,
              rf'$K^*\!\approx\!{k_star:.0f}$',
              color=C_CROSS, fontsize=10, fontweight='bold',
              ha='left', va='bottom', zorder=6)

    ax_a.text(0.45, 58,
              'Relational\ndominates', ha='center', va='center',
              color=C_REL, fontsize=9, fontstyle='italic', alpha=0.9)
    ax_a.text(2.65, 39.0,
              'Intrinsic\ndominates', ha='center', va='center',
              color=C_INT, fontsize=9, fontstyle='italic', alpha=0.9)

ax_a.set_xticks(x_pos)
ax_a.set_xticklabels(x_labels, fontsize=9)
ax_a.set_xlabel(r'Number of clients $K$ ($\rightarrow$ more fragmentation)', fontsize=10)
ax_a.set_ylabel('Test accuracy (%)', fontsize=10)
ax_a.set_title(r'(a) Relational vs. Intrinsic ($\alpha\!=\!0.3$)',
               fontsize=11, fontweight='bold', pad=10)

ax_a.legend(fontsize=9, loc='upper right',
            frameon=True, framealpha=0.95, edgecolor='#E5E8E8',
            handlelength=2.0, handletextpad=0.6, borderpad=0.5)

# ═════════════════════════════════════════════════════════════════
# Panel (b):  Gap (Rel − Int) across all α, over K
# ═════════════════════════════════════════════════════════════════
ALPHA_PALETTE = {
    0.05: ('#8E44AD', 'v'),
    0.1:  ('#2980B9', 'o'),
    0.2:  ('#16A085', 's'),
    0.3:  ('#C0392B', 'D'),
    0.5:  ('#D35400', '^'),
    1.0:  ('#7F8C8D', 'p'),
}

for alpha in sorted(RESULTS.keys()):
    d = RESULTS[alpha]
    ks_b  = sorted([k for k in d.keys() if k in Ks_ALL])
    idx_b = [Ks_ALL.index(k) for k in ks_b]
    gap_b = [d[k][0] - d[k][1] for k in ks_b]
    color, marker = ALPHA_PALETTE[alpha]
    ax_b.plot(idx_b, gap_b,
              color=color, lw=1.5, marker=marker, markersize=5.0,
              markerfacecolor='white', markeredgecolor=color,
              markeredgewidth=1.2,
              label=rf'$\alpha$={alpha}', zorder=4)

ax_b.axhline(0, color='#2C3E50', lw=1.2, ls='-', zorder=2)
ax_b.fill_between(x_pos, 0, 45, alpha=0.04, color=C_REL, zorder=1)
ax_b.fill_between(x_pos, -40, 0, alpha=0.04, color=C_INT, zorder=1)

bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
ax_b.text(0.02, 0.94, 'Rel. wins ↑', transform=ax_b.transAxes,
          color=C_REL, ha='left', va='top', fontsize=9, fontstyle='italic',
          bbox=bbox_props, zorder=5)
ax_b.text(0.02, 0.06, 'Int. wins ↓', transform=ax_b.transAxes,
          color=C_INT, ha='left', va='bottom', fontsize=9, fontstyle='italic',
          bbox=bbox_props, zorder=5)

ax_b.set_xticks(x_pos)
ax_b.set_xticklabels(x_labels, fontsize=9)
ax_b.set_xlabel(r'Number of clients $K$ ($\rightarrow$ more fragmentation)', fontsize=10)
ax_b.set_ylabel('Accuracy gap (Rel. − Int.) in pp', fontsize=10)
ax_b.set_title('(b) Gap across heterogeneity levels',
               fontsize=11, fontweight='bold', pad=10)
ax_b.set_ylim(-40, 45)

ax_b.legend(fontsize=8.5, loc='upper right', ncol=2,
            frameon=True, framealpha=0.95, edgecolor='#E5E8E8',
            columnspacing=0.8, handlelength=1.5, handletextpad=0.4,
            )

# ── Save ──────────────────────────────────────────────────────────
for ext in ['pdf', 'png']:
    path = os.path.join(OUT_DIR, f'intro_figure.{ext}')
    fig.savefig(path, bbox_inches='tight', facecolor='white', pad_inches=0.05)
    print(f'Saved: {path}')

plt.close()
print('Done.')