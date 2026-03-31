"""
合并所有结果，生成论文图表
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_intro_figure(all_results, save_path='outputs/intro_figure.pdf'):
    alphas = sorted(all_results.keys())
    union_accs = [all_results[a]['union_acc']*100 for a in alphas]
    expert_accs = [all_results[a]['best_expert_acc']*100 for a in alphas]
    fused_accs = [all_results[a]['fused_acc']*100 for a in alphas]
    deltas = [all_results[a]['delta']*100 for a in alphas]
    error_corrs = [all_results[a]['orthogonality']['error_correlation'] for a in alphas]
    oracle_accs = [all_results[a]['orthogonality']['oracle_acc']*100 for a in alphas]

    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
    })

    fig = plt.figure(figsize=(14, 4.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(alphas, fused_accs, 'o-', color='#E53935', linewidth=2.5, markersize=7, label='Fused (Ours)', zorder=5)
    ax1.plot(alphas, union_accs, 's--', color='#1E88E5', linewidth=2, markersize=6, label='Discriminative only', zorder=4)
    ax1.plot(alphas, expert_accs, '^:', color='#43A047', linewidth=2, markersize=6, label='Generative only', zorder=3)
    ax1.plot(alphas, oracle_accs, 'x-.', color='#999999', linewidth=1.5, markersize=5, label='Oracle (D∪G)', zorder=2, alpha=0.7)

    for i, a in enumerate(alphas):
        if deltas[i] > 1.0:
            ax1.annotate(f'+{deltas[i]:.1f}', xy=(a, fused_accs[i]), xytext=(0, 12), textcoords='offset points', fontsize=8, color='#E53935', fontweight='bold', ha='center')

    ax1.set_xscale('log')
    ax1.set_xlabel('Dirichlet α (← more heterogeneous)', fontsize=11)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax1.set_title('(a) Signal comparison', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8.5, loc='lower right', framealpha=0.9)
    ax1.set_xticks(alphas)
    ax1.set_xticklabels([str(a) for a in alphas])
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(min(alphas)*0.7, max(alphas)*1.4)

    ax2 = fig.add_subplot(gs[1])
    bars = ax2.bar(range(len(alphas)), deltas, color=['#E53935' if d > 0 else '#90A4AE' for d in deltas], alpha=0.85, width=0.6)
    for i, (bar, d) in enumerate(zip(bars, deltas)):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3, f'{d:+.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.set_xticks(range(len(alphas)))
    ax2.set_xticklabels([str(a) for a in alphas])
    ax2.set_xlabel('Dirichlet α', fontsize=11)
    ax2.set_ylabel('Δ Accuracy (pp)', fontsize=11)
    ax2.set_title('(b) Fusion gain', fontsize=12, fontweight='bold')
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis='y')

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(alphas, error_corrs, 'D-', color='#7B1FA2', linewidth=2, markersize=7)
    ax3.fill_between(alphas, error_corrs, alpha=0.15, color='#7B1FA2')
    ax3.set_xscale('log')
    ax3.set_xlabel('Dirichlet α', fontsize=11)
    ax3.set_ylabel('Error Correlation (Pearson ρ)', fontsize=11)
    ax3.set_title('(c) Error orthogonality', fontsize=12, fontweight='bold')
    ax3.set_xticks(alphas)
    ax3.set_xticklabels([str(a) for a in alphas])
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(min(alphas)*0.7, max(alphas)*1.4)
    ax3.annotate('← more orthogonal\n(fusion more effective)', xy=(alphas[0], error_corrs[0]), xytext=(40, -25), textcoords='offset points', fontsize=8, color='#7B1FA2', style='italic', arrowprops=dict(arrowstyle='->', color='#7B1FA2', lw=1.2))

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  主图保存: {save_path}")

def plot_per_class_analysis(all_results, save_path='outputs/per_class_analysis.pdf'):
    alphas = sorted(all_results.keys())
    nc = 10
    fig, axes = plt.subplots(1, len(alphas), figsize=(4*len(alphas), 4), sharey=True)
    if len(alphas) == 1: axes = [axes]

    for ai, alpha in enumerate(alphas):
        ax = axes[ai]
        pc = all_results[alpha]['per_class']
        classes = sorted(pc.keys())
        x = np.arange(nc)
        w = 0.25

        u_accs = [pc[c]['union_acc']*100 for c in classes]
        e_accs = [pc[c]['expert_acc']*100 for c in classes]
        f_accs = [pc[c]['fused_acc']*100 for c in classes]

        ax.bar(x - w, u_accs, w, label='Discrim.', color='#1E88E5', alpha=0.8)
        ax.bar(x, e_accs, w, label='Generative', color='#43A047', alpha=0.8)
        ax.bar(x + w, f_accs, w, label='Fused', color='#E53935', alpha=0.8)

        ax.set_xlabel('Class')
        ax.set_title(f'α = {alpha}', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in classes], fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        if ai == 0:
            ax.set_ylabel('Per-class Accuracy (%)')
            ax.legend(fontsize=8)

    plt.suptitle('Per-class signal comparison across heterogeneity levels', fontsize=13, fontweight='bold', y=1.02)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  逐类图保存: {save_path}")

def plot_orthogonality_detail(all_results, save_path='outputs/orthogonality_detail.pdf'):
    alphas = sorted(all_results.keys())
    fig, axes = plt.subplots(1, len(alphas), figsize=(3.5*len(alphas), 3))
    if len(alphas) == 1: axes = [axes]

    for ai, alpha in enumerate(alphas):
        ax = axes[ai]
        orth = all_results[alpha]['orthogonality']
        N = orth['both_correct'] + orth['union_only'] + orth['expert_only'] + orth['both_wrong']
        table = np.array([[orth['both_correct']/N*100, orth['union_only']/N*100],[orth['expert_only']/N*100, orth['both_wrong']/N*100]])
        im = ax.imshow(table, cmap='YlOrRd', vmin=0, vmax=max(table.flatten())*1.1)
        ax.set_xticks([0,1]); ax.set_xticklabels(['E ✓','E ✗'], fontsize=9)
        ax.set_yticks([0,1]); ax.set_yticklabels(['D ✓','D ✗'], fontsize=9)
        for i in range(2):
            for j in range(2):
                ax.text(j,i,f'{table[i,j]:.1f}%',ha='center',va='center',fontsize=11,fontweight='bold',color='white' if table[i,j]>30 else 'black')
        ax.set_title(f'α={alpha}\nρ={orth["error_correlation"]:.3f}',fontsize=10,fontweight='bold')
    plt.suptitle('Error contingency (D=Discriminative, E=Expert/Generative)',fontsize=11,y=1.05)
    plt.savefig(save_path,dpi=200,bbox_inches='tight')
    plt.close()
    print(f"  正交性图保存: {save_path}")

def save_results_text(all_results, save_path='outputs/analysis_results.txt'):
    with open(save_path,'w') as f:
        f.write("="*80+"\nIntro Experiment Results\n"+"="*80+"\n\n")
        alphas=sorted(all_results.keys())
        f.write(f"{'α':>8s}|{'Union':>8s}|{'Expert':>8s}|{'Fused':>8s}|{'Δ':>8s}|{'ErrCorr':>8s}|{'Oracle':>8s}|{'Util':>8s}\n")
        f.write("-"*80+"\n")
        for a in alphas:
            r=all_results[a]
            f.write(f"{a:>8.2f}|{r['union_acc']:>7.2%}|{r['best_expert_acc']:>7.2%}|{r['fused_acc']:>7.2%}|{r['delta']:>+7.2%}|{r['orthogonality']['error_correlation']:>8.4f}|{r['orthogonality']['oracle_acc']:>7.2%}|{r['orthogonality']['fuse_utilization']:>7.2%}\n")
    print(f"  结果文本保存: {save_path}")

def main():
    ALPHAS = [0.05,0.1,0.3,0.5,1.0]
    all_results={}
    for a in ALPHAS:
        with open(f'outputs/result_{a}.json') as f:
            all_results[a]=json.load(f)
    
    os.makedirs('outputs',exist_ok=True)
    plot_intro_figure(all_results)
    plot_per_class_analysis(all_results)
    plot_orthogonality_detail(all_results)
    save_results_text(all_results)
    
    print("\n🎉 全部任务完成！")
    print("📌 论文图：outputs/intro_figure.pdf")
    print("📌 数据文件：outputs/analysis_results.txt")

if __name__=="__main__":
    main()