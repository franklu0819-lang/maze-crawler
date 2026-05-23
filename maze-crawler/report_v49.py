"""Generate v49 training report with charts as PDF."""
import re
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'Heiti TC', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_v17_v49.log')
PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report_v49.pdf')

# === Parse log accurately ===
# 1. Get all regular log lines with step, best, eval_R, entropy
regular_lines = []  # (step, best, eval_R_or_None, entropy)
eval_r_data = []    # (step, eval_R)
entropy_data = []   # (step, entropy)

with open(LOG) as f:
    lines = f.readlines()

for line in lines:
    m = re.match(r'\s*\[(\d+)/2000\].*best=\s*([\d.]+)%.*e=\s*([\d.]+)', line)
    if m:
        step = int(m.group(1))
        best = float(m.group(2))
        ent = float(m.group(3))
        mr = re.search(r'eval_R=\s*([\d.]+)%', line)
        er = float(mr.group(1)) if mr else None
        regular_lines.append((step, best, er, ent))
        entropy_data.append((step, ent))
        if er is not None:
            eval_r_data.append((step, er))

# 2. Parse New best lines - associate with the NEXT regular line that shows the new best value
# New best appears between log lines, so we find the first line after it with matching best value
best_breakthroughs = []  # (step, best_wr)

for i, line in enumerate(lines):
    if 'New best eval WR' in line:
        mb = re.search(r'New best eval WR (\d+)%', line)
        if mb:
            wr = int(mb.group(1))
            # Look FORWARD for the first regular line showing this new best value
            for j in range(i + 1, min(len(lines), i + 10)):
                m2 = re.match(r'\s*\[(\d+)/2000\].*best=\s*' + str(wr) + r'\.0%', lines[j])
                if m2:
                    best_breakthroughs.append((int(m2.group(1)), wr))
                    break

print(f"Breakthroughs: {best_breakthroughs}")

# 3. Build step-wise best curve (forward fill from regular lines)
# Use the actual best values from regular log lines
best_steps = [s for s, b, er, ent in regular_lines]
best_vals = [b for s, b, er, ent in regular_lines]

# Stats
final_best = best_vals[-1] if best_vals else 0
total_breakthroughs = len(best_breakthroughs)
max_eval_r = max(v for _, v in eval_r_data) if eval_r_data else 0
avg_eval_r = sum(v for _, v in eval_r_data) / len(eval_r_data) if eval_r_data else 0
final_eval_r = eval_r_data[-1][1] if eval_r_data else 0
max_entropy = max(v for _, v in entropy_data) if entropy_data else 0
min_entropy = min(v for _, v in entropy_data) if entropy_data else 0
final_entropy = entropy_data[-1][1] if entropy_data else 0
first_entropy = entropy_data[0][1] if entropy_data else 0

best_seq = [wr for _, wr in best_breakthroughs]
best_str = ' -> '.join(f'{b}%' for b in best_seq)

# === Generate PDF ===
with PdfPages(PDF) as pdf:
    # Page 1: Summary
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')

    summary = (
        f"v49 训练总结报告\n"
        f"训练配置：train_v17.py, 2000轮\n\n"
        f"══════════════════════════════════════\n"
        f"核心指标\n"
        f"══════════════════════════════════════\n"
        f"  最终Best WR:     {final_best:.0f}%\n"
        f"  Best WR突破次数: {total_breakthroughs}次\n"
        f"  Best WR历史:     {best_str}\n\n"
        f"  最终eval_R:      {final_eval_r:.0f}%\n"
        f"  峰值eval_R:      {max_eval_r:.0f}%\n"
        f"  平均eval_R:      {avg_eval_r:.1f}%\n\n"
        f"  最终熵e:         {final_entropy:.3f}\n"
        f"  熵变化范围:      {min_entropy:.3f} ~ {max_entropy:.3f}\n\n"
        f"══════════════════════════════════════\n"
        f"训练阶段分析\n"
        f"══════════════════════════════════════\n"
        f"  早期(0~500):    快速学习, best {best_seq[0]}% -> {best_seq[min(4,len(best_seq)-1)]}%\n"
        f"  中期(500~1000): 稳步提升, best 达到 {best_seq[min(8,len(best_seq)-1)]}%\n"
        f"  后期(1000~1500):加速突破, best 达到 {best_seq[min(13,len(best_seq)-1)]}%\n"
        f"  收尾(1500~2000):收敛, 最终best {final_best:.0f}%\n\n"
        f"══════════════════════════════════════\n"
        f"对战记录\n"
        f"══════════════════════════════════════\n"
        f"  vs random(500局):  419W-67L-14D (83.8%)\n"
        f"  vs v50 best(500局): 66W-114L-320D (13.2%)\n"
    )

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='sans-serif',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    pdf.savefig(fig)
    plt.close()

    # Page 2: Best WR + eval_R curves
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), gridspec_kw={'hspace': 0.35})

    # Best WR curve from actual log data
    ax1.step(best_steps, best_vals, where='post', color='#2196F3', linewidth=1.5, label='Best WR')
    ax1.fill_between(best_steps, best_vals, step='post', alpha=0.15, color='#2196F3')
    ax1.set_xlabel('训练轮次')
    ax1.set_ylabel('胜率 (%)')
    ax1.set_title('v49 Best WR 变化曲线', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 2050)
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Mark breakthroughs
    for step, wr in best_breakthroughs:
        ax1.plot(step, wr, 'r^', markersize=8, zorder=5)
        ax1.annotate(f'{wr}%', xy=(step, wr), xytext=(step + 30, wr + 2),
                     fontsize=8, color='#C62828', fontweight='bold')

    # eval_R curve
    er_s = [s for s, _ in eval_r_data]
    er_v = [v for _, v in eval_r_data]
    ax2.plot(er_s, er_v, color='#FF5722', linewidth=1, alpha=0.7, label='eval_R (vs random)')
    ax2.plot(er_s, er_v, 'o', color='#FF5722', markersize=2)
    if len(er_v) > 5:
        window = 5
        ma = [sum(er_v[max(0, i - window):i + 1]) / min(i + 1, window) for i in range(len(er_v))]
        ax2.plot(er_s, ma, color='#D32F2F', linewidth=2, label=f'移动平均(MA-{window})')
    ax2.set_xlabel('训练轮次')
    ax2.set_ylabel('评估胜率 (%)')
    ax2.set_title('v49 eval_R 变化曲线', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 2050)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    pdf.savefig(fig)
    plt.close()

    # Page 3: Entropy curve
    fig, ax = plt.subplots(figsize=(11, 8.5))
    en_s = [s for s, _ in entropy_data]
    en_v = [v for _, v in entropy_data]
    ax.plot(en_s, en_v, color='#4CAF50', linewidth=0.8, alpha=0.7)
    if len(en_v) > 10:
        window = 10
        ma = [sum(en_v[max(0, i - window):i + 1]) / min(i + 1, window) for i in range(len(en_v))]
        ax.plot(en_s, ma, color='#2E7D32', linewidth=2, label=f'移动平均(MA-{window})')
    ax.set_xlabel('训练轮次')
    ax.set_ylabel('熵')
    ax.set_title('v49 熵 (策略探索度) 变化曲线', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 2050)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax.annotate(f'起始: {first_entropy:.3f}', xy=(entropy_data[0][0], first_entropy),
                fontsize=9, color='#1B5E20')
    ax.annotate(f'结束: {final_entropy:.3f}', xy=(entropy_data[-1][0], final_entropy),
                fontsize=9, color='#1B5E20')

    pdf.savefig(fig)
    plt.close()

    # Page 4: Combined overview
    fig, ax1 = plt.subplots(figsize=(11, 8.5))
    ax2 = ax1.twinx()

    ax1.step(best_steps, best_vals, where='post', color='#2196F3', linewidth=2, label='Best WR (%)')
    ax1.set_ylabel('胜率 (%)', color='#2196F3')
    ax1.tick_params(axis='y', labelcolor='#2196F3')

    ax2.plot(en_s, en_v, color='#4CAF50', linewidth=0.8, alpha=0.5)
    if len(en_v) > 10:
        ma = [sum(en_v[max(0, i - window):i + 1]) / min(i + 1, window) for i in range(len(en_v))]
        ax2.plot(en_s, ma, color='#4CAF50', linewidth=2, label='熵 (MA-10)')
    ax2.set_ylabel('熵', color='#4CAF50')
    ax2.tick_params(axis='y', labelcolor='#4CAF50')

    ax1.set_xlabel('训练轮次')
    ax1.set_title('v49 训练总览: Best WR vs 熵', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 2050)
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    pdf.savefig(fig)
    plt.close()

print(f"PDF generated: {PDF}")
