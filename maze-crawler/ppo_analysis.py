#!/usr/bin/env python3
"""
PPO Training Log Analysis - v17_v50
生成可视化图表和PDF报告
"""
import re
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.ticker import MultipleLocator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_log(log_path):
    """解析训练日志"""
    data = {
        'iterations': [],
        'sp': [],  # Self-play win rate
        'best': [],  # Best mixed eval WR
        'p_loss': [],  # Policy loss
        'v_loss': [],  # Value loss
        'entropy': [],
        'eval_m': [],  # Mixed eval result
        'time_s': []  # Cumulative time in seconds
    }
    
    best_breakpoints = []
    current_best = 0
    
    with open(log_path, 'r') as f:
        for line in f:
            # 检测New best
            if 'New best mixed eval WR' in line:
                match = re.search(r'New best mixed eval WR (\d+)%', line)
                if match:
                    wr = int(match.group(1))
                    # 找到对应的迭代轮次
                    pass
            
            # 匹配迭代行
            match = re.match(r'\[(\s*\d+)/2000\] SP=\s*(\d+\.\d+)% best=\s*(\d+\.\d+)% p=([-\d.]+) v=([\d.]+) e=([\d.]+) n=(\d+)', line)
            if match:
                iters = int(match.group(1))
                sp = float(match.group(2))
                best = float(match.group(3))
                p_loss = float(match.group(4))
                v_loss = float(match.group(5))
                entropy = float(match.group(6))
                
                data['iterations'].append(iters)
                data['sp'].append(sp)
                data['best'].append(best)
                data['p_loss'].append(p_loss)
                data['v_loss'].append(v_loss)
                data['entropy'].append(entropy)
                
                # 记录突破点
                if best > current_best:
                    best_breakpoints.append((iters, best))
                    current_best = best
                
                # 提取时间或eval
                if 'eval_M' in line:
                    eval_match = re.search(r'eval_M=\s*(\d+\.\d+)%', line)
                    if eval_match:
                        data['eval_m'].append((iters, float(eval_match.group(1))))
                    
                    time_match = re.search(r't=(\d+)s', line)
                    if time_match:
                        data['time_s'].append(int(time_match.group(1)))
                else:
                    time_match = re.search(r't=(\d+)s', line)
                    if time_match:
                        data['time_s'].append(int(time_match.group(1)))
    
    return data, best_breakpoints

def plot_performance_curve(data, best_breakpoints, output_path):
    """绘制性能曲线"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Best WR曲线
    ax.plot(data['iterations'], data['best'], 'b-', linewidth=2.5, label='Best Mixed Eval WR', alpha=0.8)
    
    # Eval点
    if data['eval_m']:
        eval_iters = [x[0] for x in data['eval_m']]
        eval_vals = [x[1] for x in data['eval_m']]
        ax.scatter(eval_iters, eval_vals, c='red', s=40, alpha=0.6, label='每10轮评估')
    
    # 标注突破点
    for iters, best in best_breakpoints:
        ax.annotate(f'{best:.0f}%', xy=(iters, best), xytext=(iters+20, best+2),
                   arrowprops=dict(arrowstyle='->', color='darkblue', lw=1.5),
                   fontsize=10, fontweight='bold', color='darkblue')
    
    ax.set_xlabel('训练轮次', fontsize=12, fontweight='bold')
    ax.set_ylabel('胜率 (%)', fontsize=12, fontweight='bold')
    ax.set_title('PPO训练 - Best Mixed Eval WR 性能曲线', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 80)
    
    # 添加阶段标记
    stages = [(500, 75, '前500轮\n59%'), (1000, 75, '1000轮\n66%'), (1500, 75, '1500轮\n69%'), (2000, 75, '2000轮\n69%')]
    for x, y, text in stages:
        ax.axvline(x=x, color='gray', linestyle='--', alpha=0.3)
        ax.text(x, y, text, ha='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 性能曲线已保存: {output_path}")

def plot_sp_trend(data, output_path):
    """绘制SP胜率趋势"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 原始SP数据
    ax.plot(data['iterations'], data['sp'], 'g-', linewidth=1.5, alpha=0.4, label='SP原始数据')
    
    # 移动平均
    window_size = 20
    sp_array = np.array(data['sp'])
    sp_ma = np.convolve(sp_array, np.ones(window_size)/window_size, mode='valid')
    ma_x = data['iterations'][window_size-1:]
    ax.plot(ma_x, sp_ma, 'darkgreen', linewidth=3, label=f'{window_size}轮移动平均')
    
    # 添加收敛区域标记
    ax.axhline(y=35, color='orange', linestyle='--', alpha=0.5)
    ax.text(2050, 35, '收敛区间\n25%-40%', ha='left', va='center', fontsize=10, color='orange')
    
    # 阶段分析
    ax.axvspan(0, 100, alpha=0.1, color='red', label='快速迭代期 (SP剧烈下降)')
    ax.axvspan(100, 1000, alpha=0.1, color='yellow', label='持续优化期')
    ax.axvspan(1000, 2000, alpha=0.1, color='green', label='收敛稳定期')
    
    ax.set_xlabel('训练轮次', fontsize=12, fontweight='bold')
    ax.set_ylabel('Self-Play 胜率 (%)', fontsize=12, fontweight='bold')
    ax.set_title('SP Self-Play 胜率趋势', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ SP趋势图已保存: {output_path}")

def plot_entropy_curve(data, output_path):
    """绘制Entropy衰减曲线"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.plot(data['iterations'], data['entropy'], 'purple', linewidth=2.5, alpha=0.8)
    
    # 标注关键值
    ax.annotate(f'初始: {data["entropy"][0]:.3f}', xy=(0, data['entropy'][0]), 
               xytext=(100, 0.68), arrowprops=dict(arrowstyle='->', color='purple'),
               fontsize=10, color='purple', fontweight='bold')
    
    ax.annotate(f'最终: {data["entropy"][-1]:.3f}', xy=(2000, data['entropy'][-1]), 
               xytext=(1700, 0.25), arrowprops=dict(arrowstyle='->', color='purple'),
               fontsize=10, color='purple', fontweight='bold')
    
    # 衰减率
    decay_rate = (1 - data['entropy'][-1] / data['entropy'][0]) * 100
    ax.text(1000, 0.5, f'总衰减率: {decay_rate:.1f}%', fontsize=12, 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           ha='center', fontweight='bold')
    
    ax.set_xlabel('训练轮次', fontsize=12, fontweight='bold')
    ax.set_ylabel('Entropy', fontsize=12, fontweight='bold')
    ax.set_title('策略Entropy衰减曲线', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Entropy曲线图已保存: {output_path}")

def plot_loss_curves(data, output_path):
    """绘制Loss变化曲线"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Policy Loss
    ax1.plot(data['iterations'], data['p_loss'], 'red', linewidth=1.5, alpha=0.6)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.set_ylabel('Policy Loss', fontsize=11, fontweight='bold')
    ax1.set_title('Policy Loss 变化趋势', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Value Loss
    ax2.plot(data['iterations'], data['v_loss'], 'blue', linewidth=2, alpha=0.8)
    ax2.annotate(f'初始: {data["v_loss"][0]:.2f}', xy=(0, data['v_loss'][0]), 
                xytext=(100, 11), arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, color='blue')
    ax2.annotate(f'最终: {data["v_loss"][-1]:.2f}', xy=(2000, data['v_loss'][-1]), 
                xytext=(1700, 3), arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, color='blue')
    ax2.set_xlabel('训练轮次', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Value Loss', fontsize=11, fontweight='bold')
    ax2.set_title('Value Loss 变化趋势', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Loss曲线图已保存: {output_path}")

def plot_training_efficiency(data, best_breakpoints, output_path):
    """绘制训练效率分析"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 阶段提升分析
    stages = [0, 500, 1000, 1500, 2000]
    stage_best = [0, 59, 66, 69, 69]  # 根据数据手动整理
    stage_gain = [59, 7, 3, 0]
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    
    bars = ax.bar(range(4), stage_gain, color=colors, alpha=0.8, width=0.6)
    
    ax.set_xticks(range(4))
    ax.set_xticklabels(['0-500轮', '500-1000轮', '1000-1500轮', '1500-2000轮'], fontsize=11)
    ax.set_ylabel('Best WR 提升幅度 (百分点)', fontsize=12, fontweight='bold')
    ax.set_title('各阶段训练效率对比 - 边际收益递减分析', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (bar, gain) in enumerate(zip(bars, stage_gain)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'+{gain}pp\n({stage_best[i+1]}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 累计百分比
        cumulative = sum(stage_gain[:i+1])
        total = sum(stage_gain)
        percentage = (gain / total * 100) if total > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{percentage:.0f}%',
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # 添加说明
    ax.text(1.5, 55, '前500轮贡献了85%的总提升\n后500轮提升为0', 
           ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 训练效率图已保存: {output_path}")

def main():
    log_path = '/Users/leo/Projects/kaggle/maze-crawler/train_v17_v50.log'
    output_dir = '/Users/leo/Projects/kaggle/maze-crawler/analysis_output'
    os.makedirs(output_dir, exist_ok=True)
    
    print("📊 开始解析训练日志...")
    data, best_breakpoints = parse_log(log_path)
    print(f"✅ 解析完成，共 {len(data['iterations'])} 轮数据")
    print(f"🎯 Best WR突破点: {best_breakpoints}")
    
    # 生成各图表
    plot_performance_curve(data, best_breakpoints, 
                          os.path.join(output_dir, 'performance_curve.png'))
    plot_sp_trend(data, os.path.join(output_dir, 'sp_trend.png'))
    plot_entropy_curve(data, os.path.join(output_dir, 'entropy_curve.png'))
    plot_loss_curves(data, os.path.join(output_dir, 'loss_curves.png'))
    plot_training_efficiency(data, best_breakpoints, 
                           os.path.join(output_dir, 'training_efficiency.png'))
    
    print("\n🎉 所有图表生成完成！")

if __name__ == '__main__':
    main()
