#!/usr/bin/env python3
"""Generate training report PDF with charts"""

import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# Parse log file
log_path = "/Users/leo/Projects/kaggle/maze-crawler/train_v17_v49.log"

iterations = []
best_wr = []
entropy = []
eval_iters = []
eval_r = []

# Extract hyperparameters from first lines
hp_lines = []
with open(log_path, 'r') as f:
    for line in f:
        if line.strip().startswith('Step') or line.strip().startswith('Terminal') or line.strip().startswith('Training'):
            hp_lines.append(line.strip())
        elif line.strip().startswith('[') and 'SP=' in line:
            break

# Now parse the training data
with open(log_path, 'r') as f:
    for line in f:
        # Match iteration lines: [  1/2000] SP= 19.0% best=  0.0% p=-0.0007 v=2.8761 e=0.580 n=3613 t=10s
        match = re.search(r'\[\s*(\d+)/2000\].*best=\s*([\d.]+)%.*e=([\d.]+)', line)
        if match:
            iter_num = int(match.group(1))
            best = float(match.group(2))
            e = float(match.group(3))
            
            iterations.append(iter_num)
            best_wr.append(best)
            entropy.append(e)
        
        # Match eval lines: eval_R= 38.0%
        eval_match = re.search(r'\[\s*(\d+)/2000\].*eval_R=\s*([\d.]+)%', line)
        if eval_match:
            iter_num = int(eval_match.group(1))
            r = float(eval_match.group(2))
            eval_iters.append(iter_num)
            eval_r.append(r)

# Find best eval WR milestones
milestones = []
current_best = 0
with open(log_path, 'r') as f:
    for line in f:
        if "New best eval WR" in line:
            match = re.search(r'New best eval WR (\d+)% saved.*\[(\d+)/', line)
            if match:
                wr = int(match.group(1))
                iter_num = int(match.group(2))
                milestones.append((iter_num, wr))

# Create PDF
with PdfPages('/Users/leo/Projects/kaggle/maze-crawler/training_report_v49.pdf') as pdf:
    # Page 1: Summary
    plt.figure(figsize=(11, 8.5))
    plt.axis('off')
    
    summary_text = """
    Maze Crawler Training Report - v49
    ===================================
    
    Final Results:
    - Best Eval WR: 96% (achieved at iteration 1820)
    - Final Eval vs Random: 226W-186L-88D (45.2%)
    - Total Iterations: 2000
    - Total Training Time: ~15.9 hours (57222 seconds)
    
    Key Milestones:
    """
    
    for iter_num, wr in milestones:
        summary_text += f"  - {wr}% at iteration {iter_num}\n"
    
    summary_text += "\nHyperparameters:\n"
    for line in hp_lines:
        summary_text += f"  {line}\n"
    
    summary_text += """
    Training Observations:
    - Early rapid improvement: 0% → 47% in first 240 iterations
    - Steady progress through mid-training: 47% → 91% by iteration 1260
    - Continued refinement in late stage: 91% → 96% by iteration 1820
    - Entropy steadily decreased from 0.58 to ~0.18, indicating increasing policy certainty
    - Policy value head steadily increased from 2.88 to ~2.07, indicating improved win probability estimation
    """
    
    plt.text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top', family='monospace')
    plt.title('Maze Crawler Training Report - v49', fontsize=16, y=0.98)
    pdf.savefig()
    plt.close()
    
    # Page 2: Best WR Chart
    plt.figure(figsize=(11, 8.5))
    plt.plot(iterations, best_wr, linewidth=1.5, color='blue')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Win Rate (%)', fontsize=12)
    plt.title('Best Win Rate vs Training Iteration', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    # Annotate milestones
    for iter_num, wr in milestones:
        plt.annotate(f'{wr}%', xy=(iter_num, wr), xytext=(iter_num+50, wr+3),
                     arrowprops=dict(arrowstyle='->', color='red'), fontsize=9)
    
    pdf.savefig()
    plt.close()
    
    # Page 3: Eval R Chart
    plt.figure(figsize=(11, 8.5))
    plt.plot(eval_iters, eval_r, linewidth=1.5, color='green', marker='o', markersize=3)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Eval Win Rate vs Random (%)', fontsize=12)
    plt.title('Evaluation Win Rate vs Random Agent', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    # Annotate milestones
    for iter_num, wr in milestones:
        plt.annotate(f'{wr}%', xy=(iter_num, wr), xytext=(iter_num+50, wr+3),
                     arrowprops=dict(arrowstyle='->', color='red'), fontsize=9)
    
    pdf.savefig()
    plt.close()
    
    # Page 4: Entropy Chart
    plt.figure(figsize=(11, 8.5))
    plt.plot(iterations, entropy, linewidth=1.5, color='purple')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Policy Entropy', fontsize=12)
    plt.title('Policy Entropy Over Training', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(iterations, entropy, 3)
    p = np.poly1d(z)
    plt.plot(iterations, p(iterations), "r--", alpha=0.5, label='Trend Line')
    plt.legend()
    
    pdf.savefig()
    plt.close()
    
    # Page 5: Combined chart
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 12))
    
    # Best WR
    ax1.plot(iterations, best_wr, linewidth=1.5, color='blue')
    ax1.set_ylabel('Best WR (%)', fontsize=10)
    ax1.set_title('Training Metrics Summary', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Eval R
    ax2.plot(eval_iters, eval_r, linewidth=1.5, color='green', marker='o', markersize=2)
    ax2.set_ylabel('Eval R (%)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Entropy
    ax3.plot(iterations, entropy, linewidth=1.5, color='purple')
    ax3.set_xlabel('Iteration', fontsize=10)
    ax3.set_ylabel('Entropy', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pdf.savefig()
    plt.close()

print("Report generated successfully!")
print(f"PDF saved to: /Users/leo/Projects/kaggle/maze-crawler/training_report_v49.pdf")
