#!/usr/bin/env python3
"""Generate statistical assessment figure for presentation."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# TBS Color palette
COLORS = {
    'primary': '#990000',
    'secondary': '#4D4D4D',
    'accent1': '#990000',
    'accent2': '#CC3333',
    'accent3': '#660000',
    'highlight': '#990000',
    'light': '#666666',
    'connection': '#4D4D4D',
    'white': '#FFFFFF',
    'black': '#1C1C1C',
    'gray_light': '#E8E8E8',
    'gray_medium': '#999999',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
})

# Output directory
OUTPUT_DIR = PROJECT_ROOT / 'master_dissertation_ppt' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load master results
master_file = PROJECT_ROOT / 'data' / 'models' / 'master_results.json'
with open(master_file) as f:
    master_results = json.load(f)

tickers = master_results['tickers']
results = master_results['results']

# Collect statistics
stats_data = []
for ticker in tickers:
    if 'underlying_stats' in results[ticker]:
        stats = results[ticker]['underlying_stats']
        stats_data.append({
            'Ticker': ticker,
            'Current Vol (%)': stats['current_vol'] * 100,
            'Mean Vol (%)': stats['mean_vol'] * 100,
            'Vol of Vol': stats['vol_of_vol'],
            'Leverage Corr': stats['leverage_corr'],
            'κ Estimate': stats['kappa_estimate'],
            'Skewness': stats['skewness'],
            'Kurtosis': stats['kurtosis']
        })

stats_df = pd.DataFrame(stats_data)

# Create figure with table
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')

# Create table
table_data = []
table_data.append(list(stats_df.columns))
for _, row in stats_df.iterrows():
    table_data.append([
        row['Ticker'],
        f"{row['Current Vol (%)']:.2f}",
        f"{row['Mean Vol (%)']:.2f}",
        f"{row['Vol of Vol']:.4f}",
        f"{row['Leverage Corr']:.3f}",
        f"{row['κ Estimate']:.1f}",
        f"{row['Skewness']:.3f}",
        f"{row['Kurtosis']:.3f}"
    ])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.10, 0.13, 0.12, 0.11, 0.13, 0.11, 0.11, 0.11])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Header row styling
for i in range(len(stats_df.columns)):
    cell = table[(0, i)]
    cell.set_facecolor(COLORS['primary'])
    cell.set_text_props(weight='bold', color='white')
    cell.set_edgecolor(COLORS['white'])
    cell.set_linewidth(1.5)

# Data rows styling with alternating colors
for i in range(1, len(table_data)):
    for j in range(len(stats_df.columns)):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor(COLORS['gray_light'])
        else:
            cell.set_facecolor(COLORS['white'])
        cell.set_edgecolor(COLORS['gray_medium'])
        cell.set_linewidth(0.5)

        # Ticker column in bold
        if j == 0:
            cell.set_text_props(weight='bold')

plt.title('Underlying Securities Statistical Assessment',
          fontsize=16, fontweight='bold', color=COLORS['primary'], pad=20)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'statistical_assessment.pdf', bbox_inches='tight', dpi=300)
fig.savefig(OUTPUT_DIR / 'statistical_assessment.png', bbox_inches='tight', dpi=300)
plt.close(fig)

print(f"✓ Generated: statistical_assessment.pdf/png")
print(f"✓ Saved to: {OUTPUT_DIR}")
