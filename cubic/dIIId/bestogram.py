'''
Histogram of the front and wake accuracies for the best solutions to the MAST 2023 edge pedestal obtained in best.py
'''
import sys
import pathlib
# ensure project root is on sys.path so the sibling `modules` package is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load as DataFrames (no plotting here) and pull a numeric series from each.
bwr_df = pd.read_csv('../../analytic_fits/DIIID/best_wake_rrmses.csv', header=None, names=['rrmse'])
bfr_df = pd.read_csv('../../analytic_fits/DIIID/best_front_rrmses.csv', header=None, names=['rrmse'])

bwr_rrmses = pd.to_numeric(bwr_df['rrmse'], errors='coerce').dropna().astype(float).to_numpy()
bfr_rrmses = pd.to_numeric(bfr_df['rrmse'], errors='coerce').dropna().astype(float).to_numpy()

bwr_average = np.mean(bwr_rrmses)
bfr_average = np.mean(bfr_rrmses)

print(f'Average RRMSE for best wake solutions: {bwr_average:.4f}')
print(f'Average RRMSE for best front solutions: {bfr_average:.4f}')

bwr_std = np.std(bwr_rrmses)
bfr_std = np.std(bfr_rrmses)

print(f'Standard deviation of RRMSE for best wake solutions: {bwr_std:.4f}')
print(f'Standard deviation of RRMSE for best front solutions: {bfr_std:.4f}')

bwr_max = np.max(bwr_rrmses)
bfr_max = np.max(bfr_rrmses)

print(f'Max RRMSE for best wake solutions: {bwr_max:.4f}')
print(f'Max RRMSE for best front solutions: {bfr_max:.4f}')

bwr_min = np.min(bwr_rrmses)
bfr_min = np.min(bfr_rrmses)

print(f'Min RRMSE for best wake solutions: {bwr_min:.4f}')
print(f'Min RRMSE for best front solutions: {bfr_min:.4f}')

all_vals = np.concatenate([bwr_rrmses, bfr_rrmses])
bins = np.linspace(all_vals.min(), all_vals.max(), 21)  # 20 bins

N_sweep = bwr_rrmses.size # Both arrays come from same sweep

fig, ax = plt.subplots(1, 2, figsize = (10, 4), sharey=True)

ax[0].hist(bwr_rrmses, bins=bins, alpha=0.5)
ax[0].set_xlabel('Range-Normalized Relative Root Mean Squared Error (RRMSE)')
ax[0].set_ylabel('Frequency')
ax[0].set_title(f'Histogram of RRMSE for Best Wake Solutions to DIIID 2005 Edge Pedestal, N={N_sweep}')
ax[0].legend()

ax[1].hist(bfr_rrmses, bins=bins, alpha=0.5)
ax[1].set_xlabel('Range-Normalized Relative Root Mean Squared Error (RRMSE)')
ax[1].set_ylabel('Frequency')
ax[1].set_title(f'Histogram of RRMSE for Best Front Solutions to DIIID 2005 Edge Pedestal Sweep, N={N_sweep}')
ax[1].legend()

tick_edges = bins[::2]  # every 2nd edge so labels are readable
for a in ax:
    a.set_xticks(tick_edges)
    a.set_xticklabels([f'{x:.4f}' for x in tick_edges], rotation=45, ha='right')

plt.tight_layout()
plt.show()
