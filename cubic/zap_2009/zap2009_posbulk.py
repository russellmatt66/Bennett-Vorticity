import sys
import pathlib
# ensure project root is on sys.path so the sibling `modules` package is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from modules import constants as cnst
from modules import spitzer as spz
from modules import cubic_pureflow_module as cpfm
from modules import plasma_properties as pp
from modules import powerbalance as pb

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import mean_squared_error

uz_data = pd.read_csv('../../experimental_data/zap_2009/zap2009_uz_fig9.csv', header=0, skiprows=[1])

# print(uz_data.columns)

# uz_tau_neg_0pt10 = uz_data['uz_tau_neg_0pt10'].dropna()
# ***Could get rid of the 'name' field - just put tau into an array
uz_tau_neg_0pt10 = pd.DataFrame({
    'r (mm)': uz_data['uz_tau_neg_0pt10'],
    'uz (km/s)': uz_data['Unnamed: 1'],
    'name' : '-0.10'
}).dropna()

uz_tau_0pt10 = pd.DataFrame({
    'r (mm)': uz_data['uz_tau_0pt10'],
    'uz (km/s)': uz_data['Unnamed: 3'],
    'name' : '0.10'
}).dropna()

uz_tau_0pt16 = pd.DataFrame({
    'r (mm)': uz_data['uz_tau_0pt16'],
    'uz (km/s)': uz_data['Unnamed: 5'],
    'name' : '0.16'
}).dropna()

uz_tau_0pt34 = pd.DataFrame({
    'r (mm)': uz_data['uz_tau_0pt34'],
    'uz (km/s)': uz_data['Unnamed: 7'],
    'name': '0.34'
}).dropna()

uz_tau_0pt56 = pd.DataFrame({
    'r (mm)': uz_data['uz_tau_0pt56'],
    'uz (km/s)': uz_data['Unnamed: 9'],
    'name': '0.56'
}).dropna()

print(f'uz_tau_neg_0pt10: {uz_tau_neg_0pt10}')
print(f'uz_tau_0pt10: {uz_tau_0pt10}')
print(f'uz_tau_0pt16: {uz_tau_0pt16}')
print(f'uz_tau_0pt34: {uz_tau_0pt34}')
print(f'uz_tau_0pt56: {uz_tau_0pt56}')

# uz_data.columns = ['Radius (mm)', 'uz (10^{4} m / s)']

# r_data = uz_data['Radius (mm)'].to_numpy() * 1e-3 # Convert to meters
# uz_data = uz_data['uz (10^{4} m / s)'].to_numpy() * 1e4 # Convert to m/s

uz_df_list = [uz_tau_neg_0pt10, uz_tau_0pt10, uz_tau_0pt16, uz_tau_0pt34, uz_tau_0pt56]
# uzpos_list = []
# uzneg_list = []
# rpos_list = []
# rneg_list = []

uedge_pos = []
uedge_neg = []
u0 = []

rp_pos = []
rp_neg = []

r_core = [] # location of the pinch core based on the minimum velocity of the data [m]
rp_core_pos = [] # Pinch radius of the positive half-chord based on the location of the minimum velocity of the data [m]
rp_core_neg = [] # Pinch radius of the negative half-chord based on the location of the minimum velocity of the data [m]

r_balance = [] # point at which the positive and negative half-chords will have the same sized pinch radius
rp_balance_pos = [] # Pinch radius of the positive half-chord at the balance point
rp_balance_neg = [] # Pinch radius of the negative half-chord at the balance point
# Technically, the above lists (should be) are the same

for uz_df in uz_df_list:
    # Doesn't seem to be a point to the below
    # r_data = uz_df['r (mm)'].to_numpy() * 1e-3 # Convert to meters
    # uz_data = uz_df['uz (km/s)'].to_numpy() * 1e3 # Convert to m/s

    # uzpos = uz_data[r_data > 0]
    # uzneg = uz_data[r_data < 0] 
    # rpos = r_data[r_data > 0]
    # rneg = r_data[r_data < 0] 

    # uzpos_list.append(uzpos)
    # uzneg_list.append(uzneg)
    # rpos_list.append(rpos)
    # rneg_list.append(rneg)
    uedge_pos.append(uz_df.loc[uz_df['r (mm)'] == uz_df['r (mm)'].max(), 'uz (km/s)'].values[0] * 1e3) # Convert to m/s
    # uedge_pos.append(uzpos[np.where(r_data == r_data.max())[0][0]] * 1e3) # Convert to m/s
    uedge_neg.append(uz_df.loc[uz_df['r (mm)'] == uz_df['r (mm)'].min(), 'uz (km/s)'].values[0] * 1e3) # Convert to m/s
    # uedge_neg.append(uzneg[np.where(r_data == r_data.min())[0][0]] * 1e3) # Convert to m/s
    # u0.append(uz_data['uz (km/s)'].max() * 1e3) # Convert to m/s
    u0.append(uz_df.loc[uz_df['r (mm)'].abs().idxmin(), 'uz (km/s)'] * 1e3) # Convert to m/s
    r_core.append(uz_df.loc[uz_df['uz (km/s)'].idxmin(), 'r (mm)'] * 1e-3) # Convert to m
    length = uz_df.loc[uz_df['r (mm)'].abs().idxmax(), 'r (mm)'] - uz_df.loc[uz_df['r (mm)'].abs().idxmin(), 'r (mm)'] * 1e-3
    uz_df_core = uz_df.copy() # is this bad?
    uz_df_balance = uz_df.copy()
    uz_df_core['r (mm)'] -= r_core[-1] * 1e3 # Shift r so that core is at r=0
    rp_core_pos.append(uz_df_core.loc[uz_df_core['r (mm)'] > 0, 'r (mm)'].max() * 1e-3) # Convert to m
    rp_core_neg.append(-uz_df_core.loc[uz_df_core['r (mm)'] < 0, 'r (mm)'].min() * 1e-3) # Convert to m, make positive
    rp_pos.append(uz_df.loc[uz_df['r (mm)'] > 0, 'r (mm)'].max() * 1e-3) # Convert to m
    rp_neg.append(-uz_df.loc[uz_df['r (mm)'] < 0, 'r (mm)'].min() * 1e-3) # Convert to m, make positive
    r_balance.append(0.5 * (rp_pos[-1] + np.abs(rp_neg[-1]))) # Balance point is at the location where the positive and negative half-chords have the same sized pinch radius
    uz_df_balance['r (mm)'] -= r_balance[-1] * 1e3 # Shift r so that balance point is at r=0
    rp_balance_pos.append(uz_df_balance.loc[uz_df_balance['r (mm)'] > 0, 'r (mm)'].max() * 1e-3) # Convert to m
    rp_balance_neg.append(-uz_df_balance.loc[uz_df_balance['r (mm)'] < 0, 'r (mm)'].min() * 1e-3) # Convert to m, make positive

print(f'uedge_pos: {uedge_pos}')
print(f'uedge_neg: {uedge_neg}')
print(f'u0: {u0}')
# uzpos = uz_data[r_data > 0]
# uzneg = uz_data[r_data < 0] 
# rpos = r_data[r_data > 0]
# rneg = r_data[r_data < 0] 

"""
Make fits of Bennett vortices to each half-chord
"""
n0 = 1e22 # Plasma density [m^-3]; 1e22 - 1e23
# rp = 10e-3 # Pinch radius [m];   
Tp = 150 * cnst.eV_to_K # Plasma temperature [K]; T = Te + Ti = 150 - 200 eV
# uedge = 4e4 # Edge flow velocity [m/s]; 
# u0 = 10e4 # Core flow velocity [m/s]; 

# c.f. Figure 8. of Shumlak et. al (2009) Nucl Fusion 49 075039, which defines normalized time tau w.r.t proper time pulse
def t(tau):
    return 44 * tau + 34

cbts_pos = []
cbts_neg = []

# Manually recorded from Figure 9 of Shumlak et. al (2009) Nucl Fusion 49 075039. Not as accurate as reading from .csv
# rp_pos = [15e-3, 5e-3, 15e-3, 15e-3, 10e-3] # Pinch radius [m];
# rp_neg = [20e-3, 25e-3, 15e-3, 15e-3, 25e-3]

uzpos_fits = []
uzneg_fits = []

num_r = 100
r_uzpos = []
r_uzneg = []

for ir in range(len(rp_pos)):
    r_uzpos.append(np.linspace(0, rp_pos[ir], num_r))
    r_uzneg.append(np.linspace(0, rp_neg[ir], num_r))

uzpos_roots = []
uzneg_roots = []
"""
Solve
"""
for i in range(len(uz_df_list)):
    uz0_temp_pos = cpfm.root_solve_chi2_posbulk(uedge_pos[i], u0[i], n0, rp_pos[i], Tp)
    uz0_temp_neg = cpfm.root_solve_chi2_posbulk(uedge_neg[i], u0[i], n0, rp_neg[i], Tp)
    uzpos_roots.append(uz0_temp_pos)
    uzneg_roots.append(uz0_temp_neg)
    cbts_pos_temp = []
    cbts_neg_temp = []
    uzpos_temp = []
    uzneg_temp = []
    for uz0 in uz0_temp_pos:
        cbt_pos = cpfm.cbt(n0, np.abs(uz0), rp_pos[i], Tp) # Vortex constant [m]
        uzpos_fit = cpfm.uz_chi2cubic_posbulk(cbt_pos, np.abs(uz0), u0[i], r_uzpos[i])
        cbts_pos_temp.append(cbt_pos)
        uzpos_temp.append(uzpos_fit)
    for uz0 in uz0_temp_neg:
        cbt_neg = cpfm.cbt(n0, np.abs(uz0), rp_neg[i], Tp) # Vortex constant [m]
        uzneg_fit = cpfm.uz_chi2cubic_posbulk(cbt_neg, np.abs(uz0), u0[i], r_uzneg[i])
        cbts_neg_temp.append(cbt_neg)
        uzneg_temp.append(uzneg_fit)
    cbts_pos.append(tuple(cbts_pos_temp))
    cbts_neg.append(tuple(cbts_neg_temp))
    uzpos_fits.append(uzpos_temp) # This is a list of lists of arrays containing the analytic solutions for positive half-chord
    uzneg_fits.append(uzneg_temp) # " " " etc.., Negative half-chord

    # cbt_temp = cpfm.cbt(n0, uz0_mag, rp, Tp) # Vortex constant [m]
    # print(f'cbt for uz0 = {uz0_mag} m/s: {cbt_temp} m')
    # cbt.append(cbt_temp)
  
"""
Plot
"""
for i, uz_df in enumerate(uz_df_list):
    plt.figure()
    # plt.plot(uz_df['r (mm)'], uz_df['uz (km/s)'])
    plt.scatter(uz_df['r (mm)'], uz_df['uz (km/s)'])
    plt.title(f'Axial Velocity, Zap 2009, $\\tau$ = {uz_df["name"].iloc[0]}')
    for j in range(len(uzpos_fits[i])):
        plt.plot(r_uzpos[i] * 1e3, uzpos_fits[i][j] / 1e3, label=f'Root {j+1}p: uz0 = {uzpos_roots[i][j]:.3e}')

    for j in range(len(uzneg_fits[i])):
        plt.plot(-r_uzneg[i] * 1e3, uzneg_fits[i][j] / 1e3, label=f'Root {j+1}n: uz0 = {uzneg_roots[i][j]:.3e}')
    
    plt.legend()
    # plt.plot(rpos_list[i], uzpos_list[i])
    # plt.plot(rneg_list[i], uzneg_list[i])
    # uzpos_fit = cpfm.uz_chi2cubic_posbulk(cbt_temp, uz0_mag, u0, rpos_list[i])
    # uzneg_fit = cpfm.uz_chi2cubic_negbulk(cbt_temp, uz0_mag, u0, -rneg_list[i]) # Make rneg positive for calculating
    
plt.show()