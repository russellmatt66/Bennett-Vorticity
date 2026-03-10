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
uzpos_list = []
uzneg_list = []
rpos_list = []
rneg_list = []

uedge_pos = []
uedge_neg = []
u0 = []

for uz_df in uz_df_list:
    r_data = uz_df['r (mm)'].to_numpy() * 1e-3 # Convert to meters
    uz_data = uz_df['uz (km/s)'].to_numpy() * 1e3 # Convert to m/s

    uzpos = uz_data[r_data > 0]
    uzneg = uz_data[r_data < 0] 
    rpos = r_data[r_data > 0]
    rneg = r_data[r_data < 0] 

    uzpos_list.append(uzpos)
    uzneg_list.append(uzneg)
    rpos_list.append(rpos)
    rneg_list.append(rneg)

    uedge_pos.append(uz_df.loc[uz_df['r (mm)'] == uz_df['r (mm)'].max(), 'uz (km/s)'].values[0] * 1e3) # Convert to m/s
    # uedge_pos.append(uzpos[np.where(r_data == r_data.max())[0][0]] * 1e3) # Convert to m/s
    uedge_neg.append(uz_df.loc[uz_df['r (mm)'] == uz_df['r (mm)'].min(), 'uz (km/s)'].values[0] * 1e3) # Convert to m/s
    # uedge_neg.append(uzneg[np.where(r_data == r_data.min())[0][0]] * 1e3) # Convert to m/s
    # u0.append(uz_data['uz (km/s)'].max() * 1e3) # Convert to m/s
    u0.append(uz_df.loc[uz_df['r (mm)'].abs().idxmin(), 'uz (km/s)'] * 1e3) # Convert to m/s

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
n0 = 1e23 # Plasma density [m^-3]; 1e22 - 1e23
# rp = 10e-3 # Pinch radius [m];   
Tp = 200 * cnst.eV_to_K # Plasma temperature [K]; T = Te + Ti = 150 - 200 eV
# uedge = 4e4 # Edge flow velocity [m/s]; 
# u0 = 10e4 # Core flow velocity [m/s]; 

cbt_pos = []
cbt_neg = []

# Manually recorded from Figure 9 of Shumlak et. al (2009) Nucl Fusion 49 075039
rp_pos = [15e-3, 5e-3, 15e-3, 15e-3, 10e-3] # Pinch radius [m];
rp_neg = [20e-3, 25e-3, 15e-3, 15e-3, 25e-3]

uzpos_fits = []
uzneg_fits = []

"""
Plot
"""
for i, uz_df in enumerate(uz_df_list):
    # cbt_temp = cpfm.cbt(n0, uz0_mag, rp, Tp) # Vortex constant [m]
    # print(f'cbt for uz0 = {uz0_mag} m/s: {cbt_temp} m')
    # cbt.append(cbt_temp)
    plt.figure()
    plt.plot(uz_df['r (mm)'], uz_df['uz (km/s)'])
    plt.title(f'Axial Velocity, Zap 2009, $\\tau$ = {uz_df["name"].iloc[0]}')
    # plt.plot(rpos_list[i], uzpos_list[i])
    # plt.plot(rneg_list[i], uzneg_list[i])
    # uzpos_fit = cpfm.uz_chi2cubic_posbulk(cbt_temp, uz0_mag, u0, rpos_list[i])
    # uzneg_fit = cpfm.uz_chi2cubic_negbulk(cbt_temp, uz0_mag, u0, -rneg_list[i]) # Make rneg positive for calculating
    
plt.show()