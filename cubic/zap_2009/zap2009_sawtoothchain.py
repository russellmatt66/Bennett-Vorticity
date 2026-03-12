"""
Take experimental data and fit a chain of cubic, chi=2 vortices to it.
"""
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

uz_df_list = [uz_tau_neg_0pt10, uz_tau_0pt10, uz_tau_0pt16, uz_tau_0pt34, uz_tau_0pt56]

n0 = 1e22 # Plasma density [m^-3]; 1e22 - 1e23
Tp = 150 * cnst.eV_to_K # Plasma temperature [K]; T = Te + Ti ~ 150 eV

num_r = 100

def fit_vortex_chain(uz_df: pd.DataFrame) -> tuple[list[list[np.ndarray]], list[np.ndarray], list[list[float]]]:
    """
    Fit a chain of cubic, chi=2 vortices to the given uz data.
    """
    # ***Need to convert r from mm to m, and uz from km/s to m/s
    r_data = uz_df['r (mm)'].to_numpy() * 1e-3 # Convert to meters
    uz_data = uz_df['uz (km/s)'].to_numpy() * 1e3 # Convert to m/s

    uz_fits = []
    r_arrays = []
    cbts = []

    # ***Need to fit a chain of vortices, not just a single vortex
    N_vortices = uz_data.size - 1 # Essentially, how many cells are there = how many vortices
    nv = 0
    while nv < N_vortices:
        if (r_data[nv] < 0): # Fit left to right (negative half-plane)
            u0 = uz_data[nv+1]
            uedge = uz_data[nv]
            rp = (r_data[nv+1] - r_data[nv])
            # r_array = np.linspace(r_data[nv+1], r_data[nv], num_r) 
            r_array = np.linspace(0, rp, num_r) # Will shift later
            if u0 < uedge:
                uz0_roots = cpfm.root_solve_chi2_posbulk(uedge, u0, n0, rp, Tp)
            elif u0 > uedge:
                uz0_roots = cpfm.root_solve_chi2_negbulk(uedge, u0, n0, rp, Tp)
            analytic_solns = []
            cbts_temp = []
            for uz0 in uz0_roots:
                cbt = cpfm.cbt(n0, np.abs(uz0), rp, Tp)
                # print(f'cbt for uz0 = {uz0} m/s: {cbt} m')
                if u0 < uedge:
                    analytic_solns.append(cpfm.uz_chi2cubic_posbulk(cbt, np.abs(uz0), u0, r_array))
                elif u0 > uedge:
                    analytic_solns.append(cpfm.uz_chi2cubic_negbulk(cbt, np.abs(uz0), u0, r_array))
                cbts_temp.append(cbt)
            uz_fits.append(analytic_solns)
            r_array = np.linspace(r_data[nv+1], r_data[nv], num_r) # Shift r_array to correct location for plotting
            r_arrays.append(r_array)
            cbts.append(cbts_temp)
            nv += 1
        elif (r_data[nv] > 0): # Fit right to left (positive half-plane)
            u0 = uz_data[nv]
            uedge = uz_data[nv+1]
            rp = (r_data[nv] - r_data[nv+1])
            r_array = np.linspace(0, rp, num_r) # Will shift later
            if u0 < uedge:
                uz0_roots = cpfm.root_solve_chi2_posbulk(uedge, u0, n0, rp, Tp)
            elif u0 > uedge:
                uz0_roots = cpfm.root_solve_chi2_negbulk(uedge, u0, n0, rp, Tp)
            analytic_solns = []
            cbts_temp = []
            for uz0 in uz0_roots:
                cbt = cpfm.cbt(n0, np.abs(uz0), rp, Tp)
                # print(f'cbt for uz0 = {uz0} m/s: {cbt} m')
                if u0 < uedge:
                    analytic_solns.append(cpfm.uz_chi2cubic_posbulk(cbt, np.abs(uz0), u0, r_array))
                elif u0 > uedge:
                    analytic_solns.append(cpfm.uz_chi2cubic_negbulk(cbt, np.abs(uz0), u0, r_array))
                cbts_temp.append(cbt)
            uz_fits.append(analytic_solns)
            r_array = np.linspace(r_data[nv], r_data[nv+1], num_r) # Shift r_array to correct location for plotting
            r_arrays.append(r_array)
            cbts.append(cbts_temp)
            nv += 1

    return uz_fits, r_arrays, cbts

swtc_uz_0pt10, r_0pt10, cbts_0pt10 = fit_vortex_chain(uz_tau_neg_0pt10)
print(f'Vortex chain fitted for uz: {uz_tau_neg_0pt10["name"].iloc[0]}')

def plot_vortex_chain(uz_df: pd.DataFrame, uz_fits: list[list[np.ndarray]], r_arrays: list[np.ndarray], cbts: list[list[float]]):
    """
    Plot the given vortex chain fits against the given uz data.
    """
    for i in range(len(uz_fits)):
        for j in range(len(uz_fits[i])):
            plt.figure(j)
            if i == 0: # Plot experimental data first time through
                plt.scatter(uz_df['r (mm)'], uz_df['uz (km/s)'])
            plt.plot(r_arrays[i] * 1e3, uz_fits[i][j] / 1e3, label=f'Vortex {i+1}, Root {j+1}, cbt = {cbts[i][j]:.3e} m')
            plt.title(f'Vortex Chain fit to Zap 2009 axial velocity, $\\tau$ = {uz_df["name"].iloc[0]}')
            plt.legend()

plot_vortex_chain(uz_tau_neg_0pt10, swtc_uz_0pt10, r_0pt10, cbts_0pt10)

plt.show()