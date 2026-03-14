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
# from zap2009_pureflow import t # Don't do this, totally destroys execution

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

# Stiching the magnetic fields of the individual vortices together doesn't respect Ampere's Law globally because only the current enclosed 
# by the individual vortex is considered.
# To calculate the magnetic field of the vortex chain at a given point, we just need to consider the current enclosed. 
def fit_vortex_chain(uz_df: pd.DataFrame) -> tuple[list[list[np.ndarray]], list[np.ndarray], list[list[float]], list[list[complex]]]:
    """
    Fit a chain of cubic, chi=2 vortices to the given uz data.
    """
    # ***Need to convert r from mm to m, and uz from km/s to m/s
    r_data = uz_df['r (mm)'].to_numpy() * 1e-3 # Convert to meters
    uz_data = uz_df['uz (km/s)'].to_numpy() * 1e3 # Convert to m/s

    uz_fits = []
    r_arrays = []
    cbts = []

    uz0_allroots = []

    bfields = []

    # ***Need to fit a chain of vortices, not just a single vortex
    N_vortices = uz_data.size - 1 # Essentially, how many cells are there = how many vortices
    nv = 0
    while nv < N_vortices:
        if (r_data[nv] < 0): # Fit left to right (negative half-plane)
            u0 = uz_data[nv+1] 
            uedge = uz_data[nv]
            rp = np.abs(r_data[nv+1] - r_data[nv])
            # r_array = np.linspace(r_data[nv+1], r_data[nv], num_r) 
            r_array = np.linspace(0, rp, num_r) # Will shift later
            if u0 < uedge:
                uz0_roots = cpfm.root_solve_chi2_posbulk(uedge, u0, n0, rp, Tp)
            elif u0 > uedge:
                uz0_roots = cpfm.root_solve_chi2_negbulk(uedge, u0, n0, rp, Tp)
            analytic_solns = []
            analytic_bfields = []
            cbts_temp = []
            for uz0 in uz0_roots:
                cbt = cpfm.cbt(n0, np.abs(uz0), rp, Tp)
                # print(f'cbt for uz0 = {uz0} m/s: {cbt} m')
                if u0 < uedge:
                    analytic_solns.append(cpfm.uz_chi2cubic_posbulk(cbt, np.abs(uz0), u0, r_array))
                    analytic_bfields.append(cpfm.btheta_chi2_posbulk(cbt, np.abs(uz0), u0, n0, r_array))
                elif u0 > uedge:
                    analytic_solns.append(cpfm.uz_chi2cubic_negbulk(cbt, np.abs(uz0), u0, r_array))
                    analytic_bfields.append(cpfm.btheta_chi2_negbulk(cbt, np.abs(uz0), u0, n0, r_array))
                cbts_temp.append(cbt)
            uz_fits.append(analytic_solns)
            r_array = np.linspace(r_data[nv+1], r_data[nv], num_r) # Shift r_array to correct location for plotting
            r_arrays.append(r_array)
            cbts.append(cbts_temp)
            uz0_allroots.append(uz0_roots)
            bfields.append(analytic_bfields)
            nv += 1
        elif (r_data[nv] > 0): # Fit right to left (positive half-plane)
            u0 = uz_data[nv]
            uedge = uz_data[nv+1]
            rp = np.abs((r_data[nv] - r_data[nv+1]))
            r_array = np.linspace(0, rp, num_r) # Will shift later
            if u0 < uedge:
                uz0_roots = cpfm.root_solve_chi2_posbulk(uedge, u0, n0, rp, Tp)
            elif u0 > uedge:
                uz0_roots = cpfm.root_solve_chi2_negbulk(uedge, u0, n0, rp, Tp)
            analytic_solns = []
            analytic_bfields = []
            cbts_temp = []
            for uz0 in uz0_roots:
                cbt = cpfm.cbt(n0, np.abs(uz0), rp, Tp)
                # print(f'cbt for uz0 = {uz0} m/s: {cbt} m')
                if u0 < uedge:
                    analytic_solns.append(cpfm.uz_chi2cubic_posbulk(cbt, np.abs(uz0), u0, r_array))
                    analytic_bfields.append(cpfm.btheta_chi2_posbulk(cbt, np.abs(uz0), u0, n0, r_array))
                elif u0 > uedge:
                    analytic_solns.append(cpfm.uz_chi2cubic_negbulk(cbt, np.abs(uz0), u0, r_array))
                    analytic_bfields.append(cpfm.btheta_chi2_negbulk(cbt, np.abs(uz0), u0, n0, r_array))
                cbts_temp.append(cbt)
            uz_fits.append(analytic_solns)
            r_array = np.linspace(r_data[nv], r_data[nv+1], num_r) # Shift r_array to correct location for plotting
            r_arrays.append(r_array)
            cbts.append(cbts_temp)
            uz0_allroots.append(uz0_roots)
            bfields.append(analytic_bfields)
            nv += 1

    return uz_fits, r_arrays, cbts, uz0_allroots, bfields

swtc_uz_neg0pt10, r_neg0pt10, cbts_neg0pt10, uz0_allroots_neg0pt10, bfields_neg0pt10 = fit_vortex_chain(uz_tau_neg_0pt10)
print(f'Vortex chain fitted for uz: {uz_tau_neg_0pt10["name"].iloc[0]}')

swtc_uz_0pt10, r_0pt10, cbts_0pt10, uz0_allroots_0pt10, bfields_0pt10 = fit_vortex_chain(uz_tau_0pt10)
print(f'Vortex chain fitted for uz: {uz_tau_0pt10["name"].iloc[0]}')

swtc_uz_0pt16, r_0pt16, cbts_0pt16, uz0_allroots_0pt16, bfields_0pt16 = fit_vortex_chain(uz_tau_0pt16)
print(f'Vortex chain fitted for uz: {uz_tau_0pt16["name"].iloc[0]}')

swtc_uz_0pt34, r_0pt34, cbts_0pt34, uz0_allroots_0pt34, bfields_0pt34 = fit_vortex_chain(uz_tau_0pt34)
print(f'Vortex chain fitted for uz: {uz_tau_0pt34["name"].iloc[0]}')

swtc_uz_0pt56, r_0pt56, cbts_0pt56, uz0_allroots_0pt56, bfields_0pt56 = fit_vortex_chain(uz_tau_0pt56)
print(f'Vortex chain fitted for uz: {uz_tau_0pt56["name"].iloc[0]}')

nfig = 0

# Importing this breaks the code for some reason 
def t(tau: float) -> float:
    return 44 * tau + 34

def plot_vortex_chain(nfig: int, uz_df: pd.DataFrame, uz_fits: list[list[np.ndarray]], r_arrays: list[np.ndarray], cbts: list[list[float]], uz0_allroots: list[list[complex]]):
    """
    Plot the given vortex chain fits against the given uz data.
    """
    for i in range(len(uz_fits)):
        for j in range(len(uz_fits[i])):
            plt.figure(j + nfig)
            if i == 0: # Plot experimental data first time through
                # plt.scatter(uz_df['r (mm)'], uz_df['uz (km/s)'])
                plt.plot(uz_df['r (mm)'], uz_df['uz (km/s)'], 'b--', label='Experimental data')
            plt.plot(r_arrays[i] * 1e3, uz_fits[i][j] / 1e3, label=f'Root {j+1}, uz0 = {uz0_allroots[i][j]:.3e} m/s, cbt = {cbts[i][j]:.3e} m')
            plt.title(f'Vortex Chain fit to Zap 2009 axial velocity, $\\tau$ = {t(float(uz_df["name"].iloc[0]))} $\mu s$, $n0 = {n0:.1e}$ m$^{{-3}}$, $T_p = {Tp/cnst.eV_to_K:.1f}$ eV')
            plt.xlabel('Radius (mm)')
            plt.ylabel('Axial Velocity (km/s)')
            plt.legend()
         
plot_vortex_chain(nfig, uz_tau_neg_0pt10, swtc_uz_neg0pt10, r_neg0pt10, cbts_neg0pt10, uz0_allroots_neg0pt10)
nfig += len(swtc_uz_neg0pt10[0]) # Cubic, chi=2 vortices will all have four roots
plot_vortex_chain(nfig, uz_tau_0pt10, swtc_uz_0pt10, r_0pt10, cbts_0pt10, uz0_allroots_0pt10)
nfig += len(swtc_uz_0pt10[0]) 
plot_vortex_chain(nfig, uz_tau_0pt16, swtc_uz_0pt16, r_0pt16, cbts_0pt16, uz0_allroots_0pt16)
nfig += len(swtc_uz_0pt16[0])
plot_vortex_chain(nfig, uz_tau_0pt34, swtc_uz_0pt34, r_0pt34, cbts_0pt34, uz0_allroots_0pt34)
nfig += len(swtc_uz_0pt34[0])
plot_vortex_chain(nfig, uz_tau_0pt56, swtc_uz_0pt56, r_0pt56, cbts_0pt56, uz0_allroots_0pt56)
nfig += len(swtc_uz_0pt56[0])

"""
Solve for magnetic fields across both half-chords, and plot
"""
def Iencl(uz: np.ndarray, rprime: np.ndarray) -> np.ndarray:
    term1 = 2.0 * np.pi * cnst.q_e * n0 
    return -term1 * np.trapezoid(uz * rprime, rprime) 

def Btheta(Iencl: np.ndarray, r: np.ndarray) -> np.ndarray:
    # mu0 = 4 * np.pi * 1e-7
    return cnst.mu0 * Iencl / (2 * np.pi * r)

def plot_bfield_chain(nfig: int, uz_fits: list[list[np.ndarray]], r_arrays: list[np.ndarray], cbts: list[list[float]], uz0_allroots: list[list[float]]) -> None:
    r_array = np.concatenate(r_arrays)
    Iencl_array = []
    bfield_array = []
    # for i in range(len(uz_fits)):
    #     for j in range(len(uz_fits[i])):
    for j, segments in enumerate(zip(*uz_fits)):
        uz = np.concatenate(segments)
        Iencl_rootj = Iencl(uz, np.abs(r_array))
        bfield = Btheta(Iencl_rootj, np.abs(r_array))
        Iencl_array.append(Iencl_rootj)
        bfield_array.append(bfield)
        plt.figure(j + nfig)
        plt.plot(r_array * 1e3, bfield_array[j] * 1e3, label=f'Root {j+1}')
        # , uz0 = {uz0_allroots[i][j]:.3e} m/s, cbt = {cbts[i][j]:.3e} m')
        plt.legend()

plot_bfield_chain(nfig, swtc_uz_neg0pt10, r_neg0pt10, cbts_neg0pt10, uz0_allroots_neg0pt10)

"""
Compare these plots to the fields obtained by solving Ampere's law.

I think the problem with these plots is that they solve the magnetic field for a single vortex, not a chain of vortices.
Therefore, they don't capture the full magnetic field structure of the system, because they are treating each vortex like it's surrounded by vacuum.

Put simply, the magnetic field of a vortex chain is not just the sum of the fields of individual vortices in their own domains because then
globally Ampere's Law is not being respected. 
"""
# def plot_bfield_chain(nfig: int, uz_df: pd.DataFrame, bfields: list[list[np.ndarray]], r_arrays: list[np.ndarray], cbts: list[list[float]], uz0_allroots: list[list[float]]):
#     """
#     Plot the magnetic fields corresponding to the given vortex chain fits.
#     """
#     for i in range(len(bfields)):
#         for j in range(len(bfields[i])):
#             plt.figure(j + nfig)
#             # bfield = bfields[i][j]
#             # r_array = r_arrays[i]
#             cbt = cbts[i][j]
#             uz0 = uz0_allroots[i][j]
#             plt.plot(r_arrays[i] * 1e3, np.abs(bfields[i][j]), label=f'Root {j+1}, uz0 = {uz0:.3e} m/s, cbt = {cbt:.3e} m') # negative sign is just opposite direction in bfield
#             plt.title(f'Magnetic field of Vortex Chain fit to Zap 2009 axial velocity, $\\tau$ = {t(float(uz_df["name"].iloc[0]))} $\mu s$, $n0 = {n0:.1e}$ m$^{{-3}}$, $T_p = {Tp/cnst.eV_to_K:.1f}$ eV')
#             plt.xlabel('Radius (mm)')
#             plt.ylabel('Magnetic Field (T)')
#             plt.legend()

# plot_bfield_chain(nfig, uz_tau_neg_0pt10, bfields_neg0pt10, r_neg0pt10, cbts_neg0pt10, uz0_allroots_neg0pt10)
# nfig += len(bfields_neg0pt10[0])
# plot_bfield_chain(nfig, uz_tau_0pt10, bfields_0pt10, r_0pt10, cbts_0pt10, uz0_allroots_0pt10)
# nfig += len(bfields_0pt10[0])
# plot_bfield_chain(nfig, uz_tau_0pt16, bfields_0pt16, r_0pt16, cbts_0pt16, uz0_allroots_0pt16)
# nfig += len(bfields_0pt16[0])   
# plot_bfield_chain(nfig, uz_tau_0pt34, bfields_0pt34, r_0pt34, cbts_0pt34, uz0_allroots_0pt34)
# nfig += len(bfields_0pt34[0])
# plot_bfield_chain(nfig, uz_tau_0pt56, bfields_0pt56, r_0pt56, cbts_0pt56, uz0_allroots_0pt56)

plt.show()