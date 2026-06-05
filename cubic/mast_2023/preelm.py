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

'''
Two vortex fit to MAST pre-ELM current density profile, one with positive bulk flow and one with negative bulk flow.

Interior (Wake)
Exterior (Front)
'''
uz_df = pd.read_csv('../../experimental_data/mast_2023/Jtoroidal_MAST_preELM.csv')
uz_df.drop(columns=['Unnamed: 1'], inplace=True)
uz_df.sort_values(by='Radius (m)', inplace=True)

# print(uz_df.head())

# rminidx = uz_df['J_phi (MA / m^2)'].idxmin() # Index of minimum current density
# rminidx = uz_df['Radius (m)'].idxmin() # Index of minimum radius
rminidx = uz_df['J_phi (MA / m^2)'][:uz_df['J_phi (MA / m^2)'].idxmax()].idxmin() # Index of minimum current density before max
rmin = uz_df['Radius (m)'][rminidx] # Radius at minimum current density

rmax = uz_df['Radius (m)'].max() # Maximum radius
rmaxidx = uz_df['Radius (m)'].idxmax() # Index of maximum radius

Jmaxidx = uz_df['J_phi (MA / m^2)'].idxmax() # Index of maximum current density
r_Jmax = uz_df['Radius (m)'][Jmaxidx] # R at maximum current density
Jzmax = uz_df['J_phi (MA / m^2)'][Jmaxidx]

print(f'Max current density Jzmax = {Jzmax} MA/m^2 at R = {r_Jmax - rmin} m')

# Wake (positive bulk flow)
num_r = 1000
r_wake = np.linspace(0.0, r_Jmax - rmin, num_r) # Radial positions for wake fit, from 0 to R at max current density
n0 = 2e19 # Plasma density [m^-3]; 1e19 - 1e20
Tp = 250 * cnst.eV_to_K # Plasma temperature [K]; T = Te + Ti = 200 - 300 eV is experimental temperature of MAST pre-ELM
rp = r_Jmax - rmin # Pinch radius [m]; Tie to dataset for better fidelity   
u0 = 1e6 * uz_df['J_phi (MA / m^2)'][rminidx] / (cnst.q_e * n0) # Core flow velocity [m/s]; J = n e u => u = J / (n e)
uedge = 1e6 * uz_df['J_phi (MA / m^2)'][Jmaxidx] / (cnst.q_e * n0) # Edge flow velocity [m/s]; J = n e u => u = J / (n e)

print('Wake fit parameters:\n')
print(f'u0 = {u0} m/s')
print(f'uedge = {uedge} m/s')

# Solve flow roots 
uz0_roots = cpfm.root_solve_chi2_posbulk(uedge, u0, n0, rp, Tp)

wake_fits = []
wake_cbts = []
for uz0_root in uz0_roots:
    uz0 = np.abs(uz0_root) # Take magnitude
    cbt = cpfm.cbt(n0, uz0, rp, Tp) # Vortex constant [m]
    print(f'cbt for uz0 = {uz0:2e} m/s: {cbt} m')
    wake_cbts.append(cbt)

    uz_fit = cpfm.uz_chi2cubic_posbulk(cbt, uz0, u0, r_wake)
    wake_fits.append(uz_fit)


# Front (negative bulk flow)
r_front = np.linspace(r_Jmax - rmin, rmax - rmin, num_r) # Radial positions for front fit, from R at max current density to max radius

rp = np.abs(rmax - r_Jmax)
r_front_calc = np.linspace(0.0, rp, num_r) # Radial positions for calculating front fit, from 0 to rp

u0_front = 1e6 * Jzmax / (cnst.q_e * n0) # Core flow velocity [m/s]; J = n e u => u = J / (n e)
uedge_front = 1e6 * uz_df['J_phi (MA / m^2)'][rmaxidx] / (cnst.q_e * n0) # Edge flow velocity [m/s]; J = n e u => u = J / (n e)

print('Front fit parameters:\n')
print(f'rp = {rp} m')
print(f'u0 = {u0_front} m/s')
print(f'uedge = {uedge_front} m/s')

uz0_roots = cpfm.root_solve_chi2_negbulk(uedge_front, u0_front, n0, rp, Tp)

front_fits = []
front_cbts = []
for uz0_root in uz0_roots:
    uz0 = np.abs(uz0_root) # Take magnitude
    cbt = cpfm.cbt(n0, uz0, rp, Tp) # Vortex constant [m]
    print(f'cbt for uz0 = {uz0} m/s: {cbt} m')
    front_cbts.append(cbt)

    uz_fit = cpfm.uz_chi2cubic_negbulk(cbt, uz0, u0_front, r_front_calc)
    front_fits.append(uz_fit)

plt.figure()
plt.plot(uz_df['Radius (m)'][rminidx:] - rmin, uz_df['J_phi (MA / m^2)'][rminidx:], label='MAST Pre-ELM J_phi')

for i, uz_fit in enumerate(wake_fits):
    plt.plot(r_wake, uz_fit / 1e6 * cnst.q_e * n0, label=f'Root {i+1}, cbt = {wake_cbts[i]:.4f} m')

for i, uz_fit in enumerate(front_fits):
    plt.plot(r_front, uz_fit / 1e6 * cnst.q_e * n0, label=f'Root {i+1}, cbt = {front_cbts[i]:.4f} m')

plt.title(f'Cubic vortex solutions to MAST pre-ELM toroidal current density profile, $r_p = {rp:.3f}$ m, $n_0 = {n0:.2e}$ m$^{{-3}}$, $T_p = {Tp / cnst.eV_to_K}$ eV, $u_0 = {u0_front / 1e6:.2f}$ Mm/s)')  
plt.xlabel('Radius (m)')
plt.ylabel('$J_\\phi$ (MA/m$^2$)')

plt.legend()

# Just plot the best roots together
plt.figure()
plt.plot(uz_df['Radius (m)'][rminidx:] - rmin, uz_df['J_phi (MA / m^2)'][rminidx:], label='MAST Pre-ELM J_phi')
plt.plot(r_wake, wake_fits[2] / 1e6 * cnst.q_e * n0, label=f'Wake fit, cbt = {wake_cbts[2]:.4f} m')
plt.plot(r_front, front_fits[2] / 1e6 * cnst.q_e * n0, label=f'Front fit, cbt = {front_cbts[2]:.4f} m')

plt.title(f'Cubic vortex solutions to MAST pre-ELM toroidal current density profile, $r_p = {rp:.3f}$ m, $n_0 = {n0:.2e}$ m$^{{-3}}$, $T_p = {Tp / cnst.eV_to_K}$ eV, $u_0 = {u0_front / 1e6:.2f}$ Mm/s)')
plt.xlabel('Radius (m)')
plt.ylabel('$J_\\phi$ (MA/m$^2$)')

plt.legend()

RRMSEwake = []
uz_wake = uz_df['J_phi (MA / m^2)'][rminidx:Jmaxidx] / (cnst.q_e * n0) * 1e6 # Convert J to uz for wake region
for uz_fit in wake_fits:
    uz_fit_interp = np.interp(uz_df['Radius (m)'][rminidx:Jmaxidx] - rmin, r_wake, uz_fit) # Interpolate fit to data points
    rmse = np.sqrt(mean_squared_error(uz_wake, uz_fit_interp))
    # rrmse = rmse / np.mean(uz_wake) # Mean normalization skews accuracy because of high variance in current density over small region
    rrmse = rmse / (uz_wake.max() - uz_wake.min()) # Normalize by range of data
    RRMSEwake.append(rrmse)

print(f'RRMSE for wake fits: {RRMSEwake}')

RRMSEfront = []
uz_front = uz_df['J_phi (MA / m^2)'][Jmaxidx:rmaxidx] / (cnst.q_e * n0) * 1e6 # Convert J to uz for front region
for uz_fit in front_fits:
    uz_fit_interp = np.interp(uz_df['Radius (m)'][Jmaxidx:rmaxidx] - rmin, r_front, uz_fit) # Interpolate fit to data points
    rmse = np.sqrt(mean_squared_error(uz_front, uz_fit_interp))
    # rrmse = rmse / np.mean(uz_front) # Mean normalization skews accuracy because of high variance in current density over small region
    rrmse = rmse / (uz_front.max() - uz_front.min()) # Normalize by range of data
    RRMSEfront.append(rrmse)

print(f'RRMSE for front fits: {RRMSEfront}')

plt.show()