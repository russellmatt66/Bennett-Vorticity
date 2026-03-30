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

uz_data = pd.read_csv('../../experimental_data/zap_2001/zap2001_uz_fig5.csv')

uz_data.columns = ['Radius (mm)', 'uz (10^{4} m / s)']

r_data = uz_data['Radius (mm)'].to_numpy() * 1e-3 # Convert to meters
uz_data = uz_data['uz (10^{4} m / s)'].to_numpy() * 1e4 # Convert to m/s

uzpos = uz_data[r_data > 0]
uzneg = uz_data[r_data < 0] 
rpos = r_data[r_data > 0]
rneg = r_data[r_data < 0] 

"""
Make two fits of bulk, chi=2 cubic vortex profile to each half of the data, and then incorporate edge-localized vortices to model shear layer.
"""
n0 = 1e22 # Plasma density [m^-3]; 1e22 - 1e23
Tp = 200 * cnst.eV_to_K # Plasma temperature [K]; T = Te + Ti = 150 - 200 eV is experimental temperature of Zap 2001
# uedge = 4e4 # Edge flow velocity [m/s]; % Tie to dataset for better fidelity but 4e4 m/s is a reasonable estimate based on the data, which shows edge velocities around 40 km/s (4e4 m/s) at r = +/- 10 mm. This is consistent with the observed velocity profile, where the velocity at the edge (r = +/- 10 mm) is approximately 40 km/s. Using this value for uedge allows us to anchor our fits to the experimental data more accurately, ensuring that the reconstructed velocity profiles align well with the observed edge velocities in the Zap 2001 dataset.
u0 = 10e4 # Core flow velocity [m/s]; 
# rp = 10e-3 # Pinch radius [m]; 10mm

# Calculate edge velocities, core velocities, and pinch radii from uz_data
# Need to find location of shear layer - that gives everything
peidx = np.argmax(np.gradient(uzpos)) + 1 # Edge index needs to be shifted 
neidx = np.argmax(np.gradient(uzneg)) + 1 # 

rp_pos = rpos[peidx] # Pinch radius is location of maximum velocity gradient
rp_neg = np.abs(rneg[neidx]) # Pinch radius

uedge_pos = uzpos[peidx] # Edge velocity is velocity at location of maximum velocity gradient
uedge_neg = uzneg[neidx] # Edge velocity is velocity at location of maximum velocity gradient

uedge_elv_pos = uzpos[-1] # Edge velocity is velocity at last point in dataset
uedge_elv_neg = uzneg[0] # Edge velocity is velocity at first point in dataset

rp_elv_pos = rpos[-1] - rp_pos # Pinch radius is location of last point in dataset
rp_elv_neg = np.abs(rneg[0]) - rp_neg # Pinch radius is location of first point in dataset

print(f'Pinch radius (positive side) = {rp_pos*1e3} mm')
print(f'Pinch radius (negative side) = {rp_neg*1e3} mm')

print(f'Positive HC ELV pinch radius = {rp_elv_pos*1e3} mm')
print(f'Negative HC ELV pinch radius = {rp_elv_neg*1e3} mm')

print(f'Edge velocity (+HC) = {uedge_pos/1e3} km/s')
print(f'Edge velocity (-HC) = {uedge_neg/1e3} km/s')

print(f'ELV Edge velocity (+HC) = {uedge_elv_pos/1e3} km/s')
print(f'ELV Edge velocity (-HC) = {uedge_elv_neg/1e3} km/s')

# uz0_pos = cpfm.root_solve_chi2_negbulk(uedge, u0, n0, rp, Tp)
uz0_pos = cpfm.root_solve_chi2_posbulk(uedge_pos, u0, n0, rp_pos, Tp)
uz0_neg = cpfm.root_solve_chi2_negbulk(uedge_neg, u0, n0, rp_neg, Tp)

uz0_elv_pos = cpfm.root_solve_chi2_negbulk(uedge_elv_pos, uedge_pos, n0, rp_elv_pos, Tp)
uz0_elv_neg = cpfm.root_solve_chi2_negbulk(uedge_elv_neg, uedge_neg, n0, rp_elv_neg, Tp)

# When to use real uz0 vs magnitude? Real uz0 loses energy when it becomes complex, but magnitude may overestimate the velocity if uz0 is complex, leading to unphysical results if uz0 > c becomes a solution.
# uz0_pos_real = np.real(uz0_pos)
# uz0_neg_real = np.real(uz0_neg)
# print(f'uz0_pos = {uz0_pos_real} m/s')
# print(f'uz0_neg = {uz0_neg_real} m/s')

num_r = 500
r_pos = np.linspace(0, rp_pos, num_r)
r_neg = np.linspace(0, rp_neg, num_r)

num_r_elv = 500
r_elv_pos = np.linspace(0, rp_elv_pos, num_r_elv)
r_elv_neg = np.linspace(0, rp_elv_neg, num_r_elv)

cbt = []
cbt_elv = []
uzpos_fits = []
uzneg_fits = []
uzpos_elv_fits = []
uzneg_elv_fits = []

root_num = 1
for uz0p, uz0n, uz0p_elv, uz0n_elv in zip(uz0_pos, uz0_neg, uz0_elv_pos, uz0_elv_neg):
    uz0p = np.abs(uz0p) # Take magnitude
    uz0n = np.abs(uz0n) # Take magnitude
    uz0p_elv = np.abs(uz0p_elv) # Take magnitude
    uz0n_elv = np.abs(uz0n_elv) # Take magnitude
    cbt_temp_pos = cpfm.cbt(n0, uz0p, rp_pos, Tp) # Vortex constant [m]
    cbt_temp_neg = cpfm.cbt(n0, uz0n, rp_neg, Tp) # Vortex constant [m]
    cbt_temp_pos_elv = cpfm.cbt(n0, uz0p_elv, rp_elv_pos, Tp) # Vortex constant [m]
    cbt_temp_neg_elv = cpfm.cbt(n0, uz0n_elv, rp_elv_neg, Tp) # Vortex constant [m]
    print(f'cbt for uz0 = {uz0p} m/s: {cbt_temp_pos} m')
    cbt.append((cbt_temp_pos, cbt_temp_neg))
    cbt_elv.append((cbt_temp_pos_elv, cbt_temp_neg_elv))

    plt.figure()
    uzpos_fit = cpfm.uz_chi2cubic_posbulk(cbt_temp_pos, uz0p, u0, r_pos)
    # uzpos_fit = cpfm.uz_chi2cubic_negbulk(cbt_temp, uz0, u0, rpos)
    uzneg_fit = cpfm.uz_chi2cubic_negbulk(cbt_temp_neg, uz0n, u0, r_neg) # Make rneg positive for calculating
    
    uzpos_fit_elv = cpfm.uz_chi2cubic_negbulk(cbt_temp_pos_elv, uz0p_elv, uedge_pos, r_elv_pos) # Shift rpos for edge-localized vortex
    uzneg_fit_elv = cpfm.uz_chi2cubic_negbulk(cbt_temp_neg_elv, uz0n_elv, uedge_neg, r_elv_neg) # Shift rneg for edge-localized vortex

    uzpos_fits.append(uzpos_fit)
    uzneg_fits.append(uzneg_fit)
    uzpos_elv_fits.append(uzpos_fit_elv)
    uzneg_elv_fits.append(uzneg_fit_elv)

    # plt.plot(rpos[:peidx] * 1e3, uzpos_fit / 1e3, 'bo', label='Bulk, $\chi=2$, positive cubic vortex')
    # plt.plot(rneg[neidx:] * 1e3, uzneg_fit / 1e3, 'ro', label='Bulk, $\chi=2$, negative cubic vortex')
    # plt.plot(rpos[peidx:] * 1e3, uzpos_fit_elv / 1e3, 'b--', label='Edge-localized, $\chi=2$, positive cubic vortex')
    # plt.plot(rneg[:neidx] * 1e3, uzneg_fit_elv / 1e3, 'r--', label='Edge-localized, $\chi=2$, negative cubic vortex')
    plt.plot(r_pos * 1e3, uzpos_fit / 1e3, 'bo', label='Bulk, $\chi=2$, positive cubic vortex')
    plt.plot(-r_neg * 1e3, uzneg_fit / 1e3, 'ro', label='Bulk, $\chi=2$, negative cubic vortex')
    plt.plot((rp_pos + r_elv_pos) * 1e3, uzpos_fit_elv / 1e3, 'b--', label='Edge-localized, $\chi=2$, positive cubic vortex')
    plt.plot(-(rp_neg + r_elv_neg) * 1e3, uzneg_fit_elv / 1e3, 'r--', label='Edge-localized, $\chi=2$, negative cubic vortex')
    plt.plot(r_data * 1e3, uz_data / 1e3, 'kx', label='Zap 2001 Axial Velocity Data')

    # plt.title(f'Analytic reconstruction of Zap 2001 axial velocity data, Root {root_num}, $r_p = {rp*1e3}$ mm, $n_0 = {n0:.2e}$ m$^{{-3}}$, $T_p = {Tp / cnst.eV_to_K}$ eV, $u_0 = {u0 / 1e3} $ km/s)')
    plt.xlabel('Radius (mm)')
    plt.ylabel('Axial Velocity (km/s)')

    plt.ylim(0, 150)

    plt.legend()
    root_num += 1

# for elv_fit in uzpos_elv_fits:
#     print(f'uzpos_elv_fit = {elv_fit}')

for nelv_fit in uzneg_elv_fits:
    print(f'uzneg_elv_fit = {nelv_fit}')

# RRMSEpos = []
# RRMSEpos_all = []
# for uz_fit in uzpos_fits:
#     rmse_all = np.sqrt(mean_squared_error(uzpos, uz_fit)) 
#     rmse = np.sqrt(mean_squared_error(uzpos[:-1-1], uz_fit[:-1-1])) # Exclude the last two points
#     rrmse_all = rmse_all / np.mean(uzpos)
#     rrmse = rmse / np.mean(uzpos)
#     RRMSEpos_all.append(rrmse_all)
#     RRMSEpos.append(rrmse)

# RRMSEneg = []
# RRMSEneg_all = []
# for uz_fit in uzneg_fits:
#     # print(f'uzfitneg = {uz_fit}')
#     rmse_all = np.sqrt(mean_squared_error(uzneg, uz_fit)) 
#     rmse = np.sqrt(mean_squared_error(uzneg[1:-1], uz_fit[1:-1])) # Exclude the two points furthest from core 
#     rrmse_all = rmse_all / np.mean(uzneg) # Use mean of absolute values for normalization
#     rrmse = rmse / np.mean(uzneg) # Use mean of absolute values for normalization
#     RRMSEneg_all.append(rrmse_all)
#     RRMSEneg.append(rrmse)

# print(f'RRMSEpos_all = {RRMSEpos_all}')
# print(f'RRMSEpos = {RRMSEpos}')

# print(f'RRMSEneg_all = {RRMSEneg_all}')
# print(f'RRMSEneg = {RRMSEneg}')

# Calculate plasma properties
# for uz0p, uz0n in zip(uz0_pos, uz0_neg):
#     cbt_temp = cpfm.cbt(n0, np.abs(uz0p), rp, Tp) # Vortex constant [m]
#     p0 = cpfm.p0(cbt_temp, n0, np.abs(uz0p), rp) # Core plasma pressure [Pa]
#     Bmax = np.abs(cpfm.btheta_chi2_negbulk(cbt_temp, np.abs(uz0p), u0, n0, rp)) # Edge magnetic field [T]
#     tauE = cpfm.tauE(p0, np.abs(uz0p), rp, Tp, spz.KappaPerp_spitzer_e(n0, Tp, pp.omega_ce(Bmax), spz.tau_e(n0, Tp, spz.coulombLog_ei(n0, Tp, 1)), spz.coulombLog_ei(n0, Tp, 1))) # Energy confinement time [s]
#     tauA = rp / pp.vA(Bmax, n0) # Alfvén time [s]
#     # peak_shear = cpfm.peakshear_chi2cubic()
#     peak_shear = (8.0 / 27.0) * uz0p / cbt_temp 

#     print(f'For uz0 = {uz0p} m/s:')
#     print(f'  cbt = {cbt_temp} m')
#     print(f'  p0 = {p0} Pa')
#     print(f'  Bmax = {Bmax} T')
#     print(f'  tauE = {tauE} s')
#     print(f'  tauA = {tauA} s')
#     print(f'  tauE / tauA = {tauE / tauA}')
#     print(f'  Peak shear = {peak_shear} s^-1')

plt.show()