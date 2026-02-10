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
Make two fits of bulk, chi=2 cubic vortex profile to each half of the data
"""
n0 = 1e23 # Plasma density [m^-3]; 1e22 - 1e23
Tp = 200 * cnst.eV_to_K # Plasma temperature [K]; T = Te + Ti = 150 - 200 eV
uedge = 4e4 # Edge flow velocity [m/s]; 
u0 = 10e4 # Core flow velocity [m/s]; 
rp = 10e-3 # Pinch radius [m]; 10mm

# uz0_pos = cpfm.root_solve_chi2_negbulk(uedge, u0, n0, rp, Tp)
uz0_pos = cpfm.root_solve_chi2_posbulk(uedge, u0, n0, rp, Tp)
uz0_neg = cpfm.root_solve_chi2_negbulk(uedge, u0, n0, rp, Tp)

# uz0_pos_real = np.real(uz0_pos)
# uz0_neg_real = np.real(uz0_neg)
# print(f'uz0_pos = {uz0_pos_real} m/s')
# print(f'uz0_neg = {uz0_neg_real} m/s')

uz0_mag = np.abs(uz0_pos) # Use the magnitude of the first root for the fit
print(f'uz0 = {uz0_mag}')

cbt = []
uzpos_fits = []
uzneg_fits = []
for uz0 in np.unique(uz0_mag):
    cbt_temp = cpfm.cbt(n0, uz0, rp, Tp) # Vortex constant [m]
    print(f'cbt for uz0 = {uz0} m/s: {cbt_temp} m')
    cbt.append(cbt_temp)

    plt.figure()
    uzpos_fit = cpfm.uz_chi2cubic_posbulk(cbt_temp, uz0, u0, rpos)
    uzneg_fit = cpfm.uz_chi2cubic_negbulk(cbt_temp, uz0, u0, -rneg) # Make rneg positive for calculating
    
    uzpos_fits.append(uzpos_fit)
    uzneg_fits.append(uzneg_fit)

    # plt.plot(rpos, uzpos_fit, 'bo', label='Bulk, $\chi=2$, positive cubic vortex')
    # plt.plot(rneg, uzneg_fit, 'ro', label='Bulk, $\chi=2$, negative cubic vortex')
    # plt.plot(r_data, uz_data, 'kx', label='Zap 2001 Axial Velocity Data')

    # plt.title(f'Fit of Bennett cubic vortices to Zap 2001 axial velocity data, $r_p = 10$ mm, $n_0 = 10^{{22}}$ m$^{{-3}}$, $T_p = 75$ eV, $u_0 = 10^5$ m/s, $uz0 = {uz0:.3e}$')
    # plt.xlabel('Radius (m)')
    # plt.ylabel('Axial Velocity (m/s)')

    # plt.ylim(0, 1.5e5)

    # plt.legend()

RRMSEpos = []
for uz_fit in uzpos_fits:
    rmse = np.sqrt(mean_squared_error(uzpos, uz_fit))
    rrmse = rmse / np.mean(uzpos)
    RRMSEpos.append(rrmse)

RRMSEneg = []
for uz_fit in uzneg_fits:
    rmse = np.sqrt(mean_squared_error(uzneg, uz_fit))
    rrmse = rmse / np.mean(np.abs(uzneg)) # Use mean of absolute values for normalization
    RRMSEneg.append(rrmse)

print(RRMSEpos)
print(RRMSEneg)

# Calculate plasma properties
for uz0 in np.unique(uz0_mag):
    cbt_temp = cpfm.cbt(n0, uz0, rp, Tp) # Vortex constant [m]
    p0 = cpfm.p0(cbt_temp, n0, uz0, rp) # Core plasma pressure [Pa]
    Bmax = np.abs(cpfm.btheta(cbt_temp, uz0, n0, rp)) # Edge magnetic field [T]
    tauE = cpfm.tauE(p0, uz0, rp, Tp, spz.KappaPerp_spitzer_e(n0, Tp, pp.omega_ce(Bmax), spz.tau_e(n0, Tp, spz.coulombLog_ei(n0, Tp, 1)), spz.coulombLog_ei(n0, Tp, 1))) # Energy confinement time [s]
    tauA = rp / pp.vA(Bmax, n0) # Alfvén time [s]
    # peak_shear = cpfm.peakshear_chi2cubic()
    peak_shear = (8.0 / 27.0) * uz0 / cbt_temp 

    print(f'For uz0 = {uz0} m/s:')
    print(f'  cbt = {cbt_temp} m')
    print(f'  p0 = {p0} Pa')
    print(f'  Bmax = {Bmax} T')
    print(f'  tauE = {tauE} s')
    print(f'  tauA = {tauA} s')
    print(f'  tauE / tauA = {tauE / tauA}')
    print(f'  Peak shear = {peak_shear} s^-1')

plt.show()