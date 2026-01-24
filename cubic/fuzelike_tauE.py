import constants as cnst
import spitzer as spz
import cubic_pureflow_module as cpfm
import plasma_properties as pp

import numpy as np
'''
Calculating energy confinement time for FuZE-like cubic vortices
'''

n0 = 1e24 # Plasma density [m^-3]
Tp = 5e3 * cnst.eV_to_K # Plasma temperature [K]
uz0 = 2e5 # Edge flow velocity [m/s]
rp = 5e-3 # Pinch radius [m]

cbt = cpfm.cbt(n0, uz0, rp, Tp) # Vortex constant [m]
print(f'FuZE-like cbt = {cbt:.3e} m')

p0 = cpfm.p0(cbt, n0, uz0, rp) # Core plasma pressure [Pa]
print(f'FuZE-like p0 = {p0:.3e} Pa')

Z_h = 1 # Ionization state for hydrogen
lambda_C = spz.coulombLog_ei(n0, Tp, Z_h) # Coulomb logarithm
print(f'FuZE-like Coulomb logarithm = {lambda_C:.3f}')

Bmax = np.abs(cpfm.btheta(cbt, uz0, n0, rp)) # Edge magnetic field [T]
print(f'FuZE-like Edge Magnetic Field = {Bmax:.3e} T')

omega_ce_edge = pp.omega_ce(Bmax) # Electron cyclotron frequency at the edge [rad/s]
print(f'FuZE-like Edge Cyclotron Frequency = {omega_ce_edge:.3e} rad/s')

tauee = spz.tau_ee(n0, Tp, lambda_C) # Electron-electron collision time [s]
print(f'FuZE-like Electron-Electron Collision Time = {tauee:.3e} s')

kappa_perp = spz.edgeKappaPerp_spitzer(n0, Tp, omega_ce_edge, tauee, lambda_C) # Perpendicular thermal conductivity at the edge [W/m/K]
print(f'FuZE-like Edge Perpendicular Spitzer Thermal Conductivity = {kappa_perp:.3e} W/m/K')

tau_E = cpfm.tauE(p0, uz0, rp, Tp, kappa_perp) # Energy confinement time [s]
print(f'FuZE-like tauE = {tau_E:.3e} s')