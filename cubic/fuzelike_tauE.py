# import constants as cnst
# import spitzer as spz
# import cubic_pureflow_module as cpfm
# import plasma_properties as pp

import sys
import pathlib
# ensure project root is on sys.path so the sibling `modules` package is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from modules import constants as cnst
from modules import spitzer as spz
from modules import cubic_pureflow_module as cpfm
from modules import plasma_properties as pp
from modules import powerbalance as pb

import numpy as np
'''
Calculating energy confinement time for FuZE-like cubic vortices
'''

n0 = 1e24 # Plasma density [m^-3]
Tp = 1e2 * cnst.eV_to_K # Plasma temperature [K]
uz0 = 5e4 # Edge flow velocity [m/s]
rp = 10e-3 # Pinch radius [m]

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
omega_ci_edge = pp.omega_ci(Bmax, Z_h) # Ion cyclotron frequency at the edge [rad/s]
print(f'FuZE-like Edge Electron Cyclotron Frequency = {omega_ce_edge:.3e} rad/s')
print(f'FuZE-like Edge Ion Cyclotron Frequency = {omega_ci_edge:.3e} rad/s')

taue = spz.tau_e(n0, Tp, lambda_C) # Electron collision time [s]
taui = spz.tau_i(n0, Tp, lambda_C) # Ion collision time [s]
print(f'FuZE-like Electron Collision Time = {taue:.3e} s')
print(f'FuZE-like Ion Collision Time = {taui:.3e} s')

kappa_perp_e = spz.KappaPerp_spitzer_e(n0, Tp, omega_ce_edge, taue, lambda_C)
kappa_perp_i = spz.KappaPerp_spitzer_i(n0, Tp, Z_h, omega_ci_edge, taui, lambda_C)
print(f'FuZE-like Perpendicular Spitzer Electron Thermal Conductivity = {kappa_perp_e:.3e} W/m/K')
print(f'FuZE-like Perpendicular Spitzer Ion Thermal Conductivity = {kappa_perp_i:.3e} W/m/K')

tau_E = cpfm.tauE(p0, uz0, rp, Tp, kappa_perp_e) # Energy confinement time [s]
tau_E_ion = cpfm.tauE(p0, uz0, rp, Tp, kappa_perp_i)
print(f'FuZE-like electron energy confinement time = {tau_E:.3e} s')
print(f'FuZE-like ion energy confinement time = {tau_E_ion:.3e} s')

tau_E_parabolic = cpfm.tauE_parabolic(p0, uz0, rp, Tp, kappa_perp_e) # parabolic Energy confinement time [s]
tau_E_ion_parabolic = cpfm.tauE_parabolic(p0, uz0, rp, Tp, kappa_perp_i)
print(f'FuZE-like electron parabolic energy confinement time = {tau_E_parabolic:.3e} s')
print(f'FuZE-like ion parabolic energy confinement time = {tau_E_ion_parabolic:.3e} s')

vple = 3 # cubic vortex power law exponent
ddfc = pb.DDFusionCalculator(vple)
Teff_e = ddfc.Teff_e(n0, rp, Tp, omega_ce_edge, taue, lambda_C)

print(f"The effective vortex temperature is {Teff_e * cnst.K_to_keV:.3e} keV")