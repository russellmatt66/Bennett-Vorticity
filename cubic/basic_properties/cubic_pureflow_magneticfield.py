import sys
import pathlib
# ensure project root is on sys.path so the sibling `modules` package is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from modules import constants as cnst
from modules import cubic_pureflow_module as cpfm
'''
Plot the normalized azimuthal magnetic field profile of an r^3 Bennett Vortex.

Why does this file not use the library? 
Because it's copied from here: https://github.com/russellmatt66/imhd-CUDA/blob/main/python/bennett-vorticity/temp-rcubed/temp-rcubed_btheta.py
'''
rp = 10**6  # [m]
n0 = 1e12 # [m^{-3}]
Tp = 10 * cnst.eV_to_K # [eV]


uedge = 1.25e6 # m/s
uz0 = cpfm.root_solve_chi2_pure(uedge, n0, rp, Tp)

C = cpfm.cbt(n0, uz0, rp, Tp)

print(f"C_{{B,T}} = {C} m")

phi = rp / C 
# phi = np.logspace(-6, 1, 500) # r / C_{B,T}

btheta_tilde = (1.0) / (phi * (phi + 1)) * (phi**3 - 3*phi**2 - 6*phi + 6*phi* np.log(1 + phi) + 6*np.log(phi + 1))  # normalized B_theta profile - btheta_tilde = B_theta / B_theta_max
print(f"btheta_tilde = {btheta_tilde} [T]")

btheta_tilde_sum = np.sum(btheta_tilde) 
answer = btheta_tilde_sum / 4.0
print(f"Answer  = {answer:.4e}")

# plt.plot(phi, btheta_tilde)
# plt.xlabel('$\\phi = r / C_{B,T}$')
# plt.ylabel('$\\tilde{B}_{\\theta} = \\frac{B_{\\theta}}{\\mu_{0}e n_{0}u_{z,0}C_{B,T}}$')
# plt.title('Normalized Azimuthal Magnetic Field Profile of an $r^{3}$ Bennett Vortex')
# plt.xscale('log')
# plt.grid(True)

# plt.show()