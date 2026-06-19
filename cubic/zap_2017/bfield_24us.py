import sys
import pathlib
# ensure project root is on sys.path so the sibling `modules` package is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from modules import constants as cnst
from modules import cubic_pureflow_module as cpfm

from scipy.integrate import cumulative_trapezoid
'''
Calculate the magnetic field for Zap-HD values
'''
Tp = 1e3 * cnst.eV_to_K # Plasma temperature [K] 
n0 = 1e23 # m^-3
rp = 3e-3 # Pinch radius [m]

# t = 24us
u0 = 75e3 # Core flow velocity [m/s] 
uedge = 175e3 # Edge flow velocity [m/s]

num_r = 100
r = np.linspace(0, rp, num_r) # Radial grid for fitting

# Calculate the current density profile
# This depends on which pulse you are using
# u0 and uedge will need to be adjusted accordingly
uz0_roots = cpfm.root_solve_chi2_posbulk(uedge, u0, n0, rp, Tp)

uz_profiles = []
bfields = []
for root in uz0_roots:
    cbt = cpfm.cbt(n0, np.abs(root), rp, Tp)
    uz_profile = cpfm.uz_chi2cubic_posbulk(cbt, np.abs(root), u0, r)
    # Integrate the magnetic field
    bfield = cumulative_trapezoid(uz_profile, r, initial=0) * cnst.mu0 * n0 * cnst.q_e
    uz_profiles.append(uz_profile)
    bfields.append(bfield)

plt.figure()
for root, uz_profile in zip(uz0_roots, uz_profiles):
    plt.plot(r, uz_profile, label=f'uz0 = {root:.2e} m/s')

plt.figure()
for root, bfield in zip(uz0_roots, bfields):
    plt.plot(r, bfield, label=f'B-field for uz0 = {root:.2e} m/s')

plt.show()