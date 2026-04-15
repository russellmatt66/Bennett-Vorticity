"""
Investigating the behavior of a cubic pure-flow velocity and a bulk cubic plasma current for pollutant ion deposition

o Compute flow profile -> Compute magnetic field -> Compute waterfall drift speed

ACTUALLY, instead of doing this here, it's done in ../drifts/pureflow_air.py because that's where it was first done
"""
import sys
import pathlib
# ensure project root is on sys.path so the sibling `modules` package is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from modules import constants as cnst
# from modules import spitzer as spz
from modules import cubic_pureflow_module as cpfm
# from modules import plasma_properties as pp
# from modules import powerbalance as pb

import numpy as np
import matplotlib.pyplot as plt

n0 = 1e25 # Density of the air plasma; [m^{-3}]
Tp = 1000 # Temperature of the air plasma; [degK]
rp = 10e-3 # Characteristic radius of the plasma; [m]
uedge = 1 # Characteristic flow velocity; [m/s]

num_r = 100
r = np.linspace(0, rp, num_r)

uz0_roots = cpfm.root_solve_chi2_pure(uedge, n0, rp, Tp)

for uz0_root in uz0_roots:
    temp_cbt = cpfm.cbt(n0, np.abs(uz0_root), rp, Tp)

# cbt = cpfm.cbt(n0, , rp, Tp)

# uz = cpfm.uz_chi2cubic_pure(cbt, uz0, r)
# plt.plot(r, uz)

plt.show()