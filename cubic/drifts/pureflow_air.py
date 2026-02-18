"""
Plot the plasma drifts for a pureflow cubic vortex 
"""
import sys
import pathlib
# `pureflow.py` is in `cubic/drifts/`; repository root is 2 directories above that.
# Path.parents is 0-indexed: [0]=drifts, [1]=cubic, [2]=repo root.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from modules import constants as cnst
from modules import cubic_pureflow_module as cpfm
from modules import plasma_properties as pp
from modules import drifts 

import numpy as np
import matplotlib.pyplot as plt

###
mj = cnst.mair # Ion mass [kg]
qj = cnst.q_e # Ion charge [C]
g0 = 9.81 # Gravitational acceleration on the surface of the earth [m/s^2]

n0 = 1e25 # Plasma density [m^-3]
Tp = 1e3 # Plasma temperature [K]
uedge = 1 # Edge flow velocity [m/s]
rp = 10e-3 # Pinch radius [m]

num_r = 100
r = np.linspace(0, rp, num_r)

uz0_roots = cpfm.root_solve_chi2_pure(uedge, n0, rp, Tp)

cbts = []
Bthetas = []

# radial drifts
vg_wf = []
v_resistive = []

# axial drifts
vg_shell = []
vg_hd = []
v_EB = []
v_D = []
v_gradB = []

for uz0 in uz0_roots:
    cbt = cpfm.cbt(n0, np.abs(uz0), rp, Tp) # Vortex constant [m]
    cbts.append(cbt)
    Bthetas.append(cpfm.btheta_chi2(cbt, np.abs(uz0), n0, r))


plt.figure()

plt.show()