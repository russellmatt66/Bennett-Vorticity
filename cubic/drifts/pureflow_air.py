"""
Plot the plasma drifts for a pureflow cubic air vortex 
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
L = 1 # Pinch length [m]

num_r = 100
rinterior = 1e-9 # Avoid singularity at r=0 for calculating drifts
r = np.linspace(rinterior, rp, num_r)

uz0_roots = cpfm.root_solve_chi2_pure(uedge, n0, rp, Tp)
# Get largest real and imaginary parts
uz0_Remax = np.max(np.real(uz0_roots))
uz0_Immax = np.max(np.imag(uz0_roots))
print(f"Max Real: {uz0_Remax:.2e}, Max Imag: {uz0_Immax:.2e}")

uzs = []

cbts = []
Bthetas = []

# radial drifts
vg_wf = [] # waterfall gravitational drifts
v_resistive = []

# axial drifts
vg_shell = []
vg_hd = []
v_EB = []
# v_D = [] # This is the same as the velocity profile 
v_gradB = []    

for uz0 in uz0_roots:
    cbt = cpfm.cbt(n0, np.abs(uz0), rp, Tp) # Vortex constant [m]
    uz = cpfm.uz_chi2cubic_pure(cbt, np.abs(uz0), r)
    btheta = cpfm.btheta_chi2(cbt, np.abs(uz0), n0, r)
    uzs.append(uz)
    cbts.append(cbt)
    Bthetas.append(btheta)
    # radial drifts
    vg_wf.append(drifts.gravitational_drift_waterfall(mj, qj, g0, btheta))
    v_resistive.append(drifts.resistive_drift(qj, n0, uz, btheta))
    # axial drifts
    vg_shell.append(drifts.gravitational_drift_shell(mj, qj, g0, L, btheta))
    # vg_hd.append(drifts.gravitational_drift_hotdog(mj, qj, g0, btheta)) # Depends on theta, so calculate later
    # v_EB.append(drifts.ExB_drift(uz, btheta)) # Depends on LapU, so calculate later

# Plot the plasma drifts and flow profile for each root on the same plot
figs_and_axs = []

for i, uz0 in enumerate(uz0_roots):
    flow_fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    flow_ax = axs[0]
    drift_ax = axs[1]
    flow_ax.plot(uz0.real, uz0.imag, 'o')
    flow_ax.set_title(f'uz0 = {uz0.real:.2e} + {uz0.imag:.2e}j')
    flow_ax.set_xlabel('Real(uz0)')
    flow_ax.set_ylabel('Imag(uz0)')
    flow_ax.set_xlim(-2 * uz0_Remax, 2 * uz0_Remax)
    flow_ax.set_ylim(-2 * uz0_Immax, 2 * uz0_Immax)
    drift_ax.plot(r, uzs[i], label=f'Diamagnetic drift (velocity profile)')
    drift_ax.plot(r, vg_wf[i], label=f'Waterfall gravitational (radial) drift')
    drift_ax.plot(r, v_resistive[i], label=f'Resistive (radial) drift')
    drift_ax.plot(r, vg_shell[i], label=f'Shell gravitational (axial) drift')
    # plt.plot(r, vg_hd[i], label=f'Hotdog gravitational (axial) drift')
    # plt.plot(r, v_EB[i], label=f'ExB (axial) drift')
    plt.legend()
    figs_and_axs.append((flow_fig, axs))

plt.show()