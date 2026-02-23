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
from modules import spitzer as spz
# from spitzer import coulombLog_ei as lambda_C 

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

num_r = 32000
rinterior = 1e-6 # Avoid singularity at r=0 for calculating drifts
r = np.linspace(rinterior, rp, num_r)
T = Tp / rp**3 * r**3 # Cubic

lambda_C = spz.coulombLog_ei(n0, Tp, qj) # Coulomb logarithm for air plasma (assuming Z=1)

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

vpeak_resistive = []
vpeak_wf = []
vpeak_shell = []
vpeak_EB = []
vpeak_gradB = []

vmin_resistive = []
vmin_wf = []
vmin_shell = []
vmin_EB = []
vmin_gradB = []

for uz0 in uz0_roots:
    cbt = cpfm.cbt(n0, np.abs(uz0), rp, Tp) # Vortex constant [m]
    uz = cpfm.uz_chi2cubic_pure(cbt, np.abs(uz0), r)
    btheta = cpfm.btheta_chi2(cbt, np.abs(uz0), n0, r)
    lapU = cpfm.LapU_chi2cubic_pure(cbt, np.abs(uz0), r)
    gradB = cpfm.gradbtheta_chi2cubic_pure(cbt, np.abs(uz0), n0, r)
    uzs.append(uz)
    cbts.append(cbt)
    Bthetas.append(btheta)
    # radial drifts
    vg_wf.append(drifts.gravitational_drift_waterfall(mj, qj, g0, btheta))
    # v_resistive.append(drifts.resistive_drift(qj, n0, uz, btheta, T, lambda_C))
    v_resistive.append(drifts.resistive_drift(qj, n0, uz, btheta, T, lambda_C))
    # axial drifts
    vg_shell.append(drifts.gravitational_drift_shell(mj, qj, g0, L, btheta))
    # vg_hd.append(drifts.gravitational_drift_hotdog(mj, qj, g0, btheta)) # Depends on theta, so calculate later
    v_EB.append(drifts.ExB_drift(mj, T, btheta, qj, uz, lapU)) # Depends on LapU, so calculate later
    v_gradB.append(drifts.gradB_drift(mj, qj, btheta, gradB, uz)) # Depends on gradB, so calculate later
    vpeak_resistive.append(np.max(v_resistive[-1]))
    vpeak_wf.append(np.max(vg_wf[-1]))
    vpeak_shell.append(np.max(vg_shell[-1]))
    vpeak_EB.append(np.max(v_EB[-1]))
    vpeak_gradB.append(np.max(v_gradB[-1]))
    vmin_resistive.append(np.min(v_resistive[-1]))
    vmin_wf.append(np.min(vg_wf[-1]))
    vmin_shell.append(np.min(vg_shell[-1]))
    vmin_EB.append(np.min(v_EB[-1]))
    vmin_gradB.append(np.min(v_gradB[-1]))

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
    # drift_ax.plot(r*1e3, uzs[i], label=f'Diamagnetic drift (velocity profile)') # meters to millimeters
    # drift_ax.plot(r*1e3, vg_wf[i], label=f'Waterfall gravitational (radial) drift')
    # drift_ax.plot(r*1e3, v_resistive[i], label=f'Resistive (radial) drift')
    # drift_ax.plot(r*1e3, vg_shell[i], label=f'Shell gravitational (axial) drift')
    # drift_ax.plot(r*1e3, v_gradB[i], label=f'Grad-B (axial) drift')
    # drift_ax.plot(r*1e3, v_EB[i], label=f'ExB (axial) drift')
    drift_ax.loglog(r*1e3, uzs[i], label=f'Diamagnetic drift (velocity profile)') # meters to millimeters
    drift_ax.loglog(r*1e3, vg_wf[i], label=f'Waterfall gravitational (radial) drift')
    drift_ax.loglog(r*1e3, v_resistive[i], label=f'Resistive (radial) drift')
    drift_ax.loglog(r*1e3, vg_shell[i], label=f'Shell gravitational (axial) drift')
    drift_ax.loglog(r*1e3, v_gradB[i], label=f'Grad-B (axial) drift')
    drift_ax.loglog(r*1e3, v_EB[i], label=f'ExB (axial) drift')
    # drift_ax.semilogy(r*1e3, uzs[i], label=f'Diamagnetic drift (velocity profile)') # meters to millimeters
    # drift_ax.semilogy(r*1e3, vg_wf[i], label=f'Waterfall gravitational (radial) drift')
    # drift_ax.semilogy(r*1e3, v_resistive[i], label=f'Resistive (radial) drift')
    # drift_ax.semilogy(r*1e3, vg_shell[i], label=f'Shell gravitational (axial) drift')
    drift_ax.set_title(f'Plasma drifts for $r_{{p}}={rp:.2f}$ m, L = {L:.2f} m, n0 = {n0:.2e} m^-3, Tp = {Tp:.2f} K, uz0 = {np.abs(uz0):.2f} m/s')
    drift_ax.set_xlabel('r (mm)')
    drift_ax.set_ylabel('Drift Velocity (m/s)')
    min_speed = min(vmin_resistive[i], vmin_wf[i], vmin_shell[i], vmin_EB[i], vmin_gradB[i])
    max_speed = max(vpeak_resistive[i], vpeak_wf[i], vpeak_shell[i], vpeak_EB[i], vpeak_gradB[i])
    drift_ax.set_ylim(2 * min_speed, 2 * max_speed)
    drift_ax.set_yscale('symlog', linthresh=1e-3) 
    # drift_ax.set_ylim(1e-20, 1e5)
    # plt.plot(r, vg_hd[i], label=f'Hotdog gravitational (axial) drift')
    # plt.plot(r, v_EB[i], label=f'ExB (axial) drift')
    plt.legend()
    figs_and_axs.append((flow_fig, axs))

print(f'Peak resistive drift speeds: {vpeak_resistive} m/s')
print(f'Peak waterfall gravitational drift speed: {vpeak_wf} m/s')
print(f'Peak shell gravitational drift speed: {vpeak_shell} m/s')
print(f'Minimum resistive drift speeds: {vmin_resistive} m/s')
print(f'Minimum waterfall gravitational drift speed: {vmin_wf} m/s')
print(f'Minimum shell gravitational drift speed: {vmin_shell} m/s')
# print(f'Peak diamagnetic drift speed (flow velocity): {np.max(uzs):.2e} m/s')

plt.show()