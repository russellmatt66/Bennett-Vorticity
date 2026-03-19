"""
Build a chain of pureflow to negbulk vortices that have sufficient saturation span to bring the magnetic field back to zero 
in the inter-vortice regions.
"""
import sys
import pathlib
# ensure project root is on sys.path so the sibling `modules` package is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from modules import cubic_pureflow_module as cpfm
from modules import constants as cnst

from scipy.integrate import cumulative_trapezoid

import numpy as np
import matplotlib.pyplot as plt

n0 = 1e26 # Plasma density [m^-3]; 1e22 - 1e23
rp = 10e-6 # Pinch radius [m]
Tp = 1e2 * cnst.eV_to_K # Plasma temperature [K]; T = Te + Ti = 150 - 200 eV

u0 = 10e3 # Representative core flow velocity [m/s]
u02 = 50e3 # Representative core flow velocity for second vortex - edge state for first [m/s]

num_r = 400
r = np.linspace(0, 2.0 * rp, num_r) # Radial positions for plotting [m]

n_vortices = 2
chunk_size = r.size // n_vortices

r_chunks = [r[i:i + chunk_size] for i in range(0, r.size, chunk_size)]

u0_list = [u0, u02] # BCS: [even, odd]
uedge_list = [u02, u0] # BCs: [even, odd]

num_roots = 4 # cubic, pureflow vortices have this many always 
root_tracks = [[] for _ in range(num_roots)]
cbt_tracks = [[] for _ in range(num_roots)]
uz0_tracks = [[] for _ in range(num_roots)]

# Solve the equations
for i, r_chunk in enumerate(r_chunks):
    print(f'Processing chunk {i} of {n_vortices}...')
    if i % 2 == 0: # Even index: pureflow positive bulk vortex
        uz0_roots = cpfm.root_solve_chi2_posbulk(uedge_list[0], u0_list[0], n0, rp, Tp)
        for ridx, uz0 in enumerate(uz0_roots):
            cbt = cpfm.cbt(n0, np.abs(uz0), rp, Tp)
            uz_chunk = cpfm.uz_chi2cubic_posbulk(cbt, np.abs(uz0), u0_list[0], r_chunk)
            root_tracks[ridx].append(uz_chunk)
            cbt_tracks[ridx].append(cbt)
            uz0_tracks[ridx].append(uz0)

    else: # Odd index: negative bulk vortex
        uz0_roots = cpfm.root_solve_chi2_negbulk(u0_list[1], uedge_list[1], n0, rp, Tp)
        for ridx, uz0 in enumerate(uz0_roots):
            cbt = cpfm.cbt(n0, np.abs(uz0), rp, Tp)
            uz_chunk = cpfm.uz_chi2cubic_negbulk(cbt, np.abs(uz0), u0_list[1], r_chunks[0]) 
            root_tracks[ridx].append(uz_chunk)
            cbt_tracks[ridx].append(cbt)
            uz0_tracks[ridx].append(uz0)


# uz = np.concatenate(uz_list)
uzs = []

for track in root_tracks:
    uzs.append(np.concatenate(track))

# Calculate magnetic fields
def solve_bfield(uz: np.ndarray, r: np.ndarray) -> np.ndarray:
    J = n0 * cnst.q_e * uz # Current density [A/m^2]
    Iencl = 2 * np.pi * cumulative_trapezoid(J * r, r, initial=0) # Enclosed current [A]
    B = cnst.mu0 * Iencl / (2 * np.pi * r) # Magnetic field [T]
    # B = np.insert(B, 0, 0) # Insert B=0 at r=0 for plotting
    B[0] = 0 # Set B=0 at r=0 to avoid singularity for plotting
    return B

# Plot
plt.figure()
for i, uz in enumerate(uzs):
    plt.plot(r, uz, label=f'Root {i+1}, cbt = {cbt_tracks[i][0]:.3e} m, uz0 = {uz0_tracks[i][0]:.3e} m/s')
    plt.xlabel("Radial position [m]")
    plt.ylabel("Axial velocity [m/s]")
    plt.ylim(0.0, 1.1 * uz.max())
    plt.yticks(np.linspace(0, uz.max(), 11))
    plt.title(f"Vortex sawtooth, n0 = {n0:.1e} m^-3, Tp = {Tp / cnst.eV_to_K:.1f} eV, u0 = {u0:.1e} m/s")
plt.legend()

pad_points = 3 * num_r
dr = r[1] - r[0] # uniformly spaced
r_pad = np.linspace(r[-1] + dr, r[-1] + dr * pad_points, pad_points)
# r_pad = np.arange(r[-1] + dr, r[-1] + dr * (pad_points + 1), dr)
# r_pad = np.arange(r[-1] + dr, r[-1] + dr * (pad_points), dr)

plt.figure()
uzs_ext = []
r_exts = []
for i, uz in enumerate(uzs):
    uz_ext = np.concatenate((uz, np.zeros(pad_points))) # Pad uz with zeros for the extended r range
    r_ext = np.concatenate((r, r_pad)) # Extended r range for plotting
    uzs_ext.append(uz_ext)
    r_exts.append(r_ext)
    B = solve_bfield(uz_ext, r_ext)
    plt.plot(r_ext, B, label=f'Root {i+1}, cbt = {cbt_tracks[i][0]:.3e} m, uz0 = {uz0_tracks[i][0]:.3e} m/s')
    plt.xlabel("Radial position [m]")
    plt.ylabel("Magnetic field [T]")
    plt.title(f"Vortex sawtooth magnetic field, n0 = {n0:.1e} m^-3, Tp = {Tp / cnst.eV_to_K:.1f} eV, u0 = {u0:.1e} m/s")
plt.legend()

plt.figure()
for i, uz_ext in enumerate(uzs_ext):
    plt.plot(r_exts[i], uz_ext, label=f'Root {i+1}, cbt = {cbt_tracks[i][0]:.3e} m, uz0 = {uz0_tracks[i][0]:.3e} m/s')
    plt.xlabel("Radial position [m]")
    plt.ylabel("Axial velocity [m/s]")
    plt.title(f"Extended vortex sawtooth, n0 = {n0:.1e} m^-3, Tp = {Tp / cnst.eV_to_K:.1f} eV, u0 = {u0:.1e} m/s")
plt.legend()

plt.show()