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

import numpy as np
import matplotlib.pyplot as plt

n0 = 1e22 # Plasma density [m^-3]; 1e22 - 1e23
rp = 10e-3 # Pinch radius [m]
Tp = 200 * cnst.eV_to_K # Plasma temperature [K]; T = Te + Ti = 150 - 200 eV

u0 = 50e3 # Representative core flow velocity [m/s]

num_r = 400
r = np.linspace(0, 2.0 * rp, num_r) # Radial positions for plotting [m]

n_vortices = 2
chunk_size = r.size // n_vortices

r_chunks = [r[i:i + chunk_size] for i in range(0, r.size, chunk_size)]

u0_list = [u0, 2 * u0] # BCS: [even, odd]
uedge_list = [2 * u0, u0] # BCs: [even, odd]

num_roots = 4 # cubic, pureflow vortices have this many always 
root_tracks = [[] for _ in range(num_roots)]
cbt_tracks = [[] for _ in range(num_roots)]

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

    else: # Odd index: negative bulk vortex
        uz0_roots = cpfm.root_solve_chi2_negbulk(u0_list[1], uedge_list[1], n0, rp, Tp)
        for ridx, uz0 in enumerate(uz0_roots):
            cbt = cpfm.cbt(n0, np.abs(uz0), rp, Tp)
            uz_chunk = cpfm.uz_chi2cubic_negbulk(cbt, np.abs(uz0), u0_list[1], r_chunk) # ***THIS NEEDS TO BE SHIFTED***
            root_tracks[ridx].append(uz_chunk)
            cbt_tracks[ridx].append(cbt)

# uz = np.concatenate(uz_list)
uzs = []

for track in root_tracks:
    uzs.append(np.concatenate(track))

for uz in uzs:
    plt.plot(r, uz)
    plt.xlabel("Radial position [m]")
    plt.ylabel("Axial velocity [m/s]")
    plt.title("Axial velocity profile")
# plt.plot(r, uz)
# plt.xlabel("Radial position [m]")
# plt.ylabel("Axial velocity [m/s]")
# plt.title("Axial velocity profile")
plt.show()