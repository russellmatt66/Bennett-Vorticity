import sys
import pathlib
# ensure project root is on sys.path so the sibling `modules` package is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np 
import matplotlib.pyplot as plt

import modules.cubic_pureflow_module as cpfm 
'''
This file plots the core pressure of a pure-flow cubic Bennett vortex

uz(r) = uz0 r^2 / (r + C)^2, C \simeq 10^{-22}n0 * uz0^2 * rp ^3 * Tp^-1 [m]

p0 = p(r = 0) = p(0)
'''
n0 = 1e20  # [m^-3] uniform number density
rp = 1e-3 # [m] pinch radius
Tp = 1e3 # [K] edge temperature

u0 = 1e6 # [m/s] u(0)
u_edge = 2e6 # [m/s] u(rp)

uz0 = cpfm.root_solve_chi2_posbulk(u_edge, u0, n0, rp, Tp)

C_list = []
p0_list = []
for uz0_root in uz0:
    uz0_abs = abs(uz0_root)
    C = cpfm.cbt(n0, uz0_abs, rp, Tp)
    print("uz0: ", uz0_root, "uz0_abs:", uz0_abs, "C:", C)
    p0 = cpfm.p0_posbulk(C, n0, uz0_abs, u0, rp)
    C_list.append(C)
    p0_list.append(p0)

print("C_list:", C_list)
print("p0_list:", p0_list)