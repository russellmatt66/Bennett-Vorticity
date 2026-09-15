import sys
import pathlib
# ensure project root is on sys.path so the sibling `modules` package is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np 
import matplotlib.pyplot as plt

import modules.cubic_pureflow_module as cpfm 

from modules import constants as cnst

n0 = 10**6 # [m^-3]
rB = 10**3 # [m]
TpB = 10**(-1) * cnst.eV_to_K # [eV] -> [K]

u_edgeB = 1.505e3 # [m/s]

uz0B = cpfm.root_solve_chi2_pure(u_edgeB, n0, rB, TpB)
CB_list = []
for uz0B_root in uz0B:
    CB = cpfm.cbt(n0, uz0B_root, rB, TpB)
    CB_list.append(CB)
print(f"CB_list = {CB_list} [m]")

rp = 1 # [m]
Tp = 10**(1) * cnst.eV_to_K # [eV] -> [K]

u_edge = 2.505e6 # [m/s]

uz0 = cpfm.root_solve_chi2_pure(u_edge, n0, rp, Tp)
C_list = []
for uz0_root in uz0:
    C = cpfm.cbt(n0, uz0_root, rp, Tp)
    C_list.append(C)
print(f"C_list = {C_list} [m]")

Lambda_list = []
for i in range(len(uz0)): # Roots must be same size
    Lambda = (np.real(uz0B[i])**2 + np.imag(uz0B[i])**2) / (np.real(uz0[i])**2 + np.imag(uz0[i])**2) * CB_list[i] / C_list[i]
    Lambda_list.append(Lambda)

phi_B = []
phi_C = []
for i in range(len(CB_list)):
    phi_B.append(rB/ CB_list[i])
    phi_C.append(rB/ C_list[i])

print(f"phi_B = {phi_B}")
print(f"phi_C = {phi_C}")
print(f"Lambda_list = {Lambda_list}")