import sys
import pathlib
# ensure project root is on sys.path so the sibling `modules` package is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np 
import matplotlib.pyplot as plt

import modules.cubic_pureflow_module as cpfm 
import modules.spitzer as spitzer
import modules.plasma_properties as pp
import modules.constants as cnst

n0 = 1e14  # [m^-3] uniform number density
rp = 1e-2 # [m] pinch radius
Tp = 0.5 * cnst.eV_to_K # [K] edge temperature

u_edge = 1e5 # [m/s] u(rp)

uz0 = cpfm.root_solve_chi2_pure(u_edge, n0, rp, Tp)

C_list = []
p0_list = []
for uz0_root in uz0:
    uz0_abs = abs(uz0_root) # Speed profile means we can handle complex velocities
    C = cpfm.cbt(n0, uz0_abs, rp, Tp)
    print("uz0: ", uz0_root, "uz0_abs:", uz0_abs, "C:", C)
    p0 = cpfm.p0(C, n0, uz0_abs, rp)
    C_list.append(C)
    p0_list.append(p0)

print("C_list [m]:", C_list)
print("p0_list [Pa]:", p0_list)

tauE_list = []
Brp_list = []
for i in range(len(C_list)):
    Brp = cpfm.btheta_chi2(C_list[i], np.abs(uz0[i]), n0, rp)
    omega_ci = pp.omega_ci(Brp, Z=1, mj=cnst.mH)
    lambda_C = spitzer.coulombLog_ei(n0, Tp, Z=1)
    tau_i = spitzer.tau_i(n0, Tp, lambda_C=lambda_C)
    kappa_perp = spitzer.KappaPerp_spitzer_i(n0, Tp, Z=1, omega_ci=omega_ci, taui=tau_i, lambda_C=lambda_C)
    tauE = cpfm.tauE(p0_list[i], np.abs(uz0[i]), rp, Tp, kappa_perp=kappa_perp)
    Brp_list.append(Brp)
    tauE_list.append(tauE)

print("tauE_list [s]:", tauE_list)
print("Brp_list [T]:", Brp_list)