'''
Just plot the best solutions and then use them to create a beautiful figure showing the range over which these solutions vary next to the experimental data.
'''
import sys
import pathlib
# ensure project root is on sys.path so the sibling `modules` package is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from modules import constants as cnst
from modules import cubic_pureflow_module as cpfm

from sklearn.metrics import mean_squared_error

# Specify sweep & read in data
Tp_min = 200 * cnst.eV_to_K # Plasma temperature [K]; T = Te + Ti = 200 - 300 eV is experimental temperature of MAST pre-ELM
Tp_max = 300 * cnst.eV_to_K # Plasma temperature [K]; T = Te + Ti = 200 - 300 eV is experimental temperature of MAST pre-ELM

n0_min = 1e19 # m^-3
n0_max = 1e20 # m^-3

N_sweep = 25
n0_sweep = np.linspace(n0_min, n0_max, N_sweep) # Sweep over density
Tp_sweep = np.linspace(Tp_min, Tp_max, N_sweep) # Sweep over temperature

uz_df = pd.read_csv('../../experimental_data/mast_2023/Jtoroidal_MAST_preELM.csv')
uz_df.drop(columns=['Unnamed: 1'], inplace=True)
uz_df.sort_values(by='Radius (m)', inplace=True)

rminidx = uz_df['J_phi (MA / m^2)'][:uz_df['J_phi (MA / m^2)'].idxmax()].idxmin() # Index of minimum current density before max
rmin = uz_df['Radius (m)'][rminidx] # Radius at minimum current density

rmax = uz_df['Radius (m)'].max() # Maximum radius
rmaxidx = uz_df['Radius (m)'].idxmax() # Index of maximum radius

Jmaxidx = uz_df['J_phi (MA / m^2)'].idxmax() # Index of maximum current density
r_Jmax = uz_df['Radius (m)'][Jmaxidx] # R at maximum current density
Jzmax = uz_df['J_phi (MA / m^2)'][Jmaxidx]

num_r = 1000
r_wake = np.linspace(0.0, r_Jmax - rmin, num_r) # Radial positions for wake solution, from 0 to R at max current density
rp_wake = r_Jmax - rmin # Pinch radius [m]; Tied to dataset for better fidelity   

r_front = np.linspace(r_Jmax - rmin, rmax - rmin, num_r) # Radial positions for plotting front solution, from R at max current density to max radius
rp_front = np.abs(rmax - r_Jmax) # Pinch radius [m]; Tied to dataset for better fidelity
r_front_calc = np.linspace(0.0, rp_front, num_r) # Radial positions for calculating front solution, from 0 to rp

# Solve problem & accumulate the naturally best solutions
# Wake solve 
all_wake_solns = []
all_wake_cbts = []
for ns in range(N_sweep):
    n0 = n0_sweep[ns]
    Tp = Tp_sweep[ns]
    u0_wake = 1e6 * uz_df['J_phi (MA / m^2)'][rminidx] / (cnst.q_e * n0) # Core flow velocity [m/s]; J = n e u => u = J / (n e)
    uedge_wake = 1e6 * uz_df['J_phi (MA / m^2)'][Jmaxidx] / (cnst.q_e * n0) # Edge flow velocity [m/s]; J = n e u => u = J / (n e)
    uz0_roots = cpfm.root_solve_chi2_posbulk(uedge_wake, u0_wake, n0, rp_wake, Tp)
    wake_solns = []
    wake_cbts = []
    for uz0_root in uz0_roots:
        uz0 = np.abs(uz0_root) # Take magnitude
        cbt = cpfm.cbt(n0, uz0, rp_wake, Tp) # Vortex constant [m]
        print(f'cbt for uz0 = {uz0:2e} m/s: {cbt} m')
        wake_cbts.append(cbt)
        uz_fit = cpfm.uz_chi2cubic_posbulk(cbt, uz0, u0_wake, r_wake)
        wake_solns.append(uz_fit)
    all_wake_solns.append(wake_solns)
    all_wake_cbts.append(wake_cbts)

# Front solve
all_front_solns = []
all_front_cbts = []
for ns in range(N_sweep):
    n0 = n0_sweep[ns]
    Tp = Tp_sweep[ns]
    u0_front = 1e6 * Jzmax / (cnst.q_e * n0) # Core flow velocity [m/s]; J = n e u => u = J / (n e)
    uedge_front = 1e6 * uz_df['J_phi (MA / m^2)'][rmaxidx] / (cnst.q_e * n0) # Edge flow velocity [m/s]; J = n e u => u = J / (n e)
    uz0_roots = cpfm.root_solve_chi2_posbulk(uedge_front, u0_front, n0, rp_front, Tp)
    front_solns = []
    front_cbts = []
    for uz0_root in uz0_roots:
        uz0 = np.abs(uz0_root) # Take magnitude
        cbt = cpfm.cbt(n0, uz0, rp_front, Tp) # Vortex constant [m]
        print(f'cbt for uz0 = {uz0:2e} m/s: {cbt} m')
        front_cbts.append(cbt)
        uz_fit = cpfm.uz_chi2cubic_posbulk(cbt, uz0, u0_front, r_front_calc)
        front_solns.append(uz_fit)
    all_front_solns.append(front_solns)
    all_front_cbts.append(front_cbts)

# Accumulate the naturally best solutions
best_wake_solns = []
best_front_solns = []

best_wake_rrmses = []
best_front_rrmses = []

uz_wake = uz_df['J_phi (MA / m^2)'][rminidx:Jmaxidx] / (cnst.q_e * n0) * 1e6 # Convert J to uz for wake region
for ns in range(N_sweep):
    wake_solns_temp = all_wake_solns[ns]
    rrmse_min = np.inf
    for wake_soln in wake_solns_temp:
        uz_soln_interp = np.interp(uz_df['Radius (m)'][rminidx:Jmaxidx] - rmin, r_wake, wake_soln) # Interpolate fit to data points
        rmse = np.sqrt(mean_squared_error(uz_wake, uz_soln_interp))
        rrmse = rmse / (uz_wake.max() - uz_wake.min()) # Normalize by range of data
        if rrmse < rrmse_min:
            rrmse_min = rrmse
            best_wake_soln = wake_soln
    best_wake_solns.append(best_wake_soln)
    best_wake_rrmses.append(rrmse_min)

uz_front = uz_df['J_phi (MA / m^2)'][Jmaxidx:rmaxidx] / (cnst.q_e * n0) * 1e6 # Convert J to uz for front region
for ns in range(N_sweep):
    front_solns_temp = all_front_solns[ns]
    rrmse_min = np.inf
    for front_soln in front_solns_temp:
        uz_soln_interp = np.interp(uz_df['Radius (m)'][Jmaxidx:rmaxidx] - rmin, r_front, front_soln) # Interpolate fit to data points
        rmse = np.sqrt(mean_squared_error(uz_front, uz_soln_interp))
        rrmse = rmse / (uz_front.max() - uz_front.min()) # Normalize by range of data
        if rrmse < rrmse_min:
            rrmse_min = rrmse
            best_front_soln = front_soln
    best_front_solns.append(best_front_soln)
    best_front_rrmses.append(rrmse_min)

print('Best solution obtainment halted')

# Create band plots showing the range over which these best solutions vary next to the experimental data
