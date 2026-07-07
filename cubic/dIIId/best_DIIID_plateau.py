'''
Just plot the best solutions and then use them to create a beautiful figure showing the range over which these solutions vary next to the experimental data.

DIIID has negative current density in both regions indicating that a solution needs to be chained together.

It also looks like the broad flat region of the current density profile is not well captured by it, which is expected because this is local and describes a large shear that suddenly onsets
Therefore, we should investigate the possibility of using a different model for the broad flat region, with the cubic model being used for the shear region.

This code doesn't fix the connection point at Jmax. Instead it raster scans over the entire plateau region.
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
Tp_min = 200 * cnst.eV_to_K # Plasma temperature [K]; 
Tp_max = 700 * cnst.eV_to_K # Plasma temperature [K]; 

n0_min = 1e19 # m^-3
n0_max = 1e20 # m^-3

N_sweep = 25 # Number of points in the sweep for both n0 and Tp
n0_sweep = np.linspace(n0_min, n0_max, N_sweep) # Sweep over density
Tp_sweep = np.linspace(Tp_min, Tp_max, N_sweep) # Sweep over temperature

uz_df = pd.read_csv('../../experimental_data/DIIID/DIIID_2005_positive.csv')
# uz_df.drop(columns=['Unnamed: 1'], inplace=True)
uz_df.columns = uz_df.columns.str.strip()
uz_df.sort_values(by='R (m)', inplace=True)

rminidx = uz_df['J (MA / m^2)'][:uz_df['J (MA / m^2)'].idxmax()].idxmin() # Index of minimum current density before max
rmin = uz_df['R (m)'][rminidx] # Radius at minimum current density

rmax = uz_df['R (m)'].max() # Maximum radius
rmaxidx = uz_df['R (m)'].idxmax() # Index of maximum radius

r_plateau = []
r_plateau_idx = []

Jmax_threshold = 0.9 * uz_df['J (MA / m^2)'].max() # Threshold for considering a point as part of the plateau region
Jmaxidx = uz_df['J (MA / m^2)'].idxmax() # Index of maximum current density
for idx, Jz in uz_df['J (MA / m^2)'].items(): # INEFFICIENT O(N_EXP)
    if Jz >= Jmax_threshold:
        r_plateau.append(uz_df['R (m)'][idx]) # Append the radius corresponding to the current density value
        r_plateau_idx.append(idx) # Append the index corresponding to the current density value

print(f'Plateau region starts at radius: {r_plateau[0]} m')
print(f'Plateau region ends at radius: {r_plateau[-1]} m')

def sweep(r_plateau: float, r_plateau_idx: int):
    r_Jmax = r_plateau
    Jzmax = uz_df['J (MA / m^2)'][r_plateau_idx]
    Jmaxidx = r_plateau_idx

    num_r = 10000
    r_wake = np.linspace(0.0, r_Jmax - rmin, num_r) # Radial positions for wake solution, from 0 to R at max current density
    rp_wake = r_Jmax - rmin # Pinch radius [m]; Tied to dataset for better fidelity   

    r_front = np.linspace(r_Jmax - rmin, rmax - rmin, num_r) # Radial positions for plotting front solution, from R at max current density to max radius
    rp_front = np.abs(rmax - r_Jmax) # Pinch radius [m]; Tied to dataset for better fidelity
    r_front_calc = np.linspace(0.0, rp_front, num_r) # Radial positions for calculating front solution, from 0 to rp

    all_wake_solns = []
    all_wake_cbts = []
    for ns in range(N_sweep):
        for ts in range(N_sweep):
            n0 = n0_sweep[ns]
            Tp = Tp_sweep[ts]
            u0_wake = 1e6 * uz_df['J (MA / m^2)'][rminidx] / (cnst.q_e * n0) # Core flow velocity [m/s]; J = n e u => u = J / (n e)
            uedge_wake = 1e6 * uz_df['J (MA / m^2)'][Jmaxidx] / (cnst.q_e * n0) # Edge flow velocity [m/s]; J = n e u => u = J / (n e)
            uz0_roots = cpfm.root_solve_chi2_posbulk(uedge_wake, u0_wake, n0, rp_wake, Tp)
            wake_solns = []
            wake_cbts = []
            for uz0_root in uz0_roots:
                if np.abs(uz0_root.imag) > 1e-6: # Skip complex roots, they lie outside the ideal framework and don't yield real solutions
                    continue
                uz0 = np.abs(uz0_root) # Take magnitude bc cbt is the same, it doesn't impact real solns, and complex solns lie outside the ideal framework anyway
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
        for ts in range(N_sweep):
            n0 = n0_sweep[ns]
            Tp = Tp_sweep[ts]
            u0_front = 1e6 * Jzmax / (cnst.q_e * n0) # Core flow velocity [m/s]; J = n e u => u = J / (n e)
            uedge_front = 1e6 * uz_df['J (MA / m^2)'][rmaxidx] / (cnst.q_e * n0) # Edge flow velocity [m/s]; J = n e u => u = J / (n e)
            uz0_roots = cpfm.root_solve_chi2_negbulk(uedge_front, u0_front, n0, rp_front, Tp)
            front_solns = []
            front_cbts = []
            for uz0_root in uz0_roots:
                uz0 = np.abs(uz0_root) # Take magnitude
                cbt = cpfm.cbt(n0, uz0, rp_front, Tp) # Vortex constant [m]
                print(f'cbt for uz0 = {uz0:2e} m/s: {cbt} m')
                front_cbts.append(cbt)
                uz_fit = cpfm.uz_chi2cubic_negbulk(cbt, uz0, u0_front, r_front_calc)
                front_solns.append(uz_fit)
            all_front_solns.append(front_solns)
            all_front_cbts.append(front_cbts)

    return all_wake_solns, all_wake_cbts, all_front_solns, all_front_cbts, r_front, r_wake, rp_front, rp_wake

def accumulate_best_solutions(all_wake_solns, all_wake_cbts, all_front_solns, all_front_cbts, r_plateau_val, r_plateau_idx_val):
    best_wake_solns = []
    best_front_solns = []

    best_wake_cbts = []
    best_front_cbts = []

    best_wake_rrmses = []
    best_front_rrmses = []

    best_wake_n0s = [] # Stale density bug occurs downstream from existence of double complex conjugate pairs 
    best_front_n0s = []

    tolerance = 1e-4
    endpoint_rtol = 1e-3
    Jmaxidx = r_plateau_idx_val
    for nss in range(N_sweep * N_sweep): # Loop over all combinations of n0 and Tp
        n0 = n0_sweep[nss // N_sweep]
        uz_wake = uz_df['J (MA / m^2)'].iloc[rminidx:Jmaxidx + 1] / (cnst.q_e * n0) * 1e6 # Convert J to uz for wake region (include both boundaries)
        wake_solns_temp = all_wake_solns[nss]
        rrmse_min = np.inf
        best_wake_soln = None
        best_wake_cbt = None
        for iw, wake_soln in enumerate(wake_solns_temp):
            uz_soln_interp = np.interp(uz_df['R (m)'].iloc[rminidx:Jmaxidx + 1] - rmin, r_wake, wake_soln) # Interpolate fit to data points
            rmse = np.sqrt(mean_squared_error(uz_wake, uz_soln_interp))
            rrmse = rmse / (uz_wake.max() - uz_wake.min()) # Normalize by range of data
            if not np.isclose(wake_soln[-1], uz_wake.iloc[-1], rtol=endpoint_rtol, atol=tolerance): # Check if the last point matches within tolerance
                continue # Skip nonphysical solutions
            if not np.isclose(wake_soln[0], uz_wake.iloc[0], rtol=endpoint_rtol, atol=tolerance): # Check if the first point matches within tolerance
                continue # Skip nonphysical solutions
            if rrmse < rrmse_min:
                rrmse_min = rrmse
                best_wake_soln = wake_soln
                best_wake_cbt = all_wake_cbts[nss][iw]
        if best_wake_soln is None: # No valid solution found, skip
            continue # Causes stale density bug if double complex conjugate pairs are found
        best_wake_solns.append(best_wake_soln)
        best_wake_cbts.append(best_wake_cbt)
        best_wake_rrmses.append(rrmse_min)
        best_wake_n0s.append(n0) # To stop the stale density bug 

    for nss in range(N_sweep * N_sweep): # Loop over all combinations of n0 and Tp
        n0 = n0_sweep[nss // N_sweep]
        uz_front = uz_df['J (MA / m^2)'].iloc[Jmaxidx:rmaxidx + 1] / (cnst.q_e * n0) * 1e6 # Convert J to uz for front region (include both boundaries)
        front_solns_temp = all_front_solns[nss]
        rrmse_min = np.inf
        best_front_soln = None
        best_front_cbt = None
        for ifr, front_soln in enumerate(front_solns_temp):
            uz_soln_interp = np.interp(uz_df['R (m)'].iloc[Jmaxidx:rmaxidx + 1] - rmin, r_front, front_soln) # Interpolate fit to data points
            rmse = np.sqrt(mean_squared_error(uz_front, uz_soln_interp))
            rrmse = rmse / (uz_front.max() - uz_front.min()) # Normalize by range of data
            if not np.isclose(front_soln[-1], uz_front.iloc[-1], rtol=endpoint_rtol, atol=tolerance): # Check if the last point matches within tolerance
                continue # Skip nonphysical solutions
            if not np.isclose(front_soln[0], uz_front.iloc[0], rtol=endpoint_rtol, atol=tolerance): # Check if the first point matches within tolerance
                continue # Skip nonphysical solutions
            if rrmse < rrmse_min:
                rrmse_min = rrmse
                best_front_soln = front_soln
        if best_front_soln is None: continue
        best_front_solns.append(best_front_soln)
        best_front_cbts.append(best_front_cbt)
        best_front_rrmses.append(rrmse_min)
        best_front_n0s.append(n0) # To stop the stale density bug if it occurs for the front. Doesn't seem to here though bc no double complex conjugate pairs

    print('Best solution obtainment halted')
    return best_wake_solns, best_wake_cbts, best_front_solns, best_front_cbts, best_wake_rrmses, best_front_rrmses, best_wake_n0s, best_front_n0s

for r_plateau_val, r_plateau_idx_val in zip(r_plateau, r_plateau_idx):
    all_wake_solns, all_wake_cbts, all_front_solns, all_front_cbts, r_front, r_wake, rp_front, rp_wake = \
        sweep(r_plateau_val, r_plateau_idx_val)
    best_wake_solns, best_wake_cbts, best_front_solns, best_front_cbts, best_wake_rrmses, best_front_rrmses, best_wake_n0s, best_front_n0s = \
        accumulate_best_solutions(all_wake_solns, all_wake_cbts, all_front_solns, all_front_cbts, r_plateau_val, r_plateau_idx_val)
    pd.Series(best_wake_rrmses, name='rrmse').to_csv(f'../../analytic_fits/DIIID/best_wake_rrmses_rplatidx{r_plateau_idx_val}.csv', index=False)
    pd.Series(best_front_rrmses, name='rrmse').to_csv(f'../../analytic_fits/DIIID/best_front_rrmses_rplatidx{r_plateau_idx_val}.csv', index=False)

    plt.figure(r_plateau_idx_val)
    plt.scatter(uz_df['R (m)'][rminidx:] - rmin, uz_df['J (MA / m^2)'][rminidx:], label='DIIID Experimental Data')
    # plt.legend()
    for i, wake_soln in enumerate(best_wake_solns):
        # if best_wake_rrmses[i] < 0.1 and best_wake_cbts[i] < 0.35 * rp_wake: # Threshold to avoid clutter
        if best_wake_rrmses[i] < 0.2: # Threshold to avoid clutter
            # plt.plot(r_wake, wake_soln / 1e6 * cnst.q_e * best_wake_n0s[i], label=f'Wake fit {i+1}, n0 = {best_wake_n0s[i]:.4e}, RRMSE = {best_wake_rrmses[i]:.4f}')
            plt.plot(r_wake, wake_soln / 1e6 * cnst.q_e * best_wake_n0s[i])

    for i, front_soln in enumerate(best_front_solns):
        if best_front_rrmses[i] < 0.2: # Threshold to avoid clutter
            # plt.plot(r_front, front_soln / 1e6 * cnst.q_e * best_front_n0s[i], label=f'Front fit {i+1}, RRMSE = {best_front_rrmses[i]:.4f}, n0 = {best_front_n0s[i]:.4e}')
            plt.plot(r_front, front_soln / 1e6 * cnst.q_e * best_front_n0s[i])

    plt.title(f'Shear-flow stabilized solutions to DIIID edge pedestal toroidal current density plateau index = {r_plateau_idx_val} \n $N$ = {N_sweep * N_sweep}, rp = {rp_wake:.3f} m (wake), {rp_front:.3f} m (front), n0 = {n0_min:.2e} - {n0_max:.2e} $m^{{-3}}$, Tp = {Tp_min / cnst.eV_to_K:.2f} - {Tp_max / cnst.eV_to_K:.2f} eV')
    plt.xlabel('Radius (m)')
    plt.ylabel('$J_\\phi$ (MA/m$^2$)')

# r_Jmax = uz_df['R (m)'][Jmaxidx] # R at maximum current density
# Jzmax = uz_df['J (MA / m^2)'][Jmaxidx]

# num_r = 10000
# r_wake = np.linspace(0.0, r_Jmax - rmin, num_r) # Radial positions for wake solution, from 0 to R at max current density
# rp_wake = r_Jmax - rmin # Pinch radius [m]; Tied to dataset for better fidelity   

# r_front = np.linspace(r_Jmax - rmin, rmax - rmin, num_r) # Radial positions for plotting front solution, from R at max current density to max radius
# rp_front = np.abs(rmax - r_Jmax) # Pinch radius [m]; Tied to dataset for better fidelity
# r_front_calc = np.linspace(0.0, rp_front, num_r) # Radial positions for calculating front solution, from 0 to rp

# # Solve problem & accumulate the naturally best solutions
# # Wake solve 
# all_wake_solns = []
# all_wake_cbts = []
# for ns in range(N_sweep):
#     for ts in range(N_sweep):
#         n0 = n0_sweep[ns]
#         Tp = Tp_sweep[ts]
#         u0_wake = 1e6 * uz_df['J (MA / m^2)'][rminidx] / (cnst.q_e * n0) # Core flow velocity [m/s]; J = n e u => u = J / (n e)
#         uedge_wake = 1e6 * uz_df['J (MA / m^2)'][Jmaxidx] / (cnst.q_e * n0) # Edge flow velocity [m/s]; J = n e u => u = J / (n e)
#         uz0_roots = cpfm.root_solve_chi2_posbulk(uedge_wake, u0_wake, n0, rp_wake, Tp)
#         wake_solns = []
#         wake_cbts = []
#         for uz0_root in uz0_roots:
#             if np.abs(uz0_root.imag) > 1e-6: # Skip complex roots, they lie outside the ideal framework and don't yield real solutions
#                 continue
#             uz0 = np.abs(uz0_root) # Take magnitude bc cbt is the same, it doesn't impact real solns, and complex solns lie outside the ideal framework anyway
#             cbt = cpfm.cbt(n0, uz0, rp_wake, Tp) # Vortex constant [m]
#             print(f'cbt for uz0 = {uz0:2e} m/s: {cbt} m')
#             wake_cbts.append(cbt)
#             uz_fit = cpfm.uz_chi2cubic_posbulk(cbt, uz0, u0_wake, r_wake)
#             wake_solns.append(uz_fit)
#         all_wake_solns.append(wake_solns)
#         all_wake_cbts.append(wake_cbts)

# # Front solve
# all_front_solns = []
# all_front_cbts = []
# for ns in range(N_sweep):
#     for ts in range(N_sweep):
#         n0 = n0_sweep[ns]
#         Tp = Tp_sweep[ts]
#         u0_front = 1e6 * Jzmax / (cnst.q_e * n0) # Core flow velocity [m/s]; J = n e u => u = J / (n e)
#         uedge_front = 1e6 * uz_df['J (MA / m^2)'][rmaxidx] / (cnst.q_e * n0) # Edge flow velocity [m/s]; J = n e u => u = J / (n e)
#         uz0_roots = cpfm.root_solve_chi2_negbulk(uedge_front, u0_front, n0, rp_front, Tp)
#         front_solns = []
#         front_cbts = []
#         for uz0_root in uz0_roots:
#             uz0 = np.abs(uz0_root) # Take magnitude
#             cbt = cpfm.cbt(n0, uz0, rp_front, Tp) # Vortex constant [m]
#             print(f'cbt for uz0 = {uz0:2e} m/s: {cbt} m')
#             front_cbts.append(cbt)
#             uz_fit = cpfm.uz_chi2cubic_negbulk(cbt, uz0, u0_front, r_front_calc)
#             front_solns.append(uz_fit)
#         all_front_solns.append(front_solns)
#         all_front_cbts.append(front_cbts)

# Accumulate the naturally best solutions
# best_wake_solns = []
# best_front_solns = []

# best_wake_cbts = []
# best_front_cbts = []

# best_wake_rrmses = []
# best_front_rrmses = []

# best_wake_n0s = [] # Stale density bug occurs downstream from existence of double complex conjugate pairs 
# best_front_n0s = []

# tolerance = 1e-4
# endpoint_rtol = 1e-3
# for nss in range(N_sweep * N_sweep): # Loop over all combinations of n0 and Tp
#     n0 = n0_sweep[nss // N_sweep]
#     uz_wake = uz_df['J (MA / m^2)'].iloc[rminidx:Jmaxidx + 1] / (cnst.q_e * n0) * 1e6 # Convert J to uz for wake region (include both boundaries)
#     wake_solns_temp = all_wake_solns[nss]
#     rrmse_min = np.inf
#     best_wake_soln = None
#     best_wake_cbt = None
#     for iw, wake_soln in enumerate(wake_solns_temp):
#         uz_soln_interp = np.interp(uz_df['R (m)'].iloc[rminidx:Jmaxidx + 1] - rmin, r_wake, wake_soln) # Interpolate fit to data points
#         rmse = np.sqrt(mean_squared_error(uz_wake, uz_soln_interp))
#         rrmse = rmse / (uz_wake.max() - uz_wake.min()) # Normalize by range of data
#         if not np.isclose(wake_soln[-1], uz_wake.iloc[-1], rtol=endpoint_rtol, atol=tolerance): # Check if the last point matches within tolerance
#             continue # Skip nonphysical solutions
#         if not np.isclose(wake_soln[0], uz_wake.iloc[0], rtol=endpoint_rtol, atol=tolerance): # Check if the first point matches within tolerance
#             continue # Skip nonphysical solutions
#         if rrmse < rrmse_min:
#             rrmse_min = rrmse
#             best_wake_soln = wake_soln
#             best_wake_cbt = all_wake_cbts[nss][iw]
#     if best_wake_soln is None: # No valid solution found, skip
#         continue # Causes stale density bug if double complex conjugate pairs are found
#     best_wake_solns.append(best_wake_soln)
#     best_wake_cbts.append(best_wake_cbt)
#     best_wake_rrmses.append(rrmse_min)
#     best_wake_n0s.append(n0) # To stop the stale density bug 

# for nss in range(N_sweep * N_sweep): # Loop over all combinations of n0 and Tp
#     n0 = n0_sweep[nss // N_sweep]
#     uz_front = uz_df['J (MA / m^2)'].iloc[Jmaxidx:rmaxidx + 1] / (cnst.q_e * n0) * 1e6 # Convert J to uz for front region (include both boundaries)
#     front_solns_temp = all_front_solns[nss]
#     rrmse_min = np.inf
#     best_front_soln = None
#     best_front_cbt = None
#     for ifr, front_soln in enumerate(front_solns_temp):
#         uz_soln_interp = np.interp(uz_df['R (m)'].iloc[Jmaxidx:rmaxidx + 1] - rmin, r_front, front_soln) # Interpolate fit to data points
#         rmse = np.sqrt(mean_squared_error(uz_front, uz_soln_interp))
#         rrmse = rmse / (uz_front.max() - uz_front.min()) # Normalize by range of data
#         if not np.isclose(front_soln[-1], uz_front.iloc[-1], rtol=endpoint_rtol, atol=tolerance): # Check if the last point matches within tolerance
#             continue # Skip nonphysical solutions
#         if not np.isclose(front_soln[0], uz_front.iloc[0], rtol=endpoint_rtol, atol=tolerance): # Check if the first point matches within tolerance
#             continue # Skip nonphysical solutions
#         if rrmse < rrmse_min:
#             rrmse_min = rrmse
#             best_front_soln = front_soln
#     if best_front_soln is None: continue
#     best_front_solns.append(best_front_soln)
#     best_front_rrmses.append(rrmse_min)
#     best_front_n0s.append(n0) # To stop the stale density bug if it occurs for the front. Doesn't seem to here though bc no double complex conjugate pairs

# print('Best solution obtainment halted')

# Save RRMSE values to ../analytic_fits/DIIID/
# pd.Series(best_wake_rrmses, name='rrmse').to_csv('../../analytic_fits/DIIID/best_wake_rrmses.csv', index=False)
# pd.Series(best_front_rrmses, name='rrmse').to_csv('../../analytic_fits/DIIID/best_front_rrmses.csv', index=False)

# # Create band plots showing the range over which these best solutions vary next to the experimental data
# plt.figure()
# plt.scatter(uz_df['R (m)'][rminidx:] - rmin, uz_df['J (MA / m^2)'][rminidx:], label='MAST Pre-ELM J_phi')
# for i, wake_soln in enumerate(best_wake_solns):
#     # if best_wake_rrmses[i] < 0.1 and best_wake_cbts[i] < 0.35 * rp_wake: # Threshold to avoid clutter
#     if best_wake_rrmses[i] < 0.2: # Threshold to avoid clutter
#         plt.plot(r_wake, wake_soln / 1e6 * cnst.q_e * best_wake_n0s[i], label=f'Wake fit {i+1}, n0 = {best_wake_n0s[i]:.4e}, RRMSE = {best_wake_rrmses[i]:.4f}')

# for i, front_soln in enumerate(best_front_solns):
#     if best_front_rrmses[i] < 0.2: # Threshold to avoid clutter
#         plt.plot(r_front, front_soln / 1e6 * cnst.q_e * best_front_n0s[i], label=f'Front fit {i+1}, RRMSE = {best_front_rrmses[i]:.4f}, n0 = {best_front_n0s[i]:.4e}')

# plt.title(f'Cubic vortex solutions to DIIID edge pedestal toroidal current density profile, $N_{{sweep}}$ = {N_sweep}, rp = {rp_wake:.3f} m (wake), {rp_front:.3f} m (front), n0 = {n0_min:.2e} - {n0_max:.2e} $m^{{-3}}$, Tp = {Tp_min / cnst.eV_to_K:.2f} - {Tp_max / cnst.eV_to_K:.2f} eV')
# plt.xlabel('Radius (m)')
# plt.ylabel('$J_\\phi$ (MA/m$^2$)')
# plt.legend()

# plt.fill_between(r_wake, wake_lo, wake_hi, alpha=0.3, label='Wake band')
# plt.fill_between(r_front, front_lo, front_hi, alpha=0.3, label='Front band')
# plt.legend()

plt.show()