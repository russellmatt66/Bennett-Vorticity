import sys
import pathlib
# ensure project root is on sys.path so the sibling `modules` package is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

'''
Calculate solutions for the water microvortex intensity profile from Balcazar et. al (2025):
https://www.nature.com/articles/s41467-025-67224-3
'''
from modules import constants as cnst

from modules import cubic_pureflow_module as cpfm
import numpy as np
import matplotlib.pyplot as plt

# Read in data and preprocess it for analysis
import pandas as pd

data = pd.read_csv("../../experimental_data/bella_2025/Intensity.csv")

space_string = "r (um)"
intensity_string = " Intensity (arb. units)"

front_data = data[data[space_string] >= 0] # don't mind including r = 0 in both datasets

# grab base point = lowest intensity in the regions
def grab_base_point(region_data):
    return region_data.loc[region_data[intensity_string].idxmin()]

front_base_idx = grab_base_point(front_data)

def grab_edge_point(region_data): # Use this for determining the maximal plateau
    return region_data.loc[region_data[intensity_string].idxmax()]

front_edge_idx = grab_edge_point(front_data)
Imax = front_edge_idx[intensity_string]

frac = 0.9
threshold = frac * Imax

points_near_max = front_data[front_data[intensity_string] >= threshold]
print(f"Points near maximal intensity (>= {frac*100}% of Imax): {points_near_max}")

r0 = front_base_idx[space_string] * 1e-6 # [m] convert from um to m
plateau_rs = points_near_max[space_string] * 1e-6 # [m] convert from um to m

pinch_radii = (plateau_rs - r0)
print(f"Pinch radii (m): {pinch_radii}")

# Sweep over experimental velocities
u_Hedge = 731e3 # [m/s] # 731 +- 39e3 m/s
u_OXedge = 191e3 # [m/s] # 191 +- 21e3 m/s 

u_Hedge_pm = 39e3 # [m/s]
u_OXedge_pm = 21e3 # [m/s]

vs_points = 25
u_Hedgelist = np.linspace(u_Hedge - u_Hedge_pm, u_Hedge + u_Hedge_pm, vs_points)
u_OXedgelist = np.linspace(u_OXedge - u_OXedge_pm, u_OXedge + u_OXedge_pm, vs_points) 

u_edgelist = np.zeros((vs_points, vs_points))
for i in range(vs_points):
    for j in range(vs_points):
        u_edgelist[i, j] = u_Hedgelist[i] * cnst.mH / (cnst.mH + cnst.mOX) + u_OXedgelist[j] * cnst.mOX / (cnst.mH + cnst.mOX)

# Sweep over maximal plateau
n0 = 1e23 # [m^-3]
Tp = 5 * cnst.eV_to_K # [K]

# Need discharge alpha and u0
# u_Hedge = 731e3 # [m/s] # 731 +- 39e3 m/s
# u_OXedge = 191e3 # [m/s] # 191 +- 21e3 m/s 
# u_edge = u_Hedge * cnst.mH / (cnst.mH + cnst.mOX) + u_OXedge * cnst.mOX / (cnst.mH + cnst.mOX)

# J_edge = cnst.q_e * n0 * u_edge
# alpha = J_edge / front_edge_idx[intensity_string] 

alpha_tensor = np.zeros((len(pinch_radii), vs_points, vs_points))
u0_tensor = np.zeros((len(pinch_radii), vs_points, vs_points))
I0 = front_base_idx[intensity_string]

J_edge = cnst.q_e * n0 * u_edgelist
# alpha = J_edge / front_edge_idx[intensity_string] 

# J_0 = alpha * front_base_idx[intensity_string]
# u_0_front = J_0 / (cnst.q_e * n0)

front_C_listlist = []
uz_front_listlist = []
num_points = 1000
for ir in range(len(pinch_radii)):
    rp = pinch_radii.iloc[ir]
    r_front = np.linspace(0, rp, num_points)
    for iH in range(vs_points):
        for iOX in range(vs_points):
            I_edge = points_near_max.iloc[ir][intensity_string]
            alpha_tensor[ir, iH, iOX] = J_edge[iH, iOX] / I_edge
            J_0 = alpha_tensor[ir, iH, iOX] * I0
            u0_tensor[ir, iH, iOX] = J_0 / (cnst.q_e * n0)
            u_edge = u_edgelist[iH, iOX]
            front_C_list = []
            uz_front_list = []
            uz0_front_roots = cpfm.root_solve_chi2_posbulk(u_edge, u0_tensor[ir, iH, iOX], n0, rp, Tp)
            for root in uz0_front_roots:
                if np.abs(root) > 3e8: continue # FTL - unphysical (or tachyon ... ?)
                if np.abs(root.imag) > 1e-12: continue # Need real velocity
                front_C_list.append(cpfm.cbt(n0, np.abs(root), rp, Tp))
                if front_C_list[-1] / rp < 0.1: continue # poorly localized shear layer - too abrupt
                uz_front_list.append(cpfm.uz_chi2cubic_posbulk(front_C_list[-1], np.abs(root), u0_tensor[ir, iH, iOX], r_front))
            front_C_listlist.append(front_C_list)
            uz_front_listlist.append(uz_front_list)

uz_front_tensor = np.array(uz_front_listlist)
print(uz_front_tensor.shape)

def rsq(y, yhat):
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

r_exp_um = front_data[space_string].to_numpy()
I_exp = front_data[intensity_string].to_numpy()

records = []

# Plot the solutions that survived and record statistics
for ir in range(len(pinch_radii)):
    plt.figure(ir)
    plt.xlabel("r [um]")
    plt.ylabel("Intensity (arb. units)")
    plt.title(f"Surviving front solutions for rp = {pinch_radii.iloc[ir]}")
    plt.scatter(front_data[space_string], front_data[intensity_string], label="Experimental Data") # Plot experimental data of the front region
    rp = pinch_radii.iloc[ir]
    for iH in range(vs_points):
        for iOX in range(vs_points):
            idx = ir * vs_points * vs_points + iH * vs_points + iOX # all the solutions were stored flattened
            uz_front = uz_front_tensor[idx, 0]
            if len(uz_front) == 0: continue
            # Calculate Rsquared value
            r_model_um = front_base_idx[space_string] + np.linspace(0, rp, num_points) * 1e6
            I_model = (cnst.q_e * n0) / alpha_tensor[ir, iH, iOX] * uz_front
            I_pred = np.interp(r_exp_um, r_model_um, I_model, left=np.nan, right=np.nan)
            rp_um = rp * 1e6
            r0_um = front_base_idx[space_string]
            mask_shear = (r_exp_um >= r0_um) & (r_exp_um <= r0_um + rp_um) & np.isfinite(I_pred)
            r2_shear = rsq(I_exp[mask_shear], I_pred[mask_shear]) if np.sum(mask_shear) >= 3 else np.nan
            records.append({
                "ir": ir,
                "iH": iH,
                "iOX": iOX,
                "rp_m": rp,
                "rp_um": rp_um,
                "u_edge_mps": u_edgelist[iH, iOX],
                "u_edgeH_mps": u_Hedgelist[iH],
                "u_edgeOX_mps": u_OXedgelist[iOX],
                "I_edge_target": points_near_max.iloc[ir][intensity_string],
                "alpha": alpha_tensor[ir, iH, iOX],
                "u0_mps": u0_tensor[ir, iH, iOX],
                "r2_shear": r2_shear,
                "n_shear_points": int(np.sum(mask_shear)),
                "has_solution": int(len(uz_front) > 0)
            })
            # Plot
            plt.plot(r_model_um, I_model)

solns_df = pd.DataFrame(records)
print(solns_df.head())

# scalar max (ignores NaN by default)
# max_r2 = solns_df["r2_shear"].max()
# print("max r2_shear =", max_r2)

# # full row for the best-fit case
# best_row = solns_df.loc[solns_df["r2_shear"].idxmax()]
# print(best_row)

# top10 = solns_df.nlargest(10, "r2_shear")
# print(top10)

n_above_090 = (solns_df["r2_shear"] > 0.90).sum()
print("Count r2_shear > 0.90:", n_above_090)

valid = solns_df["r2_shear"].notna().sum()
frac_above_090 = n_above_090 / valid if valid else np.nan
print("Fraction (of non-NaN):", frac_above_090)

solns_df.to_csv("../../analytic_fits/solns_df.csv")

plt.show()