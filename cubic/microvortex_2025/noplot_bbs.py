'''
Sweep over base-point/plateau-point pairs:
- base points between the experimental origin and the lowest-intensity point
- plateau points near maximal intensity
https://www.nature.com/articles/s41467-025-67224-3

O(Nbase * Nplateau * NV^2)
ipair = Nbase * Nplateau
'''
import sys
import pathlib
# ensure project root is on sys.path so the sibling `modules` package is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

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

# Plateau candidates (near-maximal intensity points).
Imax = front_edge_idx[intensity_string]
frac = 0.9
threshold = frac * Imax
points_near_max = front_data[front_data[intensity_string] >= threshold].reset_index(drop=True)
print(f"Plateau-point sweep candidates (>= {frac*100}% of Imax): {points_near_max}")

# Sweep base points from the experimental origin to the lowest-intensity base point.
front_origin_idx = front_data.loc[front_data[space_string].idxmin()]
r_origin_um = front_origin_idx[space_string]
r_base_um = front_base_idx[space_string]
r_min_um = min(r_origin_um, r_base_um)
r_max_um = max(r_origin_um, r_base_um)
base_points = (
    front_data[
        (front_data[space_string] >= r_min_um)
        & (front_data[space_string] <= r_max_um)
    ]
    .sort_values(space_string)
    .reset_index(drop=True)
)
print(f"Base-point sweep candidates (origin to lowest-intensity base): {base_points}")

# Build all valid base/plateau pairs (plateau must be radially outward from base).
pair_configs = []
for ibase, base_point in base_points.iterrows():
    base_r_um = base_point[space_string]
    I0 = base_point[intensity_string]
    for iplateau, plateau_point in points_near_max.iterrows():
        plateau_r_um = plateau_point[space_string]
        rp_m = (plateau_r_um - base_r_um) * 1e-6
        if rp_m <= 0:
            continue
        pair_configs.append({
            "ipair": len(pair_configs),
            "ibase": int(ibase),
            "iplateau": int(iplateau),
            "r0_um": base_r_um,
            "rp_m": rp_m,
            "rp_um": rp_m * 1e6,
            "I0_target": I0,
            "I_edge_target": plateau_point[intensity_string],
            "r_edge_um": plateau_r_um,
        })

if len(pair_configs) == 0:
    raise ValueError("No valid base/plateau pairs found with positive pinch radius.")
print(f"Number of valid base/plateau sweep pairs: {len(pair_configs)}")

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

# Sweep over base/plateau pair candidates
n0 = 1e23 # [m^-3]
Tp = 5 * cnst.eV_to_K # [K]

# Need discharge alpha and u0
# u_Hedge = 731e3 # [m/s] # 731 +- 39e3 m/s
# u_OXedge = 191e3 # [m/s] # 191 +- 21e3 m/s 
# u_edge = u_Hedge * cnst.mH / (cnst.mH + cnst.mOX) + u_OXedge * cnst.mOX / (cnst.mH + cnst.mOX)

# J_edge = cnst.q_e * n0 * u_edge
# alpha = J_edge / front_edge_idx[intensity_string] 

alpha_tensor = np.zeros((len(pair_configs), vs_points, vs_points))
u0_tensor = np.zeros((len(pair_configs), vs_points, vs_points))

J_edge = cnst.q_e * n0 * u_edgelist
# alpha = J_edge / front_edge_idx[intensity_string] 

# J_0 = alpha * front_base_idx[intensity_string]
# u_0_front = J_0 / (cnst.q_e * n0)

front_C_listlist = []
uz_front_listlist = []
num_points = 1000
for ipair, pair_cfg in enumerate(pair_configs):
    rp = pair_cfg["rp_m"]
    I0 = pair_cfg["I0_target"]
    I_edge = pair_cfg["I_edge_target"]
    r_front = np.linspace(0, rp, num_points)
    for iH in range(vs_points):
        for iOX in range(vs_points):
            alpha_tensor[ipair, iH, iOX] = J_edge[iH, iOX] / I_edge
            J_0 = alpha_tensor[ipair, iH, iOX] * I0
            u0_tensor[ipair, iH, iOX] = J_0 / (cnst.q_e * n0)
            u_edge = u_edgelist[iH, iOX]
            front_C_list = []
            uz_front_list = []
            uz0_front_roots = cpfm.root_solve_chi2_posbulk(u_edge, u0_tensor[ipair, iH, iOX], n0, rp, Tp)
            for root in uz0_front_roots:
                if np.abs(root) > 3e8: continue # FTL - unphysical (or tachyon ... ?)
                if np.abs(root.imag) > 1e-12: continue # Need real velocity
                front_C_list.append(cpfm.cbt(n0, np.abs(root), rp, Tp))
                if front_C_list[-1] / rp < 0.1: continue # poorly localized shear layer - too abrupt
                uz_front_list.append(cpfm.uz_chi2cubic_posbulk(front_C_list[-1], np.abs(root), u0_tensor[ipair, iH, iOX], r_front))
            front_C_listlist.append(front_C_list)
            uz_front_listlist.append(uz_front_list)

uz_front_tensor = np.array(uz_front_listlist)
print(uz_front_tensor.shape)

def rsq(y, yhat):
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

def shear_layer_bounds_from_transition(r_um, y_model, y0, y1, low_frac=0.10, high_frac=0.90):
    """
    Define shear layer as the 10%-90% transition interval of the modeled signal.
    Returns (r_start_um, r_end_um) or (np.nan, np.nan) if undefined.
    """
    dy = y1 - y0
    if np.abs(dy) < 1e-30:
        return np.nan, np.nan

    # Normalized transition coordinate.
    s = (y_model - y0) / dy
    s = np.asarray(s)

    def first_crossing(x, z, level):
        for k in range(len(z) - 1):
            z0 = z[k]
            z1 = z[k + 1]
            if (z0 - level) == 0:
                return x[k]
            if (z0 - level) * (z1 - level) <= 0:
                dz = z1 - z0
                if np.abs(dz) < 1e-30:
                    return x[k]
                t = (level - z0) / dz
                return x[k] + t * (x[k + 1] - x[k])
        return np.nan

    r_start = first_crossing(r_um, s, low_frac)
    r_end = first_crossing(r_um, s, high_frac)

    if np.isnan(r_start) or np.isnan(r_end):
        return np.nan, np.nan
    if r_end < r_start:
        r_start, r_end = r_end, r_start
    return r_start, r_end

r_exp_um = front_data[space_string].to_numpy()
I_exp = front_data[intensity_string].to_numpy()

records = []

# Plot the solutions that survived and record statistics
for ipair, pair_cfg in enumerate(pair_configs):
    # plt.figure(ipair)
    # plt.rcParams.update({
    #     "font.size": 16,
    #     "axes.titlesize": 18,
    #     "axes.labelsize": 24,
    #     "legend.fontsize": 16,
    #     "xtick.labelsize": 16,
    #     "ytick.labelsize": 16,
    # })
    # plt.xlabel("r [um]")
    # plt.ylabel("Intensity (arb. units)")
    # plt.scatter(front_data[space_string], front_data[intensity_string], label="Experimental Data") # Plot experimental data of the front region
    rp = pair_cfg["rp_m"]
    base_r_um = pair_cfg["r0_um"]
    plateau_r_um = pair_cfg["r_edge_um"]
    # plt.title(f"Pair {ipair}: base#{pair_cfg['ibase']} to plateau#{pair_cfg['iplateau']} | n0 = {n0:.2e} [m^-3], Tp = {cnst.K_to_eV * Tp:.2e} [eV], rp = {rp:.2e} [m], r0 = {base_r_um:.2f} [um], r_edge = {plateau_r_um:.2f} [um]")
    for iH in range(vs_points):
        for iOX in range(vs_points):
            idx = ipair * vs_points * vs_points + iH * vs_points + iOX # all the solutions were stored flattened
            uz_front = uz_front_tensor[idx, 0]
            if len(uz_front) == 0: continue
            # Calculate Rsquared value
            r_model_um = base_r_um + np.linspace(0, rp, num_points) * 1e6
            I_model = (cnst.q_e * n0) / alpha_tensor[ipair, iH, iOX] * uz_front
            I_pred = np.interp(r_exp_um, r_model_um, I_model, left=np.nan, right=np.nan)
            rp_um = rp * 1e6

            # Comparison metric over the full modeled interval [r0, r0 + rp].
            mask_full = (
                (r_exp_um >= base_r_um)
                & (r_exp_um <= base_r_um + rp_um)
                & np.isfinite(I_pred)
            )
            r2_full = rsq(I_exp[mask_full], I_pred[mask_full]) if np.sum(mask_full) >= 3 else np.nan

            # Define shear layer by modeled 20%-80% transition instead of full [r0, r0+rp].
            r_shear_start_um, r_shear_end_um = shear_layer_bounds_from_transition(
                r_model_um,
                I_model,
                pair_cfg["I0_target"],
                pair_cfg["I_edge_target"],
                low_frac=0.40,
                high_frac=0.80,
            )
            mask_shear = (
                (r_exp_um >= r_shear_start_um)
                & (r_exp_um <= r_shear_end_um)
                & np.isfinite(I_pred)
            )
            r2_shear = rsq(I_exp[mask_shear], I_pred[mask_shear]) if np.sum(mask_shear) >= 3 else np.nan
            records.append({
                "ipair": ipair,
                "ibase": pair_cfg["ibase"],
                "iplateau": pair_cfg["iplateau"],
                "iH": iH,
                "iOX": iOX,
                "rp_m": rp,
                "rp_um": rp_um,
                "u_edge_mps": u_edgelist[iH, iOX],
                "u_edgeH_mps": u_Hedgelist[iH],
                "u_edgeOX_mps": u_OXedgelist[iOX],
                "r0_um": base_r_um,
                "r_edge_um": plateau_r_um,
                "I0_target": pair_cfg["I0_target"],
                "I_edge_target": pair_cfg["I_edge_target"],
                "alpha": alpha_tensor[ipair, iH, iOX],
                "u0_mps": u0_tensor[ipair, iH, iOX],
                "r2_full": r2_full,
                "r2_shear": r2_shear,
                "r_shear_start_um": r_shear_start_um,
                "r_shear_end_um": r_shear_end_um,
                "shear_width_um": (r_shear_end_um - r_shear_start_um) if np.isfinite(r_shear_start_um) and np.isfinite(r_shear_end_um) else np.nan,
                "n_full_points": int(np.sum(mask_full)),
                "n_shear_points": int(np.sum(mask_shear)),
                "has_solution": int(len(uz_front) > 0)
            })
            # Plot
            # plt.plot(r_model_um, I_model)
            # plt.legend()

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

n_above_090 = (solns_df["r2_shear"] > 0.95).sum()
print("Count r2_shear > 0.95:", n_above_090)

topN = 10
best_solns_df = solns_df.nlargest(topN, "r2_shear")
print(f"Top {topN} solutions by shear-only r2_shear:")
print(best_solns_df[[
    "ipair", "ibase", "iplateau", "iH", "iOX", "r2_full", "r2_shear",
    "r_shear_start_um", "r_shear_end_um", "shear_width_um", "n_full_points", "n_shear_points"
]])

# n_above_090 = (solns_df["r2_shear"] > 0.90).sum()
# print("Count r2_shear > 0.90:", n_above_090)

for tipair in range(len(pair_configs)):
    pair_cfg = pair_configs[tipair]
    mask_tir = (solns_df["ipair"] == tipair) & (solns_df["r2_shear"] > 0.95)
    n_above_090_tir = mask_tir.sum()
    print(f"Count r2_shear > 0.95 for pair ipair={tipair} (base#{pair_cfg['ibase']}, plateau#{pair_cfg['iplateau']}):", n_above_090_tir)

for tipair in range(len(pair_configs)):
    pair_cfg = pair_configs[tipair]
    mask_tir = (solns_df["ipair"] == tipair)
    r2_avg = solns_df.loc[mask_tir, "r2_shear"].mean()
    r2_median = solns_df.loc[mask_tir, "r2_shear"].median()
    r2_uncertainty = solns_df.loc[mask_tir, "r2_shear"].std()
    print(f"Uncertainty r2_shear for pair ipair={tipair} (base#{pair_cfg['ibase']}, plateau#{pair_cfg['iplateau']}):", r2_uncertainty)
    print(f"Median r2_shear for pair ipair={tipair} (base#{pair_cfg['ibase']}, plateau#{pair_cfg['iplateau']}):", r2_median)
    print(f"Average r2_shear for pair ipair={tipair} (base#{pair_cfg['ibase']}, plateau#{pair_cfg['iplateau']}):", r2_avg)

valid = solns_df["r2_shear"].notna().sum()
frac_above_090 = n_above_090 / valid if valid else np.nan
print("Fraction (of non-NaN):", frac_above_090)

solns_df.to_csv("../../analytic_fits/bella_2025/solns_df.csv")

plt.show()