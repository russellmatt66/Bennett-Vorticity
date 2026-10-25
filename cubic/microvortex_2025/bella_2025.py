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

# Split dataset into front and wake regions
front_data = data[data[space_string] >= 0] # don't mind including r = 0 in both datasets
wake_data = data[data[space_string] <= 0]

# grab base point = lowest intensity in the regions
def grab_base_point(region_data):
    return region_data.loc[region_data[intensity_string].idxmin()]

front_base_idx = grab_base_point(front_data)
wake_base_idx = grab_base_point(wake_data)
print("Front base index:", front_base_idx)
print("Wake base index:", wake_base_idx)

def grab_edge_point(region_data):
    return region_data.loc[region_data[intensity_string].idxmax()]

front_edge_idx = grab_edge_point(front_data)
wake_edge_idx = grab_edge_point(wake_data)

def get_interior_point(region_data):
    return region_data.loc[region_data[space_string].idxmin()][intensity_string]

# Front: Positive half-chord
# Define vortex parameters in the front region
# Separate the region into individual vortices for analysis
# Solve the vortex model in each region
n0 = 1e23 # [m^-3]
Tp = 5 * cnst.eV_to_K # [K]
rp_front = (front_edge_idx[space_string] - front_base_idx[space_string]) * 1e-6
# rp_front = (front_edge_idx[space_string]) * 1e-6

print("rp_front [um]: ", rp_front)

u_Hedge = 731e3 # [m/s] # 731 +- 39e3 m/s
u_OXedge = 191e3 # [m/s] # 191 +- 21e3 m/s 
u_edge = u_Hedge * cnst.mH / (cnst.mH + cnst.mOX) + u_OXedge * cnst.mOX / (cnst.mH + cnst.mOX)
# u_edge = u_Hedge

J_edge = cnst.q_e * n0 * u_edge

alpha = J_edge / front_edge_idx[intensity_string] 

J_0 = alpha * front_base_idx[intensity_string]
u_0_front = J_0 / (cnst.q_e * n0)

print("u_0_front [m/s]: ", u_0_front)

front_C_list = []

uz0_front_roots = cpfm.root_solve_chi2_posbulk(u_edge, u_0_front, n0, rp_front, Tp)

num_points = 1000
r_front = np.linspace(0, rp_front, num_points)

uz_fronts_list = []

for root in uz0_front_roots:
    if np.abs(root) > 3e8: continue
    if np.abs(root.imag) > 1e-12: continue
    front_C_list.append(cpfm.cbt(n0, np.abs(root), rp_front, Tp))
    if front_C_list[-1] / rp_front < 0.1: continue # poorly localized shear layer - too abrupt
    uz_fronts_list.append(cpfm.uz_chi2cubic_posbulk(front_C_list[-1], np.abs(root), u_0_front, r_front))

print("front_C_list", front_C_list)
# Wake: Negative half-chord
# Define vortex parameters in the wake region
rp_wake = wake_data[space_string].iloc[wake_edge_idx] - wake_data[space_string].iloc[wake_base_idx]

# Separate the region into individual vortices for analysis
# Solve the vortex model in each region

# Calculate R-squared
# Experimental x,y (front region)
x_exp = front_data[space_string].to_numpy()                     # um
y_exp = front_data[intensity_string].to_numpy()                 # intensity

# Model x grid (same as your plotting transform)
x_model = front_base_idx[space_string] + r_front * 1e6          # um

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

r2_values = []

for uz_front in uz_fronts_list:
    # Convert model velocity to intensity the same way you plot it
    y_model = (cnst.q_e * n0) / alpha * uz_front

    # Keep only experimental points inside model radius range
    mask = (x_exp >= np.min(x_model)) & (x_exp <= np.max(x_model))
    x_in = x_exp[mask]
    y_in = y_exp[mask]

    # Interpolate model onto experimental x locations
    y_model_on_exp = np.interp(x_in, x_model, y_model)

    # Compute R^2 over the shared range
    r2 = r_squared(y_in, y_model_on_exp)
    r2_values.append(r2)

print("R^2 values for each front solution:", r2_values)
if len(r2_values) > 0:
    print("Best R^2:", np.nanmax(r2_values))

# Plot the results
# Generate plots for the front and wake regions to visualize the vortex intensity profiles
plt.figure(figsize=(10, 6))
# plt.scatter(front_data[space_string], (alpha / (cnst.q_e * n0)) * front_data[intensity_string], label="Experimental Data") # Plot experimental data of the front region
plt.scatter(front_data[space_string], front_data[intensity_string], label="Experimental Data") # Plot experimental data of the front region
for uz_front in uz_fronts_list:
    # plt.plot(front_base_idx[space_string] + (r_front) * 1e6, uz_front)  # Convert r_front back to micrometers for plotting
    plt.plot(front_base_idx[space_string] + (r_front) * 1e6, (cnst.q_e * n0) / alpha * uz_front, label="Vortex Profile", color='red')  # Convert r_front back to micrometers for plotting
    # plt.plot((r_front) * 1e6, uz_front)  # Convert r_front back to micrometers for plotting
plt.xlabel("r (um)")
# plt.ylabel("Uz (m/s)")
plt.ylabel("Intensity (arb. units)")
plt.title(f"Vortex solutions to Balcazar (2025) microjet, n0 = {n0} [$m^{-3}$], Tp = {Tp * cnst.K_to_eV} [eV], uH = {u_Hedge:.2e} [m/s], uOX = {u_OXedge:.2e} [m/s]")
plt.legend()
plt.show()