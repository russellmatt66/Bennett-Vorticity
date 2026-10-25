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

vs_points = 10
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
u_edge = u_Hedge * cnst.mH / (cnst.mH + cnst.mOX) + u_OXedge * cnst.mOX / (cnst.mH + cnst.mOX)

J_edge = cnst.q_e * n0 * u_edge
alpha = J_edge / front_edge_idx[intensity_string] 
J_0 = alpha * front_base_idx[intensity_string]
u_0_front = J_0 / (cnst.q_e * n0)

front_C_listlist = []
uz_front_listlist = []
num_points = 1000
for ir in range(len(pinch_radii)):
    rp = pinch_radii.iloc[ir]
    r_front = np.linspace(0, rp, num_points)
    for iH in range(vs_points):
        for iOX in range(vs_points):
            u_edge = u_edgelist[iH, iOX]
            front_C_list = []
            uz_front_list = []
            uz0_front_roots = cpfm.root_solve_chi2_posbulk(u_edge, u_0_front, n0, rp, Tp)
            for root in uz0_front_roots:
                if np.abs(root) > 3e8: continue # FTL - unphysical (or tachyon ... ?)
                if np.abs(root.imag) > 1e-12: continue # Need real velocity
                front_C_list.append(cpfm.cbt(n0, np.abs(root), rp, Tp))
                if front_C_list[-1] / rp < 0.1: continue # poorly localized shear layer - too abrupt
                uz_front_list.append(cpfm.uz_chi2cubic_posbulk(front_C_list[-1], np.abs(root), u_0_front, r_front))
            front_C_listlist.append(front_C_list)
            uz_front_listlist.append(uz_front_list)

uz_front_tensor = np.array(uz_front_listlist)
print(uz_front_tensor.shape)

# Plot the solutions that survived
for ir in range(len(pinch_radii)):
    plt.figure(ir)
    plt.xlabel("r [um]")
    plt.ylabel("uz [m/s]")
    plt.title("Surviving front solutions")
    plt.scatter(front_data[space_string], front_data[intensity_string], label="Experimental Data") # Plot experimental data of the front region
    rp = pinch_radii.iloc[ir]
    for iH in range(vs_points):
        for iOX in range(vs_points):
            idx = ir * vs_points * vs_points + iH * vs_points + iOX
            uz_front = uz_front_tensor[idx, 0]
            if len(uz_front) == 0: continue
            plt.plot(front_base_idx[space_string] + np.linspace(0, rp, num_points) * 1e6, (cnst.q_e * n0) / alpha * uz_front)

plt.show()