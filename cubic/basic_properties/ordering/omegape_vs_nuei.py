import sys
import pathlib
# ensure project root is on sys.path so the sibling `modules` package is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from modules import constants as cnst
from modules import plasma_properties as pp 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

n_min = 1e19 # m^-3
n_max = 1e22 # m^-3

T_min = 1e3 * cnst.eV_to_K # K
T_max = 40e3 * cnst.eV_to_K # K

num_points = 1000
n = np.linspace(n_min, n_max, num_points) # Sweep over density
T = np.linspace(T_min, T_max, num_points) # Sweep over temperature

ln_Lambda = 10 # Coulomb logarithm (typical value)

omega_pe = np.array([pp.omega_pe(ni) for ni in n])
nu_ei = np.array([[pp.nu_ei(ni, Ti, ln_Lambda) for Ti in T] for ni in n])

smallness = np.zeros((len(n), len(T)))

non_small = 0
threshold = 0.00001
for i, ni in enumerate(n):
    for j, Ti in enumerate(T):
        smallness[i, j] = nu_ei[i, j] / omega_pe[i]
        if smallness[i, j] > threshold:
            non_small += 1

print(f'Number of non-small values (nu_ei / omega_pe > {threshold}): {non_small} out of {len(n) * len(T)} total points')
# print(smallness)