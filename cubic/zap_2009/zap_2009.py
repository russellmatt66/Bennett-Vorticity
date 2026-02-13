import sys
import pathlib
# ensure project root is on sys.path so the sibling `modules` package is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from modules import constants as cnst
from modules import spitzer as spz
from modules import cubic_pureflow_module as cpfm
from modules import plasma_properties as pp
from modules import powerbalance as pb

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import mean_squared_error

uz_data = pd.read_csv('../../experimental_data/zap_2009/zap2009_uz_fig9.csv', header=0, skiprows=[1])

# print(uz_data.columns)

# uz_tau_neg_0pt10 = uz_data['uz_tau_neg_0pt10'].dropna()
uz_tau_neg_0pt10 = pd.DataFrame({
    'r (mm)': uz_data['uz_tau_neg_0pt10'],
    'uz (km/s)': uz_data['Unnamed: 1']
}).dropna()

uz_tau_0pt10 = pd.DataFrame({
    'r (mm)': uz_data['uz_tau_0pt10'],
    'uz (km/s)': uz_data['Unnamed: 3']
}).dropna()

uz_tau_0pt16 = pd.DataFrame({
    'r (mm)': uz_data['uz_tau_0pt16'],
    'uz (km/s)': uz_data['Unnamed: 5']
}).dropna()

uz_tau_0pt34 = pd.DataFrame({
    'r (mm)': uz_data['uz_tau_0pt34'],
    'uz (km/s)': uz_data['Unnamed: 7']
}).dropna()

uz_tau_0pt56 = pd.DataFrame({
    'r (mm)': uz_data['uz_tau_0pt56'],
    'uz (km/s)': uz_data['Unnamed: 9']
}).dropna()

print(f'uz_tau_neg_0pt10: {uz_tau_neg_0pt10}')
# print(f'uz_tau_0pt10: {uz_tau_0pt10}')
# print(f'uz_tau_0pt16: {uz_tau_0pt16}')
# print(f'uz_tau_0pt34: {uz_tau_0pt34}')
# print(f'uz_tau_0pt56: {uz_tau_0pt56}')

# uz_data.columns = ['Radius (mm)', 'uz (10^{4} m / s)']

# r_data = uz_data['Radius (mm)'].to_numpy() * 1e-3 # Convert to meters
# uz_data = uz_data['uz (10^{4} m / s)'].to_numpy() * 1e4 # Convert to m/s

# uzpos = uz_data[r_data > 0]
# uzneg = uz_data[r_data < 0] 
# rpos = r_data[r_data > 0]
# rneg = r_data[r_data < 0] 

"""
Make fits of Bennett vortices to each half-chord
"""
# n0 = 1e23 # Plasma density [m^-3]; 1e22 - 1e23
# Tp = 200 * cnst.eV_to_K # Plasma temperature [K]; T = Te + Ti = 150 - 200 eV
# uedge = 4e4 # Edge flow velocity [m/s]; 
# u0 = 10e4 # Core flow velocity [m/s]; 
# rp = 10e-3 # Pinch radius [m]; 10mm

cbt = []
uzpos_fits = []
uzneg_fits = []