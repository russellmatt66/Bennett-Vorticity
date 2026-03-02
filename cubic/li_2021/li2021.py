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

I_data = pd.read_csv('../../experimental_data/li_2021/li2021_airplasmastreamer_fig3.csv', header=0, skiprows=[1])

# print(uz_data.head())

Iz = pd.DataFrame({
    'r (mm)': I_data['Figure3_Iz'],
    'Iz (A.U.)' : I_data['Unnamed: 1'], # not currents, but intensities of light emission
    # 'name' : 'Iz'
}).dropna().sort_values(by='r (mm)', ascending=True).drop_duplicates(subset=['r (mm)'])

Ix = pd.DataFrame({
    'r (mm)': I_data['Figure3_Ix'],
    'Ix (A.U.)' : I_data['Unnamed: 3'],
    # 'name' : 'Ix'
}).dropna().sort_values(by='r (mm)', ascending=True) 

print(Iz.head())
print(Ix.head())

Iz_needletip_data = pd.read_csv('../../experimental_data/li_2021/Figure3_Iz_Needletip.csv', header=0, skiprows=[1])

# print(Iz_needletip_data.head())

Iz_needletip = pd.DataFrame({
    'r (mm)': Iz_needletip_data.iloc[:, 0], # Assuming the first column is the spatial dimension
    'Iz (A.U.)' : Iz_needletip_data.iloc[:, 1], # Assuming the second column is the intensity
    # 'name' : 'Iz'
}).dropna().sort_values(by='r (mm)', ascending=True)
    
print(Iz_needletip.head())


# Plasma properties
u0 = 0.75e6 # Core flow velocity [m/s]; mm / ns -> m/s
n0 = 1e18 # Plasma density [m^-3]; 1e17 - 1e19
# Tp = 1e3 * cnst.eV_to_K # Plasma temperature [K]; T = Te + Ti ~ 1 keV is just a guess
Tp = 300 # Li et al (2021) estimate for gas temperature. p12, S4.6

rp_front = 5e-3 # m
rp_wake = 20e-3 # m 

Iz_front = Iz[Iz['r (mm)'] < 50] * 1e-3 # Convert to meters 
Iz_wake = Iz[Iz['r (mm)'] > 50] * 1e-3 # Convert to meters

I0 = Iz['Iz (A.U.)'].max() # Use the maximum intensity as a proxy for the core flow velocity
alpha = I0 / (cnst.q_e * n0 * u0) # Proportionality constant to convert intensity to velocity
print(f'Proportionality constant alpha: {alpha}')

uedge_front = 0.0
uedge_wake = 0.0
uz0_roots_front = cpfm.root_solve_chi2_negbulk(uedge_front, u0, n0, rp_front, Tp)
uz0_roots_wake = cpfm.root_solve_chi2_negbulk(uedge_wake, u0, n0, rp_wake, Tp)

uz_fits_front = []
cbts_front = []
for uz0 in uz0_roots_front:
    print(f'uz0 root for front: {uz0} m/s')
    cbt_front = cpfm.cbt(n0, np.abs(uz0), rp_front, Tp)    
    print(f'cbt for front: {cbt_front} m')
    uz_fit = cpfm.uz_chi2cubic_negbulk(cbt_front, uz0, u0, Iz_front['r (mm)'] * 1e-3) # Convert to meters
    uz_fits_front.append(uz_fit)
    cbts_front.append(cbt_front)

uz_fits_wake = []
cbts_wake = []
for uz0 in uz0_roots_wake:
    print(f'uz0 root for wake: {uz0} m/s')
    cbt_wake = cpfm.cbt(n0, np.abs(uz0), rp_wake, Tp)    
    print(f'cbt for wake: {cbt_wake} m')
    uz_fit = cpfm.uz_chi2cubic_negbulk(cbt_wake, uz0, u0, Iz_wake['r (mm)'] * 1e-3) # Convert to meters
    uz_fits_wake.append(uz_fit)
    cbts_wake.append(cbt_wake)

# plt.figure()


# plt.figure()
# plt.plot(Iz['r (mm)'], Iz['Iz (A.U.)'], label='Iz')
# plt.plot(Iz_needletip['r (mm)'], Iz_needletip['Iz (A.U.)'], label='Needle tip Iz')

# plt.xlabel('r (mm)')
# plt.ylabel('Intensity (A.U.)')
# plt.title('Light emission profiles from Li et al. 2021 Figure 3')
# plt.legend()

# plt.figure()
# plt.plot(Ix['r (mm)'], Ix['Ix (A.U.)'], label='Ix')
# plt.xlabel('r (mm)')
# plt.ylabel('Intensity (A.U.)')
# plt.title('Light emission profiles from Li et al. 2021 Figure 3')
# plt.legend()

plt.show()