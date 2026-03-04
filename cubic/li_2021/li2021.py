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
from scipy.signal import savgol_filter

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

# high-res Iz data
Iz_highres_data = pd.read_csv('../../experimental_data/li_2021/Figure3_Iz_highres.csv', header=0, skiprows=[1])

# print(Iz_highres_data.head())

Iz_highres = pd.DataFrame({
    'r (mm)': Iz_highres_data.iloc[:, 0], 
    'Iz (A.U.)' : Iz_highres_data.iloc[:, 1],
}).dropna().sort_values(by='r (mm)', ascending=True)

threshold_dr = 0.001
Iz_highres = Iz_highres[Iz_highres['r (mm)'].diff().fillna(np.inf) > threshold_dr] # eliminate duplicates (jitter) from manual digitization of high-res data

# threshold_I = 3.0
# Iz_highres = Iz_highres[Iz_highres['Iz (A.U.)'] ]

# window_length = 21 # should always be odd 
# poly_order = 3
# Iz_highres['Iz (A.U.)'] = savgol_filter(Iz_highres['Iz (A.U.)'], window_length, poly_order)

print(Iz_highres.head())

# Plasma properties
u0 = 0.75e6 # Core flow velocity [m/s]; mm / ns -> m/s
n0 = 5e17 # Plasma density [m^-3]; 1e17 - 1e19
# Tp = 1e3 * cnst.eV_to_K # Plasma temperature [K]; T = Te + Ti ~ 1 keV is just a guess
Tp_front = 50000 # Li et al (2021) estimate for gas temperature is 300 [degK]: p12, S4.6
Tp_wake = 60000

rp_front = 10e-3 # m
rp_wake = 10e-3 # m 

Iz_front = Iz[Iz['r (mm)'] < 50] * 1e-3 # Convert to meters 
Iz_wake = Iz[Iz['r (mm)'] > 50] * 1e-3 # Convert to meters

I0 = Iz['Iz (A.U.)'].max() # Use the maximum intensity as a proxy for the core flow velocity
alpha = I0 / (cnst.q_e * n0 * u0) # Proportionality constant to convert current density to intensity
print(f'Proportionality constant alpha: {alpha}')

uedge_front = Iz_highres[Iz_highres['r (mm)'] == Iz_highres['r (mm)'].min()]['Iz (A.U.)'].values[0] / (alpha * cnst.q_e * n0) # Convert to m/s
uedge_wake = Iz_highres[Iz_highres['r (mm)'] == Iz_highres['r (mm)'].max()]['Iz (A.U.)'].values[0] / (alpha * cnst.q_e * n0) # Convert to m/s

print(f'Edge flow velocity for front profile: {uedge_front} m/s')
print(f'Edge flow velocity for wake profile: {uedge_wake} m/s')

uz0_roots_front = cpfm.root_solve_chi2_negbulk(uedge_front, u0, n0, rp_front, Tp_front)
uz0_roots_wake = cpfm.root_solve_chi2_negbulk(uedge_wake, u0, n0, rp_wake, Tp_wake)

r_front = np.linspace(0, rp_front, 100) 
r_wake = np.linspace(0, rp_wake, 100)

uz_fits_front = []
cbts_front = []
for uz0 in uz0_roots_front:
    print(f'uz0 root for front: {uz0} m/s')
    cbt_front = cpfm.cbt(n0, np.abs(uz0), rp_front, Tp_front)    
    print(f'cbt for front: {cbt_front} m')
    uz_fit = cpfm.uz_chi2cubic_negbulk(cbt_front, np.abs(uz0), u0, r_front) 
    uz_fits_front.append(uz_fit)
    cbts_front.append(cbt_front)

uz_fits_wake = []
cbts_wake = []
for uz0 in uz0_roots_wake:
    print(f'uz0 root for wake: {uz0} m/s')
    cbt_wake = cpfm.cbt(n0, np.abs(uz0), rp_wake, Tp_wake)    
    print(f'cbt for wake: {cbt_wake} m')
    uz_fit = cpfm.uz_chi2cubic_negbulk(cbt_wake, np.abs(uz0), u0, r_wake) 
    uz_fits_wake.append(uz_fit)
    cbts_wake.append(cbt_wake)

plt.figure()
for i in range(len(uz_fits_front)):
    plt.plot(r_front * 1e3, uz_fits_front[i], label=f'Vortex {i+1}, (cbt={cbts_front[i]:.2e} m, uz0 ={uz0_roots_front[i]:.2e} m/s)')
    plt.xlabel('r (mm)')
    plt.ylabel('uz (m/s)')
    plt.title(f'Cubic vortex fit to Li et al. (2021) front profile \n $r_{{p}}$={rp_front*1e3:.0f} mm, $T_{{p}}$ = {Tp_front:.0f} K')
    plt.legend()

plt.figure()
for j in range(len(uz_fits_wake)):
    plt.plot(r_wake * 1e3, uz_fits_wake[j], label=f'Vortex {j+1}, (cbt={cbts_wake[j]:.2e} m, uz0 ={uz0_roots_wake[j]:.2e} m/s)')
    plt.xlabel('r (mm)')
    plt.ylabel('uz (m/s)')
    plt.title(f'Cubic vortex fit to Li et al. (2021) wake profile \n $r_{{p}}$={rp_wake*1e3:.0f} mm, $T_{{p}}$ = {Tp_wake:.0f} K')
    plt.legend()

z0 = Iz[Iz['Iz (A.U.)'] == Iz['Iz (A.U.)'].max()]['r (mm)'].values[0] # mm
print(f'z0 (position of maximum intensity in front profile): {z0} mm')

# Experimental Data
plt.figure()
# plt.plot(Iz['r (mm)'], Iz['Iz (A.U.)'], label='Iz')
plt.plot(Iz_highres['r (mm)'], Iz_highres['Iz (A.U.)'], label='Li et al. (2021) Iz')
for i in range(len(uz_fits_front)):
    plt.plot(-r_front * 1e3 + z0, alpha * cnst.q_e * n0 * uz_fits_front[i], label=f'Vortex {i+1}')

plt.xlabel('r (mm)')
plt.ylabel('Intensity (A.U.)')
plt.title(f'Vortex fits to Li et al. (2021) intensity')
plt.legend()

# plt.figure()
# plt.plot(Ix['r (mm)'], Ix['Ix (A.U.)'], label='Ix')
# plt.xlabel('r (mm)')
# plt.ylabel('Intensity (A.U.)')
# plt.title('Light emission profiles from Li et al. 2021 Figure 3')
# plt.legend()

plt.show()