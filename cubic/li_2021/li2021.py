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

Iz_highres_raw = pd.DataFrame({
    'r (mm)': Iz_highres_data.iloc[:, 0], 
    'Iz (A.U.)' : Iz_highres_data.iloc[:, 1],
}).dropna().sort_values(by='r (mm)', ascending=True)

# threshold_dr = 0.001 # for eliminating duplicates from manual digitization of data 
# Iz_highres = Iz_highres[Iz_highres['r (mm)'].diff().fillna(np.inf) > threshold_dr] # eliminate duplicates (jitter) from manual digitization of high-res data
Iz_highres = Iz_highres_raw.groupby('r (mm)').mean().reset_index().sort_values('r (mm)')

# Attempted smoothing of high-res data with Savitsky-Golay filter, but it seems to distort the profile too much, so commenting out for now
# threshold_I = 3.0
# Iz_highres = Iz_highres[Iz_highres['Iz (A.U.)'] ]

# window_length = 21 # should always be odd 
# poly_order = 3
# Iz_highres['Iz (A.U.)'] = savgol_filter(Iz_highres['Iz (A.U.)'], window_length, poly_order)

print(Iz_highres.head())

# Plasma properties
u0 = 0.75e6 # Core flow velocity [m/s]; mm / ns -> m/s
n0 = 5e17 # Plasma density [m^-3]; 1e17 - 1e19

u0_needletip = 0.375e6 # Core flow velocity for plasma at the needletip [m/s];
n0_needletip = 1e19 # Plasma density for plasma at the needletip

# Tp = 1e3 * cnst.eV_to_K # Plasma temperature [K]; T = Te + Ti ~ 1 keV is just a guess
Tp_front = 10000 # Li et al (2021) estimate for gas temperature is 300 [degK]: p12, S4.6
Tp_wake = 100000 
Tp_needletip = 100000 # needletip edge plasma temperature [degK]

rp_front = 5e-3 # m
rp_wake = 15e-3 # m 
rp_needletip = 7.5e-3 # m

Iz_front = Iz[Iz['r (mm)'] < 50] * 1e-3 # Convert to meters 
Iz_wake = Iz[Iz['r (mm)'] > 50] * 1e-3 # Convert to meters

I0 = Iz['Iz (A.U.)'].max() # Use the maximum intensity as a proxy for the core flow velocity
alpha = I0 / (cnst.q_e * n0 * u0) # Proportionality constant to convert current density to intensity
print(f'Proportionality constant alpha: {alpha}')

uedge_front = Iz_highres[Iz_highres['r (mm)'] == Iz_highres['r (mm)'].min()]['Iz (A.U.)'].values[0] / (alpha * cnst.q_e * n0) # Convert to m/s
uedge_wake = Iz_highres[Iz_highres['r (mm)'] == Iz_highres['r (mm)'].max()]['Iz (A.U.)'].values[0] / (alpha * cnst.q_e * n0) # Convert to m/s
uedge_needletip = Iz_needletip[Iz_needletip['r (mm)'] == Iz_needletip['r (mm)'].min()]['Iz (A.U.)'].values[0] / (alpha * cnst.q_e * n0) # Convert to m/s

print(f'Edge flow velocity for front profile: {uedge_front} m/s')
print(f'Edge flow velocity for wake profile: {uedge_wake} m/s')
print(f'Edge flow velocity for needletip profile: {uedge_needletip} m/s')

uz0_roots_front = cpfm.root_solve_chi2_negbulk(uedge_front, u0, n0, rp_front, Tp_front)
uz0_roots_wake = cpfm.root_solve_chi2_negbulk(uedge_wake, u0, n0, rp_wake, Tp_wake)
uz0_roots_needletip = cpfm.root_solve_chi2_negbulk(uedge_needletip, u0_needletip, n0_needletip, rp_needletip, Tp_needletip)

num_r_front = 100
num_r_wake = 100
num_r_needletip = 100

r_front = np.linspace(0, rp_front, num_r_front) 
r_wake = np.linspace(0, rp_wake, num_r_wake)
r_needletip = np.linspace(0, rp_needletip, num_r_needletip)

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

uz_fits_needletip = []
cbts_needletip = []
for uz0 in uz0_roots_needletip:
    print(f'uz0 root for needletip: {uz0} m/s')
    cbt_needletip = cpfm.cbt(n0_needletip, np.abs(uz0), rp_needletip, Tp_needletip)    
    print(f'cbt for needletip: {cbt_needletip} m')
    uz_fit = cpfm.uz_chi2cubic_negbulk(cbt_needletip, np.abs(uz0), u0_needletip, r_needletip) 
    uz_fits_needletip.append(uz_fit)
    cbts_needletip.append(cbt_needletip)

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

plt.figure()
for k in range (len(uz_fits_needletip)):
    plt.plot(r_needletip * 1e3, uz_fits_needletip[k], label=f'Vortex {k+1}, (cbt={cbts_needletip[k]:.2e} m, uz0 ={uz0_roots_needletip[k]:.2e} m/s)')
    plt.xlabel('r (mm)')
    plt.ylabel('uz (m/s)')
    plt.title(f'Cubic vortex fit to Li et al. (2021) needletip profile \n $r_{{p}}$={rp_needletip*1e3:.0f} mm, $T_{{p}}$ = {Tp_needletip:.0f} K')
    plt.legend() 

z0 = Iz[Iz['Iz (A.U.)'] == Iz['Iz (A.U.)'].max()]['r (mm)'].values[0] # mm - middle of the streamer head 
print(f'z0 (position of maximum intensity in front profile): {z0} mm')

z0_needletip = Iz_needletip[Iz_needletip['Iz (A.U.)'] == Iz_needletip['Iz (A.U.)'].max()]['r (mm)'].values[0] # mm - core of the needletip plasma
print(f'core of needletip (position of maximum intensity in needletip profile): {z0_needletip} mm')

# Experimental Data
plt.figure()
# plt.plot(Iz['r (mm)'], Iz['Iz (A.U.)'], label='Iz')
# plt.plot(Iz_highres['r (mm)'], Iz_highres['Iz (A.U.)'], label='Li et al. (2021) Iz')
plt.scatter(Iz_highres['r (mm)'], Iz_highres['Iz (A.U.)'], label='Li et al. (2021) Iz')
plt.scatter(Iz_needletip['r (mm)'], Iz_needletip['Iz (A.U.)'], label='Li et al. (2021) Needletip Iz')
for i in range(len(uz_fits_front)):
    plt.plot(-r_front * 1e3 + z0, alpha * cnst.q_e * n0 * uz_fits_front[i], label=f'Front Vortex {i+1}, $r_{{p}}$={rp_front*1e3:.0f} mm, $C_{{B,T}}$ = {cbts_front[i]:.2e} m')

for j in range (len(uz_fits_wake)):
    plt.plot(r_wake * 1e3 + z0, alpha * cnst.q_e * n0 * uz_fits_wake[j], label=f'Wake Vortex {j+1}, $r_{{p}}$={rp_wake*1e3:.0f} mm, $C_{{B,T}}$ = {cbts_wake[j]:.2e} m')

for k in range (len(uz_fits_needletip)):
    plt.plot(-r_needletip * 1e3 + z0_needletip, alpha * cnst.q_e * n0 * uz_fits_needletip[k], label=f'Needletip Vortex {k+1}, $r_{{p}}$={rp_needletip*1e3:.0f} mm, $C_{{B,T}}$ = {cbts_needletip[k]:.2e} m')

plt.xlabel('r (mm)')
plt.ylabel('Intensity (A.U.)')
plt.title(f'Vortex fits to Li et al. (2021) intensity, u0 = {u0:.2e} m/s, n0 = {n0} $m^{{-3}}$')
plt.legend()

# plt.figure()
# plt.plot(Ix['r (mm)'], Ix['Ix (A.U.)'], label='Ix')
# plt.xlabel('r (mm)')
# plt.ylabel('Intensity (A.U.)')
# plt.title('Light emission profiles from Li et al. 2021 Figure 3')
# plt.legend()

# CALCULATE RRMSE
rmin_front = z0 - r_front.max() * 1e3
rmax_front = z0
print(f'rmin_front = {rmin_front} mm, rmax_front = {rmax_front} mm')
front_mask = (Iz_highres['r (mm)'].values >= rmin_front) & (Iz_highres['r (mm)'].values <= rmax_front)
Iz_highres_front = Iz_highres[front_mask]
for i in range(len(uz_fits_front)):
    Ifront = alpha * cnst.q_e * n0 * uz_fits_front[i]
    model_x = (-r_front * 1e3 + z0)[::-1] # reverse from descending to ascending order
    model_y = Ifront[::-1]
    exp_y = Iz_highres_front['Iz (A.U.)'].values
    Ifront_interp = np.interp(Iz_highres_front['r (mm)'].values, model_x, model_y)
    rrmse_front = np.sqrt(mean_squared_error(exp_y, Ifront_interp)) / (exp_y.max() - exp_y.min())
    print(f'RRMSE for front vortex {i+1}: {rrmse_front}')


rmin_wake = z0 
rmax_wake = z0 + r_wake.max() * 1e3
print(f'rmin_wake = {rmin_wake} mm, rmax_wake = {rmax_wake} mm')
wake_mask = (Iz_highres['r (mm)'].values >= rmin_wake) & (Iz_highres['r (mm)'].values <= rmax_wake)
Iz_highres_wake = Iz_highres[wake_mask]
for j in range(len(uz_fits_wake)):
    Iwake = alpha * cnst.q_e * n0 * uz_fits_wake[j]
    model_x = (r_wake * 1e3 + z0)
    model_y = Iwake
    exp_y = Iz_highres_wake['Iz (A.U.)'].values
    Iwake_interp = np.interp(Iz_highres_wake['r (mm)'].values, model_x, model_y)
    rrmse_wake = np.sqrt(mean_squared_error(exp_y, Iwake_interp)) / (exp_y.max() - exp_y.min())
    print(f'RRMSE for wake vortex {j+1}: {rrmse_wake}')

rmin_needletip = z0_needletip - r_needletip.max() * 1e3
rmax_needletip = z0_needletip
print(f'rmin_needletip = {rmin_needletip} mm, rmax_needletip = {rmax_needletip} mm')
needletip_mask = (Iz_needletip['r (mm)'].values >= rmin_needletip) & (Iz_needletip['r (mm)'].values <= rmax_needletip)
Iz_highres_needletip = Iz_needletip[needletip_mask]
for k in range(len(uz_fits_needletip)):
    Ineedletip = alpha * cnst.q_e * n0 * uz_fits_needletip[k]
    model_x = (-r_needletip * 1e3 + z0_needletip)[::-1]
    model_y = Ineedletip[::-1]
    Ineedletip_interp = np.interp(Iz_highres_needletip['r (mm)'].values, model_x, model_y)
    exp_y = Iz_highres_needletip['Iz (A.U.)'].values
    rrmse_needletip = np.sqrt(mean_squared_error(exp_y, Ineedletip_interp)) / (exp_y.max() - exp_y.min())
    print(f'RRMSE for needletip vortex {k+1}: {rrmse_needletip}')


plt.show()