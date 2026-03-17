import sys
import pathlib
# ensure project root is on sys.path so the sibling `modules` package is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from modules import vortex
from modules import constants as cnst

import numpy as np
import pandas as pd

CIII_df = pd.read_csv('../../experimental_data/zap_2009/zap2009_CIII_intensity.csv')
electron_df = pd.read_csv('../../experimental_data/zap_2009/zap2009_electron_intensity.csv')

n0 = 1e23 # Plasma density [m^-3]; 1e22 - 1e23
Tpe = 64 * cnst.eV_to_K # Electron temperature [K]; T = Te + Ti = 150 - 200 eV
Tpi = 71 * cnst.eV_to_K # Ion temperature [K]; T = Te + Ti = 150 - 200 eV

# Electron scattering intensity profile is posbulk to posbulk
min_idx = electron_df.iloc[:, 1].idxmin() # Find the index of the minimum intensity
electron_df.columns[0] -= electron_df.iloc[min_idx, 0] # Shift the radial positions so that the minimum is at r=0

rp_electron_pos = np.max(electron_df.columns[0].to_numpy()) # [nm]
rp_electron_neg = np.abs(np.min(electron_df.columns[0].to_numpy())) # [nm]

I0_electron = electron_df.iloc[:, 1].min() # Maximum intensity for normalization
Iedge_electron_pos = electron_df.iloc[:, 1].max() # Maximum intensity for BCs

neg_half = electron_df[electron_df.columns[0] < 0] # Select the negative half of the data
Iedge_electron_neg = neg_half.iloc[:, 1].max() # Maximum intensity for BCs

# Ion profile is negbulk b2b w/negbulk
max_idx_CIII = CIII_df.iloc[:, 1].idxmax() # Find the index of the maximum intensity
CIII_df.columns[0] -= CIII_df.iloc[max_idx_CIII, 0]

rp_ion_pos = np.max(CIII_df.columns[0].to_numpy()) # [nm]
rp_ion_neg = np.abs(np.min(CIII_df.columns[0].to_numpy())) # [nm]

I0_ion = CIII_df.iloc[:, 1].max() # Maximum intensity for normalization

neg_half_ion = CIII_df[CIII_df.columns[0] < 0] # Select the negative half of the data
Iedge_ion_neg = neg_half_ion.iloc[:, 1].min() # Minimum intensity for BCs

pos_half_ion = CIII_df[CIII_df.columns[0] > 0] # Select the positive half of the data
Iedge_ion_pos = pos_half_ion.iloc[:, 1].min() # Minimum intensity for BCs

# Convert from intensity to plasma current density
u0 = 80e3 # representative flow velocity for early in quiescent period; [m/s]
alpha_e = I0_electron / (cnst.q_e *n0 * u0)
alpha_ion = I0_ion / (cnst.q_e *n0 * u0) # Hydrogen plasma

# Create vortex objects for fitting
uedge_e_pos = Iedge_electron_pos / (alpha_e * cnst.q_e * n0) # Convert to m/s
uedge_e_neg = Iedge_electron_neg / (alpha_e * cnst.q_e * n0) # Convert to m/s

e_vortex_pos = vortex.Vortex(n0, Tp=Tpe, uedge=uedge_e_pos, u0=u0, rp=rp_electron_pos)
e_vortex_neg = vortex.Vortex(n0, Tp=Tpe, uedge=uedge_e_neg, u0=u0, rp=rp_electron_neg)

uedge_i_pos = Iedge_ion_pos / (alpha_ion * cnst.q_e * n0) # Convert to m/s
uedge_i_neg = Iedge_ion_neg / (alpha_ion * cnst.q_e * n0) # Convert to m/s

i_vortex_pos = vortex.Vortex(n0, Tp=Tpi, uedge=uedge_i_pos, u0=u0, rp=rp_ion_pos)
i_vortex_neg = vortex.Vortex(n0, Tp=Tpi, uedge=uedge_i_neg, u0=u0, rp=rp_ion_neg)