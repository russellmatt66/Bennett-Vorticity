"""
Stitching together the magnetic fields of the individual vortices does not respect Ampere's Law globally.
Therefore, the magnetic field must numerically solved for, using the current density from the fitted velocity profiles.
The fits are done in 'zap2009_sawtoothchain.py'.
"""
import sys
import pathlib
# ensure project root is on sys.path so the sibling `modules` package is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from modules import constants as cnst

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from scipy.integrate import cumulative_trapezoid

# Read in the data 
uzfit_neg0pt10 = pd.read_csv('../../analytic_fits/zap_2009/vortex_chain_fits/uz_tau_neg0pt10.csv')
uzfit_0pt10 = pd.read_csv('../../analytic_fits/zap_2009/vortex_chain_fits/uz_tau_0pt10.csv') 
uzfit_0pt16 = pd.read_csv('../../analytic_fits/zap_2009/vortex_chain_fits/uz_tau_0pt16.csv')
uzfit_0pt34 = pd.read_csv('../../analytic_fits/zap_2009/vortex_chain_fits/uz_tau_0pt34.csv')
uzfit_0pt56 = pd.read_csv('../../analytic_fits/zap_2009/vortex_chain_fits/uz_tau_0pt56.csv')

n0 = 1e22

# Solve for the magnetic field
def solve_bfield(uzfit: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve for the magnetic field using the current density from the fitted velocity profiles.
    """
    # Placeholder for the magnetic field solve
    # This will involve numerically integrating Ampere's Law using the current density derived from the velocity fits
    # The current density can be calculated as J = n * q * u, where n is the plasma density, q is the charge, and u is the velocity
    # Then, Ampere's Law can be used to find the magnetic field B from the current density J
    
    # Positive half-chord 
    r_pos = uzfit['r (mm)'].to_numpy()[uzfit['r (mm)'] > 0] * 1e-3 # Convert to meters
    uz_pos = uzfit['uz_root1 (km/s)'].to_numpy()[uzfit['r (mm)'] > 0] * 1e3 # Convert to m/s
    J_pos = n0 * cnst.q_e * uz_pos # Current density [A/m^2]
    Iencl_pos = 2 * np.pi * cumulative_trapezoid(J_pos * r_pos, r_pos, initial=0) # Enclosed current [A]
    B_pos = cnst.mu0 * Iencl_pos / (2 * np.pi * r_pos) # Magnetic field [T]
    B_pos = np.insert(B_pos, 0, 0) # Insert B=0 at r=0 for plotting
    r_pos = np.insert(r_pos, 0, 0) # r=0 at the beginning of the positive half-chord for correct ordering

    # Negative half-chord
    r_neg = np.abs(uzfit['r (mm)'].to_numpy()[uzfit['r (mm)'] < 0]) * 1e-3 # Take absolute value of radius for negative half-chord and convert to meters
    r_neg = r_neg[::-1] # Reverse the order of r_neg for correct ordering from center outwards
    uz_neg = uzfit['uz_root1 (km/s)'].to_numpy()[uzfit['r (mm)'] < 0] * 1e3 # Convert to m/s
    J_neg = n0 * cnst.q_e * uz_neg # Current density [A/m^2]
    Iencl_neg = 2 * np.pi * cumulative_trapezoid(J_neg * r_neg, r_neg, initial=0) # Enclosed current [A]
    B_neg = cnst.mu0 * Iencl_neg / (2 * np.pi * r_neg) # Magnetic field [T]
    # B_neg = np.insert(B_neg, len(B_neg), 0) # Insert B=0 at r=0 for plotting
    # r_neg = np.insert(r_neg, len(r_neg), 0) # r=0 at the end of the negative half-chord for correct ordering

    B_res = np.concatenate((B_neg[::-1], B_pos)) # Combine negative and positive half-chords, reversing the negative half for correct ordering
    r_res = np.concatenate((-r_neg[::-1], r_pos)) # Combine negative and positive half-chords, reversing the negative half for correct ordering and negating the radius for the negative half
    return B_res, r_res

# Plot the results
plt.plot(solve_bfield(uzfit_neg0pt10)[1] * 1e3, solve_bfield(uzfit_neg0pt10)[0], label='tau = -0.10')

plt.show()