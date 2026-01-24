import numpy as np
from . import constants as cnst
'''
Spitzer resistivity et. al
'''
# electron collision time
def tau_e(n0: float, Tp: float, lambda_C: float) -> float:
    '''
    Electron-electron collision time
    Units: [s]
    '''
    # This is from NRL Plasma Formulary
    numer = 3 * np.sqrt(cnst.me) * (cnst.kB * Tp)**(1.5)
    denom = 4 * np.sqrt(2 * np.pi) * n0 * cnst.q_e**4 * lambda_C
    return numer / denom

# ion collision time
def tau_i(n0: float, Tp: float, lambda_C: float) -> float:
    '''
    Ion (Hydrogen) collision time
    Units: [s]
    '''
    # This is from NRL Plasma Formulary
    numer = 3 * np.sqrt(cnst.mH) * (cnst.kB * Tp)**(1.5)
    denom = 4 * np.sqrt(np.pi) * n0 * cnst.q_e**4 * lambda_C
    return numer / denom

# ee collisions
def KappaPerp_spitzer_e(n0: float, Tp: float, omega_ce: float, taue: float, lambda_C: float) -> float:
    '''
    Perpendicular thermal conductivity at the edge
    Units: [W/m/K]
    o n0 - Edge plasma density (m^-3)
    o Tp - Edge plasma temperature (K)
    o omega_ce - Electron cyclotron frequency (rad/s)
    o taue - Electron collision time (s)
    o lambda_C - Coulomb logarithm
    '''
    # This is from NRL Plasma Formulary
    numer = 4.7 * n0 * cnst.kB * Tp
    denom = cnst.me * omega_ce**2 * taue
    return numer / denom

# For Hydrogen ions
def KappaPerp_spitzer_i(n0: float, Tp: float, Z: float, omega_ci: float, taui: float, lambda_C: float) -> float:
    '''
    Perpendicular thermal conductivity at the edge
    Units: [W/m/K]
    o n0 - Edge plasma density (m^-3)
    o Tp - Edge plasma temperature (K)
    o Z - Ionization state
    o omega_ci - Ion cyclotron frequency (rad/s)
    o taui - Ion collision time (s)
    o lambda_C - Coulomb logarithm
    '''
    # This is from NRL Plasma Formulary
    numer = 2.0 * n0 * cnst.kB * Tp
    denom = cnst.mH * omega_ci**2 * taui
    return numer / denom

def coulombLog_ee(n0: float, Tp: float) -> float:
    '''
    Coulomb logarithm
    Units: dimensionless
    o n0 - Edge plasma density (m^-3)
    o Tp - Edge plasma temperature (K)
    '''
    if Tp * cnst.K_to_eV < 10:
        return 23 - np.log(n0**0.5 * Tp**-1.5)
    else:
        return 24 - np.log(n0**0.5 * Tp**-1)

def coulombLog_ei(n0: float, Tp: float, Z: float) -> float:
    '''
    Coulomb logarithm
    Units: dimensionless
    o n0 - Edge plasma density (m^-3)
    o Tp - Edge plasma temperature (K)
    o Z - Ionization state
    '''
    if Tp * cnst.K_to_eV < 10 * Z**2:
        return 23 - np.log(n0**0.5 * Tp**-1.5) + np.log(Z)
    else:
        return 24 - np.log(n0**0.5 * Tp**-1)