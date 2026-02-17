from . import constants as cnst

import numpy as np

def omega_ce(B: float) -> float:
    '''
    Electron cyclotron frequency
    Units: [rad/s]
    o B - Magnetic field (T)
    '''
    return cnst.q_e * B / cnst.me 

def omega_ci(B: float, Z: float, mj: float) -> float:
    '''
    Ion (Hydrogen) cyclotron frequency
    Units: [rad/s]
    o B - Magnetic field (T)
    o Z - Ionization state
    o mj - Mass of the ion (kg)
    '''
    return Z * cnst.q_e * B / mj

def vA(B: float, n: float) -> float:
    '''
    Alfvén velocity
    Units: [m/s]
    o B - Magnetic field (T)
    o n - Plasma density (m^-3)
    '''
    return B / np.sqrt(cnst.mu0 * cnst.mH * n)

def vth(T: float, mj: float) -> float:
    '''
    Thermal velocity
    Units: [m/s]
    o T - Temperature (K)
    o mj - Mass of the particle (kg)
    '''
    return np.sqrt(2 * cnst.kB * T / mj)