import numpy as np
import constants as cnst
import spitzer as spz

'''
Calculating relevant physical properties for pureflow (\chi = 2) cubic vortices
'''

def cbt(n0: float, uz0: float, rp: float, Tp: float) -> float:
    '''
    C_{B,T}^{(3)} = mu0 * e**2 / (16 * kB) * (n0 * uz0**2 * rp**3) / Tp
    Units: [m]
    '''
    coeff1 = cnst.mu0 * cnst.q_e**2 / (16 * cnst.kB)
    coeff2 = (n0 * uz0**2 * rp**3) / Tp
    return coeff1 * coeff2

def uz(cbt: float, uz0: float, r: np.ndarray) -> np.ndarray:
    '''
    Velocity profile uz(r) for cubic pureflow vortex
    UnitsL [m/s]
    o cbt - The vortex constant [m]
    o uz0 - Edge flow constant [m/s]
    o r - Radial positions (m)
    '''
    term1 = r**2 / (r + cbt)**2
    return uz0 * term1

def f(cbt: float, r: np.ndarray) -> np.ndarray:
    '''
    Logarithmic constituent function for magnetic field profile in cubic pureflow vortex
    Units: [m^3]
    '''
    term1 = r**3
    term2 = 3 * r**2 * cbt
    term3 = 6 * r * cbt**2
    term4 = 6 * cbt**3
    return term1 - term2 - term3 * (1 + np.log(cbt / (r + cbt))) - term4 * np.log(cbt / (r + cbt)) 

def btheta(cbt: float, uz0: float, n0: float, r: np.ndarray) -> np.ndarray:
    '''
    Magnetic field profile btheta(r) for cubic pureflow vortex
    Units: [T]
    '''
    term1 = cnst.mu0 * cnst.q_e * uz0 * n0
    term2 = f(cbt, r) / (2.0 * r * (r + cbt))
    return -term1 * term2

def p0(cbt: float, n0: float, uz0: float, rp: float) -> float:
    '''
    Core plasma pressure for cubic pureflow vortex
    Units: [Pa]
    '''
    outfront = 0.25 * cnst.mu0 * (cnst.q_e * n0 * uz0)**2 / (rp + cbt)**2
    P_0_II = (-13 + 2*np.log(cbt)**2 + 6*np.log(rp+cbt) + 2*np.log(rp + cbt)**2 
                - 2*np.log(cbt) * (3 + 2*np.log(rp + cbt)))
    P_0_I = (-5 + 2*np.log(cbt)**2 + 8*np.log(rp + cbt) + 2*np.log(rp + cbt)**2 
                - 4*np.log(cbt) * (2 + np.log(rp + cbt)))
    P_0_0 = np.log(cbt)**2 + np.log(rp + cbt)*(5 + np.log(rp + cbt)) - np.log(cbt)*(5 + 2*np.log(rp + cbt))
    return outfront * (rp**4 - 10*rp**3*cbt + 3*rp**2*cbt**2 * P_0_II + 6*rp*cbt**3 * P_0_I + 6*cbt**4 * P_0_0)

def tauE(p0: float, uz0: float, rp: float, Tp: float, kappa_perp: float) -> float:
    '''
    Energy confinement time for a cubic pureflow vortex
    Units: [s]
    o p0 - Core plasma pressure (Pa)
    o cbt - The vortex constant (m)
    o n0 - Edge plasma density (m^-3)
    o uz0 - Edge flow velocity (m/s)
    o rp - Vortex radius (m)
    o Tp - Edge plasma temperature (K)
    o lambda_C - Coulomb logarithm
    '''
    # Bmax = btheta(cbt, uz0, n0, rp)
    # kappa_perp = spz.edgeKappaPerp_spitzer(n0, Tp, Bmax, lambda_C)
    return (3.0 / 12.0) * p0 * rp**2 / (kappa_perp * Tp)