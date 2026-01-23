import numpy as np
import constants as cnst
'''
Spitzer resistivity et. al
'''

def edgeKappaPerp_spitzer(n0: float, Tp: float, Bmax: float, lambda_C: float) -> float:
    '''
    Perpendicular thermal conductivity at the edge
    Units: [W/m/K]
    o n0 - Edge plasma density (m^-3)
    o Tp - Edge plasma temperature (K)
    o Bmax - Edge magnetic field (T)
    o lambda_C - Coulomb logarithm
    '''
    coeff1 = 4.7 * lambda_C * np.sqrt(cnst.me) * cnst.q_e**2 
    denom1 = 6 * np.sqrt(2) * np.pi**(3/2) * cnst.eps0**2
    coeff2 = n0**2 / Bmax**2 
    denom2 = (cnst.kB * Tp)**0.5 
    return coeff1 / denom1 * coeff2 / denom2  