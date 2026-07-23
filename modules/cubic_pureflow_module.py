import numpy as np

from . import constants as cnst
from . import spitzer as spz

'''
Calculating objects for pureflow (\chi = 2) cubic vortices
'''
# Mixing length (?)
def cbt(n0: float, uz0: float, rp: float, Tp: float) -> float:
    '''
    C_{B,T}^{(3)} = mu0 * e**2 / (16 * kB) * (n0 * uz0**2 * rp**3) / Tp
    Units: [m]
    '''
    coeff1 = cnst.mu0 * cnst.q_e**2 / (16 * cnst.kB)
    coeff2 = (n0 * uz0**2 * rp**3) / Tp
    return coeff1 * coeff2

""" 
Velocity forms
"""
# u_{z}^{(2)}(r) = u_{z0} * r**2 / (r + cbt)**2
def uz_chi2cubic_pure(cbt: float, uz0: float, r: np.ndarray) -> np.ndarray:
    '''
    Velocity profile uz(r) for cubic pureflow vortex
    UnitsL [m/s]
    o cbt - The vortex constant [m]
    o uz0 - Edge flow constant [m/s]
    o r - Radial positions (m)
    '''
    term1 = r**2 / (r + cbt)**2
    return uz0 * term1

# u_{z}^{(2),-}(r) = (u_{0} - u_{z0} * r**2 / (r + cbt)**2)
def uz_chi2cubic_negbulk(cbt: float, uz0: float, u0: float, r: np.ndarray) -> np.ndarray:
    '''
    Velocity profile uz(r) for cubic pureflow vortex
    UnitsL [m/s]
    o cbt - The vortex constant [m]
    o uz0 - Edge flow constant [m/s]
    o u0 - Core flow constant [m/s]
    o r - Radial positions (m)
    '''
    term1 = r**2 / (r + cbt)**2
    return u0 - uz0 * term1

# u_{z}^{(2),+}(r) = (u_{0} + u_{z0} * r**2 / (r + cbt)**2)
def uz_chi2cubic_posbulk(cbt: float, uz0: float, u0: float, r: np.ndarray) -> np.ndarray:
    '''
    Velocity profile uz(r) for cubic pureflow vortex
    UnitsL [m/s]
    o cbt - The vortex constant [m]
    o uz0 - Edge flow constant [m/s]
    o u0 - Core flow constant [m/s]
    o r - Radial positions (m)
    '''
    term1 = r**2 / (r + cbt)**2
    return u0 + uz0 * term1

# \tilde{u}_{z}^{2}(\phi) = \phi**2 / (\phi + 1)**2 = u_{z}^{(2)}(r) / u_{z0}, \phi = r / cbt
def uz_chi2cubic_norm(phi: np.ndarray) -> np.ndarray:
    '''
    Normalized velocity profile uz(phi) for cubic pureflow vortex, where phi = r / cbt
    Units
    o phi - Normalized radial positions (dimensionless)
    '''
    term1 = phi**2 / (phi + 1)**2
    return term1

# Laplacian of the velocity profile for cubic pureflow vortex
def LapU_chi2cubic_pure(cbt: float, uz0: float, r: np.ndarray) -> np.ndarray:
    '''
    Laplacian of the velocity profile for cubic pureflow vortex
    Units: [s^-1]
    '''
    numerator = 2 * cbt * (2 * cbt - r) * uz0
    denominator = (r + cbt)**4
    return numerator / denominator

"""
Normalized Velocity forms
"""
def uz_chi2cubic_norm(phi: np.ndarray) -> np.ndarray:
    '''
    Normalized velocity profile uz(phi) for cubic pureflow vortex, where phi = r / cbt
    Units
    o phi - Normalized radial positions (dimensionless)
    '''
    term1 = phi**2 / (phi + 1)**2
    return term1

def uz_chi2cubic_norm_shift(phi: np.ndarray, phi_p: float) -> np.ndarray:
    '''
    Normalized velocity profile uz(phi) for cubic pureflow vortex, where phi = r / cbt
    Units
    o phi - Normalized radial positions (dimensionless)
    o phi_p - Normalized pinch radius (dimensionless)
    '''
    term1 = (phi - phi_p)**2 / (phi - phi_p + 1)**2
    return term1

# u_{z}^{(2),-}(phi) = (u_{0} - u_{z0} * phi**2 / (phi + 1)**2)
def uz_chi2cubic_negbulk_norm(phi: np.ndarray, uz0_over_u0: float) -> np.ndarray:
    '''
    Normalized velocity profile uz(phi) for cubic pureflow vortex with negative bulk flow, where phi = r / cbt
    Units
    o phi - Normalized radial positions (dimensionless)
    o uz0_over_u0 - Ratio of edge flow velocity to core flow velocity (dimensionless)
    '''
    term1 = uz0_over_u0 * phi**2 / (phi + 1)**2
    return 1.0 - term1

# u_{z}^{(2),-}(phi - phi_p) = (u_{0} - u_{z0} * (phi - phi_p)**2 / ((phi - phi_p) + 1)**2)
def uz_chi2cubic_negbulk_norm_SHIFT(phi: np.ndarray, uz0_over_u0: float, phi_p: float) -> np.ndarray:
    '''
    Normalized velocity profile uz(phi) for cubic pureflow vortex with negative bulk flow, where phi = r / cbt
    Units
    o phi - Normalized radial positions (dimensionless)
    o uz0_over_u0 - Ratio of edge flow velocity to core flow velocity (dimensionless)
    o phi_p - Normalized pinch radius (dimensionless)
    '''
    term1 = uz0_over_u0 * (phi - phi_p)**2 / (phi - phi_p + 1)**2
    return 1.0 - term1

""" 
Density forms
"""
# -e * n(r) = J_{z}^{(2,-)}(r) / u_{z}^{(2)}(r)
def n_chi2cubic_negbulk_norm(phi: np.ndarray, uz0_over_u0: float) -> np.ndarray:
    '''
    Normalized density profile n(phi) for cubic pureflow vortex with a negative bulk flow current
    o phi - Normalized radial positions (dimensionless)
    o uz0_over_u0 - Ratio of edge flow velocity to core flow velocity (dimensionless)
    '''
    term1 = (1.0 / uz0_over_u0) * (phi + 1)**2 / phi**2
    return term1 - 1.0

def n_chi2cubic_negbulk_norm_SHIFT(phi: np.ndarray, uz0_over_u0: float, phi_p: float) -> np.ndarray:
    '''
    Normalized density profile n(phi) for cubic pureflow vortex with a negative bulk flow current
    o phi - Normalized radial positions (dimensionless)
    o uz0_over_u0 - Ratio of edge flow velocity to core flow velocity (dimensionless)
    o phi_p - Normalized pinch radius (dimensionless)
    '''
    term1 = (1.0 / uz0_over_u0) * (phi - phi_p + 1)**2 / (phi - phi_p)**2
    return term1 - 1.0

""" 
Magnetic field forms
"""
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

def btheta_chi2(cbt: float, uz0: float, n0: float, r: np.ndarray) -> np.ndarray:
    '''
    Magnetic field profile btheta(r) for cubic pureflow vortex
    Units: [T]
    B_{theta}^{(2)}(r) = -mu0 * e * uz0 * n0 * f(cbt, r) / (2 * r * (r + cbt))
                    = \frac{mu0}{r} * \int_0^r' r' * J_z^{(2)}(r') dr' 
    '''
    term1 = cnst.mu0 * cnst.q_e * uz0 * n0
    term2 = f(cbt, r) / (2.0 * r * (r + cbt))
    return -term1 * term2

def btheta_chi2_negbulk(cbt: float, uz0: float, u0: float, n0: float, r: np.ndarray) -> np.ndarray:
    '''
    Magnetic field profile btheta(r) for cubic pureflow vortex with negative bulk flow
    Units: [T]
    B_{theta}^{(2),-}(r) = -mu0 * e * n0 * (u0 * r / 2 - uz0 * f(cbt, r) / (2 * r * (r + cbt)))
    '''
    term1 = cnst.mu0 * cnst.q_e * n0
    term2 = u0 * r / 2.0 - uz0 * f(cbt, r) / (2.0 * r * (r + cbt))
    return -term1 * term2

def btheta_chi2_posbulk(cbt: float, uz0: float, u0: float, n0: float, r: np.ndarray) -> np.ndarray:
    '''
    Magnetic field profile btheta(r) for cubic pureflow vortex with positive bulk flow
    Units: [T]
    B_{theta}^{(2),+}(r) = -mu0 * e * n0 * (u0 * r / 2 + uz0 * f(cbt, r) / (2 * r * (r + cbt)))
    '''
    term1 = cnst.mu0 * cnst.q_e * n0
    term2 = u0 * r / 2.0 + uz0 * f(cbt, r) / (2.0 * r * (r + cbt))
    return -term1 * term2

# dB/dr for cubic pureflow vortex
def gradbtheta_chi2cubic_pure(cbt: float, uz0: float, n0: float, r: np.ndarray) -> np.ndarray:
    '''
    Radial gradient of the magnetic field profile for cubic pureflow vortex
    Units: [T/m]
    '''
    term1 = cnst.mu0 * cnst.q_e * n0 * uz0
    term2 = r * (r**3 + 6*cbt**3 + 9 * cbt**2 * r + 2 * cbt * r**2) / (r + cbt)**2
    term3 = 6 * cbt**2 * np.log(cbt / (r + cbt))
    return -term1 * (term2 + term3) / (2 * r**2)

# Miscellaneous
# Pureflow pressure
def p0(cbt: float, n0: float, uz0: float, rp: float) -> float:
    '''
    Core plasma pressure for cubic pureflow vortex
    Units: [Pa]
    '''
    if cbt <= 0.0 or rp <= 0.0:
        raise ValueError("cbt and rp must be positive values.")

    P_0_II = (-13 + 2*np.log(cbt)**2 + 6*np.log(rp+cbt) + 2*np.log(rp + cbt)**2 
                - 2*np.log(cbt) * (3 + 2*np.log(rp + cbt)))
    
    P_0_I = (-5 + 2*np.log(cbt)**2 + 8*np.log(rp + cbt) + 2*np.log(rp + cbt)**2 
                - 4*np.log(cbt) * (2 + np.log(rp + cbt)))
    
    P_0_0 = np.log(cbt)**2 + np.log(rp + cbt)*(5 + np.log(rp + cbt)) - np.log(cbt)*(5 + 2*np.log(rp + cbt))

    outfront = 0.25 * cnst.mu0 * (cnst.q_e * n0 * uz0)**2 / (rp + cbt)**2

    return outfront * (rp**4 - 10*rp**3*cbt + 3*rp**2*cbt**2 * P_0_II + 6*rp*cbt**3 * P_0_I + 6*cbt**4 * P_0_0)

# IMPLEMENT
def p0_negbulk() -> float:
    pass 

# IMPLEMENT
def p0_posbulk() -> float:
    pass

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

def tauE_parabolic(p0: float, uz0: float, rp: float, T0: float, kappa_perp: float) -> float:
    '''
    Energy confinement time for a parabolic pureflow vortex
    Units: [s]
    o p0 - Core plasma pressure (Pa)
    o cbt - The vortex constant (m)
    o n0 - Edge plasma density (m^-3)
    o uz0 - Edge flow velocity (m/s)
    o rp - Vortex radius (m)
    o Tp - Edge plasma temperature (K)
    o lambda_C - Coulomb logarithm
    '''
    # Bmax = btheta_parabolic(cbt, uz0, n0, rp)
    # kappa_perp = spz.edgeKappaPerp_spitzer(n0, Tp, Bmax, lambda_C)
    return (3.0 / 8.0) * p0 * rp**2 / (kappa_perp * T0)

def root_solve_chi2_pure(uedge: float, n0: float, rp: float, Tp: float) -> np.ndarray:
    """
    Analytic solution for the roots of the fourth-order polynomial that arises from the chi=2 flow boundary condition for pureflow vortices.
    """
    A = (cnst.mu0 * cnst.q_e**2 * n0 * rp**3 / (16 * cnst.kB * Tp))**2
    B = (cnst.mu0 * n0 * cnst.q_e**2 * rp**4) / (16 * cnst.kB * Tp)
    
    coeffs = [A * uedge, 0, 2 * B * uedge, -rp**2, uedge*rp**2]

    uz0_roots = np.roots(coeffs)
    print(f"Roots of the chi=2 flow boundary condition polynomial: {uz0_roots}")
        # uz0_real = uz0_roots[np.isreal(uz0_roots)].real
        # print(f"Real roots of the chi=2 flow boundary condition polynomial: {uz0_real}")
    
    return uz0_roots

def root_solve_chi2_negbulk(uedge: float, u0: float, n0: float, rp: float, Tp: float) -> np.ndarray:
    """
    Analytic solution for the roots of the fourth-order polynomial that arises from the chi=2 flow boundary condition for 
    bulk, negative, pureflow vortices.
    """
    A = (cnst.mu0 * cnst.q_e**2 * n0 * rp**3 / (16 * cnst.kB * Tp))**2
    B = (cnst.mu0 * n0 * cnst.q_e**2 * rp**4) / (16 * cnst.kB * Tp)
    
    coeffs = [A * (uedge - u0), 0, 2 * B * (uedge - u0), rp**2, (uedge - u0)*rp**2]

    uz0_roots = np.roots(coeffs)

    print(f"Roots of the chi=2 flow boundary condition polynomial: {uz0_roots}")
    
    # uz0_real = uz0_roots[np.isreal(uz0_roots)].real
    # print(f"Real roots of the chi=2 flow boundary condition polynomial: {uz0_real}")

    return uz0_roots

def root_solve_chi2_posbulk(uedge: float, u0: float, n0: float, rp: float, Tp: float) -> np.ndarray:
    """
    Analytic solution for the roots of the fourth-order polynomial that arises from the chi=2 flow boundary condition for 
    bulk, positive, pureflow vortices.
    """
    A = (cnst.mu0 * cnst.q_e**2 * n0 * rp**3 / (16 * cnst.kB * Tp))**2
    B = (cnst.mu0 * n0 * cnst.q_e**2 * rp**4) / (16 * cnst.kB * Tp)
    
    coeffs = [A * (uedge - u0), 0, 2 * B * (uedge - u0), -rp**2, (uedge - u0)*rp**2]

    uz0_roots = np.roots(coeffs)

    print(f"Roots of the chi=2 flow boundary condition polynomial: {uz0_roots}")
    
    # uz0_real = uz0_roots[np.isreal(uz0_roots)].real
    # print(f"Real roots of the chi=2 flow boundary condition polynomial: {uz0_real}")

    return uz0_roots