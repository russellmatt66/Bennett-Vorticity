"""
Calculates the equilibrium drifts
o Diamagnetic
o Grad-B
o ExB (non-uniform)
o Gravitational (Shell)
o Gravitational (Hot-dog: g_{0} = g_0\hat{z}_{0} = g_{0}\hat{x})
o Resistive
"""
import numpy as np

from . import constants as cnst
from . import plasma_properties as pp
from . import spitzer as spz

def diamagnetic_drift(uz: np.ndarray) -> np.ndarray:
    print('Calculating diamagnetic drift - it\'s just the velocity profile')
    return uz

def gradB_drift(mj: float, qj: float, btheta: np.ndarray, gradB: np.ndarray, uz: np.ndarray) -> np.ndarray:
    KE_perp = 0.5 * mj * (uz**2)  # Placeholder for actual perpendicular kinetic energy calculation
    numerator = KE_perp * gradB
    denominator = qj * btheta**2
    return numerator / denominator

def ExB_drift(mj: float, T: float, B: np.ndarray, Z: float, uz: np.ndarray, lapU: np.ndarray) -> np.ndarray:
    rhoL = pp.vth(T, mj) / pp.omega_ci(B, Z, mj) # Larmor radius
    return -(uz + rhoL**2 / 4 * lapU)

# Gravitational drift for the impact of self-gravity from the mass shell
def gravitational_drift_shell(mj: float, qj: float, n0: float, L: float, btheta: np.ndarray) -> np.ndarray:
    numerator = cnst.G * np.pi * mj * n0 * L 
    denominator = btheta * qj
    return numerator / denominator

# Gravitational drift for hot-dog geometry, where g0 is in the x-direction instead of z-direction
def gravitational_drift_hotdog(mj: float, qj: float, g0: float, theta: float, btheta: np.ndarray) -> np.ndarray:
    numerator = mj * g0 * np.cos(theta)
    denominator = btheta * qj
    return numerator / denominator

# Gravitational drift for waterfall geometry, where g0 is in the axial direction
def gravitational_drift_waterfall(mj: float, qj: float, g0: float, btheta: np.ndarray) -> np.ndarray:
    numerator = mj * g0
    denominator = btheta * qj
    return -numerator / denominator # points in negative r direction

def resistive_drift(Zj: float, n0: float, uz: np.ndarray, btheta: np.ndarray, T: np.ndarray, lambda_C: float) -> np.ndarray:
    numerator = cnst.q_e * n0 * uz * spz.eta_perp_spitzer(Zj, T, lambda_C)
    denominator = btheta
    return numerator / denominator