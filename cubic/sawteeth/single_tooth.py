import sys
import pathlib
# ensure project root is on sys.path so the sibling `modules` package is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from modules import constants as cnst
from modules import spitzer as spz
from modules import cubic_pureflow_module as cpfm
from modules import plasma_properties as pp

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

""" 
Vortices aren't matching at the sawtooth peak - need to figure out why, missing factor of two?
"""

phi_p = 1.0
beta_L = phi_p ** -1

num_phi = 500
phi = np.linspace(0, 2.0 * phi_p, num_phi) # phi = r / cbt
phi_up = phi[0:int(phi.size // 2)] # First half of the sawtooth period
phi_down = phi[int(phi.size // 2):] # Second half of the sawtooth period

uz0_over_u0 = 0.5 * (beta_L + 1)**2 # Boundary condition for transition from pure to negative bulk flow

uz_norm = cpfm.uz_chi2cubic_norm(phi_up)
uz_norm *= 1.0 / uz_norm.max() # Normalize to the maximum value of the pure flow cubic vortex
# uz_norm_bulkneg = uz_norm.max() * cpfm.uz_chi2cubic_negbulk_norm_SHIFT(phi_down, uz0_over_u0, phi_p) 
uz_norm_bulkneg = cpfm.uz_chi2cubic_negbulk_norm_SHIFT(phi_down, uz0_over_u0, phi_p) 

# triangle = 0.5 * (1.0 + signal.sawtooth(np.pi * phi, width=0.5))
triangle = np.zeros(phi.size)
triangle[0:int(phi.size // 2)] = phi_up / phi_p
triangle[int(phi.size // 2):] = 2.0 - phi_down / phi_p 
# triangle *= uz_norm.max() 

plt.plot(phi, triangle, label='Sawtooth Waveform')
plt.plot(phi_up, uz_norm, label='$\chi = 2$ Pure-flow Cubic Vortex')
plt.plot(phi_down, uz_norm_bulkneg, label='$\chi = 2$ Neg. Bulk Cubic Vortex')

plt.xlabel('$\\phi = r / cbt$')
plt.ylabel('$\\tilde{u}(\\phi)$')

plt.title('Two-Vortex chain sawtooth fit for $\phi_{p} = 1.0$')

plt.legend()

"""
Plot the associated density
"""
# plt.figure()

# n_norm_bulkneg = cpfm.n_chi2cubic_negbulk_norm_SHIFT(phi_down, uz0_over_u0, phi_p)

# plt.plot(phi_up, 1, label='Pure-flow Density')
# plt.plot(phi_down, n_norm_bulkneg, label='Neg. Bulk Density')

# plt.legend()

plt.show()