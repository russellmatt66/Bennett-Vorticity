"""
Build a chain of pureflow to negbulk vortices that have sufficient saturation span to bring the magnetic field back to zero 
in the inter-vortice regions.
"""
import sys
import pathlib
# ensure project root is on sys.path so the sibling `modules` package is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from modules import cubic_pureflow_module as cpfm
from modules import constants as cnst

import numpy as np
import matplotlib.pyplot as plt

num_phi = 200
phi = np.linspace(0, 2, num_phi)

phi_p = 1.0 # phi_p = rp / cbt constrains the problem but not really the plasma state  
beta_L = phi_p ** -1

# uz0_over_u0 = 0.5 * (beta_L + 1)**2 # Boundary condition for transition from pure to negative bulk flow
uz0_over_u0 = 1.0 

# Calculate vortex chain
# uz_list = []
num_vortices = 2
uz = np.zeros(num_phi)

uz[:num_phi//2] += cpfm.uz_chi2cubic_norm(phi[:len(phi)//2])
uz[num_phi//2:] += cpfm.uz_chi2cubic_negbulk_norm_SHIFT(phi[len(phi)//2:], uz0_over_u0, phi_p)

# uz = np.zeros(int(0.5 * num_phi))

# uz = cpfm.uz_chi2cubic_norm(phi)[:len(uz)]


# Plot
plt.plot(phi, uz)
plt.show()