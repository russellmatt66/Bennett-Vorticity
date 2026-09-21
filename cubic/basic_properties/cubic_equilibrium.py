import sys
import pathlib
# ensure project root is on sys.path so the sibling `modules` package is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import cumulative_trapezoid
'''
Plot the normalized equilibrium
'''
phi = np.logspace(-3, 3, 500) # r / C_{B,T}

uz_tilde = phi**2 / (phi + 1)**2
uzprime_tilde = phi / (phi + 1)**3
btheta_tilde = (1.0) / (phi * (phi + 1)) * (phi**3 - 3*phi**2 - 6*phi + 6*phi* np.log(1 + phi) + 6*np.log(phi + 1))  # normalized B_theta profile - btheta_tilde = B_theta / B_theta_max
Er_tilde = phi / (phi + 1)**3 * (phi**3 - 3*phi**2 - 6*phi*(1 - np.log(phi + 1)) + 6*np.log(phi + 1))

# Numerically integrate normalized plasma pressure
dptilde_dtheta = -(phi) / (phi + 1)**3 * (phi**3 - 3*phi**2 - 6*phi*(1 - np.log(phi + 1)) + 6*np.log(phi + 1))

# ptilde = np.cumsum(dptilde_dtheta) * (phi[1] - phi[0])  # simple numerical integration using the trapezoidal rule
ptilde = cumulative_trapezoid(dptilde_dtheta, phi, initial=0.0)
ptilde += np.abs(ptilde[-1]) # Add the core pressure 

fig, axs = plt.subplots(3, 2)

fig.suptitle('Normalized Cubic Bennett Vortex Equilibrium Profiles', fontsize=24)

axs[0, 0].plot(phi, uz_tilde)
axs[0, 0].set_ylabel('$\\tilde{u}_z$',fontsize=18)

axs[0, 1].plot(phi, uzprime_tilde)
axs[0, 1].set_ylabel("$\\tilde{u}_z'$", fontsize=18)

axs[1, 0].plot(phi, btheta_tilde)
axs[1, 0].set_ylabel('$\\tilde{B}_\\theta$', fontsize=18)

axs[1, 1].plot(phi, dptilde_dtheta)
axs[1, 1].set_ylabel('$d\\tilde{p}/d\\theta$', fontsize=18)

axs[2, 0].plot(phi, Er_tilde)
axs[2, 0].set_xlabel('$\\phi$', fontsize=18)
axs[2, 0].set_ylabel('$\\tilde{E}_r$', fontsize=18)

axs[2, 1].plot(phi, ptilde)
axs[2, 1].set_xlabel('$\\phi$', fontsize=18)
axs[2, 1].set_ylabel('$\\tilde{p}$', fontsize=18)

for ax in axs.flat:
    ax.set_xscale('log')

plt.show()