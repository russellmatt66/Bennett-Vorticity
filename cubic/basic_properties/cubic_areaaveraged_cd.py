import numpy as np 
import matplotlib.pyplot as plt

num_points = 10000
phi = np.logspace(-3, 3, num_points)

jz_areaaverage_norm = (phi**3 - 3 * phi**2 - 6*phi + 6*phi*np.log(phi+1) + 6*np.log(phi+1)) / (2 * phi**2 * (phi + 1))

plt.loglog(phi, jz_areaaverage_norm)
plt.xlabel("$\\phi = \\frac{r}{C}$", fontsize=24)
plt.ylabel("$\\frac{\\langle j_{z}\\rangle}{2 en_{0}u_{z,0}}$", fontsize=24)
plt.title("Area-averaged current density of the cubic Bennett-Shumlak-Hartman vortex", fontsize=24)
plt.show()