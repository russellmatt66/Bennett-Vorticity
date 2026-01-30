from numpy import pi

mu0 = 4 * pi * 1e-7 # Permeability of free space (H/m)
kB = 1.380649e-23 # Boltzmann constant (J/K)
q_e = 1.602176634e-19 # Elementary charge (C)
me = 9.10938356e-31 # Electron mass (kg)
mH = 1.6735575e-27 # Hydrogen ion mass (kg)
eps0 = 8.854187817e-12 # Permittivity of free space (F/m)
c = 3.0e8 # speed of light [m/s]
h = 6.626e-34 # Planck constant [J / Hz]

eV_to_K = 1.16045e4 # Conversion factor from eV to K
K_to_eV = 1 / eV_to_K # Conversion factor from K to eV

keV_to_K = eV_to_K * 1e-3
K_to_keV = 1 / keV_to_K