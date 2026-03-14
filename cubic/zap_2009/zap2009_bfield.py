"""
Stitching together the magnetic fields of the individual vortices does not respect Ampere's Law globally.
Therefore, the magnetic field must numerically solved for, using the current density from the fitted velocity profiles.
The fits are done in 'zap2009_sawtoothchain.py'.
"""
import sys
import pathlib
# ensure project root is on sys.path so the sibling `modules` package is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

# Read in the data 

# Solve for the magnetic field

# Plot the results