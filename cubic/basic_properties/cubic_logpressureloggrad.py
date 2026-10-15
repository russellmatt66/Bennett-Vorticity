import sys
import pathlib
# ensure project root is on sys.path so the sibling `modules` package is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt

'''
Plot d ln(p) / d ln(r) = d ln(p) / dr * (d ln(r) / r)^-1 = r * d ln(p) / dr 
= r / p * dp/dr
'''

# Calculate cubic pureflow
# Pressure
# Pressure gradient 

# Calculate cubic bulkflow 
# Pressure
# Pressure gradient