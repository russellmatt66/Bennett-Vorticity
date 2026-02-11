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

phi = np.linspace(0, 100, 500) # phi = r / cbt

triangle = 0.5 * (1.0 + signal.sawtooth(phi, width=0.5))

uz_norm = cpfm.uz_chi2cubic_norm(phi)


plt.plot(phi, triangle, label='Sawtooth Waveform')
plt.plot(phi, uz_norm, label='Normalized Flow Profile')

plt.xlabel('$\\phi = r / cbt$')
plt.ylabel('$u^{(2)}(\\phi)$')

plt.legend()

plt.show()