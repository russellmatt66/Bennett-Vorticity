'''
Studying the effective temperature when a vortex has both fusion reactions and bremsstrahlung present
- Actually, just put this in `fuzelike_tauE.py`
'''
import sys
import pathlib
# ensure project root is on sys.path so the sibling `modules` package is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from modules import powerbalance
