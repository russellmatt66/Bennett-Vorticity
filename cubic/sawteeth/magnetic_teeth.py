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