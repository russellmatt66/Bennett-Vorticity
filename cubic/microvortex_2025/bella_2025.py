import sys
import pathlib
# ensure project root is on sys.path so the sibling `modules` package is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

'''
Calculate solutions for the water microvortex intensity profile from Balcazar et. al (2025):
https://www.nature.com/articles/s41467-025-67224-3
'''