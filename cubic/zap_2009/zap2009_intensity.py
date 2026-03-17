import sys
import pathlib
# ensure project root is on sys.path so the sibling `modules` package is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pandas as pd

CIII_df = pd.read_csv('../../experimental_data/zap_2009/zap2009_CIII_intensity.csv')
electron_df = pd.read_csv('../../experimental_data/zap_2009/zap2009_electron_intensity.csv')