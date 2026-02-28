import sys
import pathlib
# ensure project root is on sys.path so the sibling `modules` package is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from modules import constants as cnst
from modules import spitzer as spz
from modules import cubic_pureflow_module as cpfm
from modules import plasma_properties as pp
from modules import powerbalance as pb

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import mean_squared_error

uz_data = pd.read_csv('../../experimental_data/li_2021/li2021_airplasmastreamer_fig3.csv', header=0, skiprows=[1])

# print(uz_data.head())

uz_Iz = pd.DataFrame({
    'r (mm)': uz_data['Figure3_Iz'],
    'Iz (A.U.)' : uz_data['Unnamed: 1'], # not currents, but intensities of light emission
    # 'name' : 'Iz'
}).dropna().sort_values(by='r (mm)', ascending=True)

uz_Ix = pd.DataFrame({
    'r (mm)': uz_data['Figure3_Ix'],
    'Ix (A.U.)' : uz_data['Unnamed: 3'],
    # 'name' : 'Ix'
}).dropna().sort_values(by='r (mm)', ascending=True) 

print(uz_Iz.head())
print(uz_Ix.head())

uz_needletip_data = pd.read_csv('../../experimental_data/li_2021/Figure3_Iz_Needletip.csv', header=0, skiprows=[1])

# print(uz_needletip_data.head())

uz_needletip = pd.DataFrame({
    'r (mm)': uz_needletip_data.iloc[:, 0], # Assuming the first column is the spatial dimension
    'Iz (A.U.)' : uz_needletip_data.iloc[:, 1], # Assuming the second column is the intensity
    # 'name' : 'Iz'
}).dropna().sort_values(by='r (mm)', ascending=True)
    
print(uz_needletip.head())


