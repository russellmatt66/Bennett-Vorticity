"""
Class for fitting a single vortex to experimental data
"""
from . import cubic_pureflow_module as cpfm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Vortex:
    def __init__(self, n0, Tp, uedge, u0, rp, num_r=100):
        self.n0 = n0
        self.Tp = Tp
        self.uedge = uedge
        self.u0 = u0
        self.rp = rp # Pinch radius [m]
        self.num_r = num_r 
        self.r = np.linspace(0, rp, num_r) # Radial grid for fitting
        # Initialize the necessary lists
        self.cbts = []
        self.uz_fits = []
        self.uz0_roots = []
        self.uz_df = []

    # Load data to fit 
    def load_data(self, filepath):
        uz_data = pd.read_csv(filepath)
        self.uz_df.append(uz_data)
        pass

    # Fit 
    def fit_chi2_pureflow(self):
        cbts_temp = []
        uz_temp = []
        uz0_temp = cpfm.root_solve_chi2_pure(self.uedge, self.u0, self.n0, self.rp, self.Tp)
        if uz0_temp is not None:
            self.uz0_roots.append(uz0_temp)
            for uz0 in uz0_temp:
                cbt_temp = cpfm.cbt(self.n0, np.abs(uz0), self.rp, self.Tp)
                cbts_temp.append(cbt_temp)
                uz_temp.append(cpfm.uz_chi2cubic(cbt_temp, np.abs(uz0), self.u0, self.r))
                self.uz_fits.append(uz_temp)
            self.cbts.append(cbt_temp)
            self.uz_fits.append(uz_temp)
        pass
    
    def fit_chi2_posbulk(self): 
        cbts_temp = []
        uz_temp = []
        uz0_temp = cpfm.root_solve_chi2_posbulk(self.uedge, self.u0, self.n0, self.rp, self.Tp)
        if uz0_temp is not None:
            self.uz0_roots.append(uz0_temp)
            for uz0 in uz0_temp:
                cbt_temp = cpfm.cbt(self.n0, np.abs(uz0), self.rp, self.Tp)
                cbts_temp.append(cbt_temp)
                uz_temp.append(cpfm.uz_chi2cubic_posbulk(cbt_temp, np.abs(uz0), self.u0, self.r))
                self.uz_fits.append(uz_temp)
            self.cbts.append(cbt_temp)
            self.uz_fits.append(uz_temp)
        pass

    def fit_chi2_negbulk(self):
        cbts_temp = []
        uz_temp = []
        uz0_temp = cpfm.root_solve_chi2_negbulk(self.uedge, self.u0, self.n0, self.rp, self.Tp)
        if uz0_temp is not None:
            self.uz0_roots.append(uz0_temp)
            for uz0 in uz0_temp:
                cbt_temp = cpfm.cbt(self.n0, np.abs(uz0), self.rp, self.Tp)
                cbts_temp.append(cbt_temp)
                uz_temp.append(cpfm.uz_chi2cubic_negbulk(cbt_temp, np.abs(uz0), self.u0, self.r))
                self.uz_fits.append(uz_temp)
            self.cbts.append(cbts_temp)
            self.uz_fits.append(uz_temp)
        pass

    # # Plot - This concern should be separated. 
    # def plot(self):
    #     for i in range(len(self.uz_fits)):
    #         plt.figure()
    #         # plt.plot(uz_df['Radius (mm)'], uz_df['uz (10^{4} m / s)'], 'ro', label='Experimental data') # 
    #         # plt.title()
    #         for j in range(len(self.uz_fits[i])):
    #             plt.plot(self.r, self.uz_fits[i][j], label=f'Root {j+1}')
            
    #         # plt.xlabel('Radius (mm)')
    #         plt.legend()
    #     pass

    # Save results
    def save(self, filepath):
        df = {'Radius (m)': self.r}

        for i in range(len(self.uz_fits)):
            for j in range(len(self.uz_fits[i])):
                df[f'uz_fit_root{j+1} (m/s)'] = self.uz_fits[i][j]

        pd.DataFrame(df).to_csv(filepath, index=False)        
        pass

    