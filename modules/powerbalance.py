import numpy as np

from . import constants as cnst 
from . import spitzer as spz

'''
Bosch-Hale 
- n = vortex power law exponent
'''
class DDFusionCalculator:
    def __init__(self, n: int):
        # Constants
        self.mrc2 = 937814.0 # Reduced mass energy in keV
        
        # Energy Yields (converted to Joules)
        # 1 eV = 1.60218e-19 J
        MeV_to_J = 1.60218e-13
        self.E_p_branch = 4.03 * MeV_to_J # alpha energies
        self.E_n_branch = 0.82 * MeV_to_J

        # Bosch-Hale
        # Coefficients for D + D -> T + p
        self.coeffs_p = {
            'BG': 31.3970,
            'C1': 5.65718e-12, 'C2': 3.41267e-3, 'C3': 1.99167e-3,
            'C4': 0.0,         'C5': 1.05060e-5, 'C6': 0.0,
            'C7': 0.0
        }

        # Coefficients for D + D -> He3 + n
        self.coeffs_n = {
            'BG': 31.3970,
            'C1': 5.43360e-12, 'C2': 5.85778e-3, 'C3': 7.68222e-3,
            'C4': 0.0,         'C5': -2.96400e-6,'C6': 0.0,
            'C7': 0.0
        }

        self.vple = n # vortex power law exponent

    def _compute_sigmav(self, T_kev, C):
        """Internal helper to calculate one branch."""
        if T_kev < 0.2: return 0.0
        
        # Calculate Theta (Padé approximant)
        # Formula: T / (1 - (T*(C2 + T*(C4 + T*C6))) / (1 + T*(C3 + T*(C5 + T*C7))))
        numerator_term = T_kev * (C['C2'] + T_kev * (C['C4'] + T_kev * C['C6']))
        denominator_term = 1 + T_kev * (C['C3'] + T_kev * (C['C5'] + T_kev * C['C7']))
        
        theta = T_kev / (1 - (numerator_term / denominator_term))
        
        # Calculate Xi
        xi = (C['BG']**2 / (4 * theta))**(1/3)
        
        # Calculate Reactivity (cm^3/s)
        sigmav = C['C1'] * theta * np.sqrt(xi / (self.mrc2 * T_kev**3)) * np.exp(-3 * xi)
        # Convert cm^3/s to m^3/s
        return sigmav * 1.0e-6 

    def get_reactivities(self, T_kev):
        """Returns tuple: (sigmav_proton_branch, sigmav_neutron_branch) in m^3/s"""
        sv_p = self._compute_sigmav(T_kev, self.coeffs_p)
        sv_n = self._compute_sigmav(T_kev, self.coeffs_n)
        return sv_p, sv_n

    def get_power_density(self, n0: float, Tp_keV: float) -> float:
        """ [W / m^{-3}]"""
        sv_p, sv_n = self.get_reactivities(Tp_keV)

        # [W / m^{-3}]
        power_p = sv_p * self.E_p_branch
        power_n = sv_n * self.E_n_branch

        power_density = 0.25 * n0**2 * (power_p + power_n)
        return power_density

    def get_sb(self, n0: float, Tp_keV: float) -> float:
        '''
        Power density of bremsstrahlung losses
        Zeff=1
        '''
        term1 = (1 / (3 * np.pi**2)) * np.sqrt(2 / np.pi)
        term2 = cnst.q_e**6 / (cnst.eps0**3 * cnst.c**3 * cnst.h * cnst.me**(3/2))
        return term1 * term2 * n0**2 * Tp_keV**(0.5)

    def Teff_e(self, n0: float, rp: float, Tp: float, omega_ce: float, taue: float, lambda_C: float) -> float:
        """
        This is an effective vortex temperature which is based on a balance of fusion energy power with bremsstrahlung, and thermal conduction losses of the edge vortex 
        """
        kappa_perp_e = spz.KappaPerp_spitzer_e(n0, Tp, omega_ce, taue, lambda_C)
        outfront = rp**2 / (2 * self.vple * kappa_perp_e)
        fusion_power_density = self.get_power_density(n0, Tp * cnst.K_to_keV) 
        brem_density = self.get_sb(n0, Tp * cnst.K_to_keV)
        return Tp-outfront * (fusion_power_density - brem_density)

    def T_vortex(self, n0: float, rp: float, Tp: float, omega_ce: float, taue: float, lambda_C: float) -> float:
        """
        This is an effective vortex temperature which is based on just a balance of fusion energy power with bremsstrahlung losses 
        """
        kappa_perp_e = spz.KappaPerp_spitzer_e(n0, Tp, omega_ce, taue, lambda_C)
        outfront = rp**2 / (2 * self.vple * kappa_perp_e)
        fusion_power_density = self.get_power_density(n0, Tp * cnst.K_to_keV) 
        brem_density = self.get_sb(n0, Tp * cnst.K_to_keV)
        return -outfront * (fusion_power_density - brem_density)