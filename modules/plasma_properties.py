import constants as cnst

def omega_ce(B: float) -> float:
    '''
    Electron cyclotron frequency
    Units: [rad/s]
    o B - Magnetic field (T)
    '''
    return cnst.q_e * B / cnst.me 