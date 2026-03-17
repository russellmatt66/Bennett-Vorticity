"""
Class for fitting a single vortex to experimental data
"""
class Vortex:
    def __init__(self, n0, Tp, uedge, u0, rp):
        self.n0 = n0
        self.Tp = Tp
        self.uedge = uedge
        self.u0 = u0
        self.rp = rp
    
    # Load data to fit 
    def load_data(self, filepath):
        pass

    # Fit 
    def fit(self, data):
        pass
    
    # Plot
    def plot(self):
        pass

    # Save results
    def save(self):
        pass