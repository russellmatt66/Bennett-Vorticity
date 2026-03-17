"""
Class for fitting a single vortex to experimental data
"""
class Vortex:
    def __init__(self, n0, Tp, uedge, u0, rp, num_r=100):
        self.n0 = n0
        self.Tp = Tp
        self.uedge = uedge
        self.u0 = u0
        self.rp = rp
        self.num_r = num_r
        # Initialize the necessary lists
        self.cbts = []
        self.uzpos_fits = []
        self.uzneg_fits = []
        self.uzpos_roots = []
        self.uzneg_roots = []
    
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