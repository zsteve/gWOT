import numpy as np

class TimeSeries():
    def __init__(self, x, dt, t_idx, D = None):
        self.x = x
        self.t_idx = t_idx
        self.T = len(np.unique(t_idx))
        self.N = np.unique(t_idx, return_counts = True)[1]
        self.D = D
        self.dt = dt
