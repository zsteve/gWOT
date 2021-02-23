import numpy as np

class TimeSeries():
    """Base class for time-series dataset
    
    :param x: `np.array` of observed datapoints. 
    :param dt: `np.array` of time increments `t[i+1] - t[i]`.
    :param t_idx: `np.array` of time indices for each datapoint in `x`. 
    :param D: diffusivity 
    """
    def __init__(self, x, dt, t_idx, D = None):
        self.x = x
        self.t_idx = t_idx
        self.T = len(np.unique(t_idx))
        self.N = np.unique(t_idx, return_counts = True)[1]
        self.D = D
        self.dt = dt
