class TimeSeries():
    def __init__(self, x, t_idx, D = None):
        self.x = x
        self.t_idx = t_idx
        self.T = len(np.unique(t_idx))
        self.D = D

