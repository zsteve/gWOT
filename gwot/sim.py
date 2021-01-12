import torch
import ot
import sklearn
import numpy as np
import scipy as sp
import math
import autograd
import autograd.numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
import copy

from gwot.ts import TimeSeries
from gwot.util import sde_integrate

class Simulation(TimeSeries):
    def __init__(self, V, dV, N, T, d, D, t_final, ic_func, pool, birth_death = False, birth = None, death = None):
        self.V = V
        self.dV = dV
        self.birth_death = birth_death
        self.birth = birth
        self.death = death
        self.N = N
        self.d = d
        self.T = T
        self.D = D
        self.t_final = t_final
        self.ic_func = ic_func
        self.dt = (t_final/(T-1))*np.ones(T-1)
        self.pool = pool

    def sample(self, steps_scale = 1, trunc = None):
        ic_all = [self.ic_func(self.N[i], self.d) for i in np.arange(0, self.T, 1)]
        def F(i):
            snap, snap_mask = sde_integrate(self.dV, nu = self.D, x0 = ic_all[i],
                                        birth_death = self.birth_death,
                                        b = self.birth, d = self.death, 
                                        t = (self.t_final)*(i/self.T), 
                                        steps = steps_scale*i, 
                                        snaps = np.array([max(steps_scale*i-1, 0), ])) 
            return snap[snap_mask, :]
        if self.pool:
            self.snaps = self.pool.map(F, np.arange(0, self.T, 1))
        else:
            self.snaps = [F(i) for i in np.arange(0, self.T, 1)]

        if trunc is not None:
            samp_sizes = np.array([s.shape[0] for s in self.snaps])
            for i in range(0, len(self.snaps)):
                self.snaps[i] = self.snaps[i][np.random.choice(samp_sizes[i], size = min(samp_sizes[i], trunc)), :]
        self.x = np.vstack(self.snaps) 
        self.t_idx = np.concatenate([np.array([i]).repeat(self.snaps[i].shape[0]) for i in range(0, len(self.snaps))])
        return self.snaps

    def sample_trajectory(self, steps_scale = 1, N = 1):
        ic = self.ic_func(N, self.d)
        snap, snap_mask = sde_integrate(self.dV, nu = self.D, x0 = ic,
            b = self.birth, d = self.death, 
            t = self.t_final, 
            steps = self.T*steps_scale, 
            snaps = np.arange(self.T)*steps_scale) 
        return np.moveaxis(snap, 0, 1)

    def __copy__(self):
        return Simulation(V = self.V, dV = self.dV, N = self.N, T = self.T, d = self.d, D = self.D, t_final = self.t_final,
                         ic_func = self.ic_func, pool = self.pool,
                         birth_death = self.birth_death, birth = self.birth, death = self.death)

    def __deepcopy__(self, memo):
        return Simulation(V = copy.deepcopy(self.V, memo), dV = copy.deepcopy(self.dV, memo), 
                             N = self.N, T = self.T, d = self.d, D = self.D, t_final = self.t_final,
                             ic_func = copy.deepcopy(self.ic_func, memo), pool = self.pool,
                             birth_death = self.birth_death, 
                             birth = copy.deepcopy(self.birth, memo), death = copy.deepcopy(self.death, memo))
