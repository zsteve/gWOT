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
import sklearn
from sklearn import metrics

# Simulations

def dW(dt, sz):
    # Wiener process increments 
    return np.sqrt(dt)*np.random.standard_normal(sz)

def sde_integrate(dV, nu, x0, t, steps, birth_death = False, b = None, d = None, g_max = 50, snaps = None):
    if birth_death:
        # store g_max*x0.size[0] particles. Output error if we try and exceed this, though.
        # g = g(x, t) = b(x, t) - d(x, t)
        x = np.zeros((g_max*x0.shape[0], x0.shape[1]))
        x[0:x0.shape[0], :] = x0
    else:
        x = np.array(x0, copy = True)
    
    x_mask = np.zeros(x.shape[0], dtype = bool)
    x_mask[0:x0.shape[0]] = True

    dt = t/steps if steps > 0 else None
    t_current = 0
    snap = np.zeros((len(snaps), ) + x.shape)
    snap_mask = np.zeros((len(snaps), ) + (x.shape[0], ), dtype = bool)
    dV_vec = np.zeros(x.shape)
    
    if steps == 0:
        if snaps is not None and 0 == snaps:
            snap[0] = x
            snap_mask[0] = x_mask

    for i in range(0, steps):
        # for j in range(0, x.shape[0]):
            # update all existing particles
            # if x_mask[j]: 
            #     dV_vec[j, :] = dV(x[j, :], t_current)
            #     x[j, :] = x[j, :] - dV_vec[j, :]*dt + np.sqrt(nu)*dW(dt, x[j, :].shape)
        dV_vec[x_mask, :] = dV(x[x_mask, :], t_current)
        x[x_mask, :] = x[x_mask, :] - dV_vec[x_mask, :]*dt + np.sqrt(nu)*dW(dt, x[x_mask, :].shape)

        # birth/death step
        if birth_death and b is not None and d is not None:
            x_mask_new = x_mask.copy()
            for j in range(0, x.shape[0]):
                if x_mask[j]:
                    u = np.random.uniform()
                    if u < dt*b(x[j, :], t_current):
                        # birth event
                        k = np.where(x_mask_new == False)[0][0]
                        x[k, :] = x[j, :]
                        x_mask_new[k] = True
                    elif u < dt*b(x[j, :], t_current) + dt*d(x[j, :], t_current):
                        # death event
                        x_mask_new[j] = False
                    else:
                        pass
            x_mask = x_mask_new 
        t_current += dt
            
        if snaps is not None and np.sum(i == snaps):
            snap[np.where(i == snaps)[0]] = x
            snap_mask[np.where(i == snaps)[0]] = x_mask
    return snap, snap_mask

# Distances

def empirical_dist(mu_spt, nu_spt, max_iter = 1000000):
    C = sklearn.metrics.pairwise_distances(mu_spt, nu_spt, metric = 'sqeuclidean')
    return ot.emd2(np.ones(mu_spt.shape[0])/mu_spt.shape[0], np.ones(nu_spt.shape[0])/nu_spt.shape[0], C, numItermax = max_iter)

def empirical_entropic_coupling(mu_spt, nu_spt, eps, max_iter = 5000, method = "sinkhorn"):
    C = sklearn.metrics.pairwise_distances(mu_spt, nu_spt, metric = 'sqeuclidean')
    return ot.sinkhorn(np.ones(mu_spt.shape[0])/mu_spt.shape[0], np.ones(nu_spt.shape[0])/nu_spt.shape[0], C, eps, numItermax = max_iter, method = method)

# kernel method

def ker_smooth(m, h):
    w_smoothed = torch.zeros(m.ts.T, m.x.shape[0])
    t_map = np.concatenate([np.array([0, ]), np.cumsum(m.ts.dt)])
    k = lambda r: math.exp(-(r/h)**2)
    for i in range(0, m.ts.T):
        for j in range(0, m.ts.T):
            w_smoothed[i, m.t_idx == j] = k(t_map[j] - t_map[i])

    w_smoothed = ((w_smoothed.T/w_smoothed.sum(dim = 1)).T)
    return w_smoothed

def pi0_kde(ts, bw_method = None, num_times = 1):
    pi0_kde = sp.stats.gaussian_kde(ts.x[np.isin(ts.t_idx, np.arange(num_times)), :].T, bw_method = bw_method)
    pi0 = torch.Tensor(pi0_kde(ts.x.T))
    return pi0/pi0.sum()

# assorted util

def density_to_grid(d, x, n = (100, 100), box = np.array([[-1, -1], [1, 1]])*2):
    # Discretize a 2D distribution with weights d supported on x onto a regular grid
    # with n = (n_x, n_y) grid elements, corresponding to box. 
    grid = np.zeros(n)
    box_w = box[1, 0] - box[0, 0]
    box_h = box[1, 1] - box[0, 1]
    for p in range(0, x.shape[0]):
            if all([x[p, 0] >= box[0, 0], x[p, 1] >= box[0, 1],
                            x[p, 0] < box[1, 0], x[p, 1] < box[1, 1]]):
                    grid_x = int((x[p, 0]-box[0, 0])*n[0]//box_w)
                    grid_y = int((x[p, 1]-box[0, 1])*n[1]//box_h)
                    grid[grid_y, grid_x] += d[p]
    return grid

def density_to_grid_1d(d, x, n = 100, box = np.array([-1, 1])*2):
    # 1D version of density_to_grid()
    grid = np.zeros(n)
    box_w = box[1] - box[0]
    for p in range(0, x.shape[0]):
            if all([x[p] >= box[0], x[p] < box[1]]):
                    grid_x = int((x[p]-box[0])*n//box_w)
                    grid[grid_x] += d[p]
    return grid


def to_grid_coord_1d(x, n = 100, box = np.array([-1, 1])*2):
    # Convert 1D coordinates x to grid indices on a 1D grid of size n, corresponding to box.
    grid_coords = np.zeros(x.shape[0], dtype = np.int)
    box_w = box[1] - box[0]
    for p in range(0, x.shape[0]):
            if all([x[p] >= box[0], x[p] < box[1]]):
                    grid_x = int((x[p]-box[0])*n//box_w)
                    grid_coords[p] = grid_x
    return grid_coords

def prod_to_grid(gamma, mu_spt, nu_spt, n = (20, 20), box = np.array([[-2,-2], [2,2]])):
    # Discretize a joint distribution gamma on the product space, i.e. supported on mu_spt x nu_spt
    # onto a grid of n = (n_x, n_y), corresponding to box. 
    gamma_grid = np.zeros(n)
    mu_spt_gridded = to_grid_coord_1d(mu_spt, n = n[0], box = box[:, 0])
    nu_spt_gridded = to_grid_coord_1d(nu_spt, n = n[1], box = box[:, 1])
    for i in range(0, gamma.shape[0]):
        for j in range(0, gamma.shape[1]):
                    gamma_grid[mu_spt_gridded[i], nu_spt_gridded[j]] += gamma[i, j]
    return gamma_grid

def velocity_from_coupling(gamma, mu_spt, nu_spt, dt):
    return (gamma @ nu_spt - gamma.sum(1).view(-1, 1) * mu_spt)/dt
