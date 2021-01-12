import torch
import ot
import sklearn
import numpy as np
import scipy as sp
import math
from pathos.multiprocessing import ProcessingPool as Pool
import copy
import anndata


def get_C_mean(adata_s, t, t_next = None, mode = "tr"):
    if mode == "tr":
        C = sklearn.metrics.pairwise_distances(adata_s.obsm['X_pca'][adata_s.obs.day == t, :], adata_s.obsm['X_pca'][adata_s.obs.day == t_next, :], metric = 'sqeuclidean')
    elif mode == "self":
        C = sklearn.metrics.pairwise_distances(adata_s.obsm['X_pca'][adata_s.obs.day == t, :], adata_s.obsm['X_pca'][adata_s.obs.day == t, :], metric = 'sqeuclidean')
    return np.mean(C)


def subsamp(adata, day, size):
    idx = np.nonzero(np.array((adata.obs.day == day).array))[0]
    idx_samp = np.random.choice(idx, size = size, replace = False)
    return adata[idx_samp, :]


def geo_interp_wot(ot_model, t0, t1, N = 100, interp_frac = 0.5, coord_to_use = 'X_pca_orig', tmap = None):
    if tmap is None:
        T = torch.tensor(ot_model.compute_transport_map(t0, t1).X, dtype = torch.float64)
    else:
        T = torch.tensor(tmap.X, dtype = torch.float64)
    adata = ot_model.matrix
    T_norm = T @ torch.diag(1/(T.sum(dim = 0)**(1 - interp_frac)))
    T_norm = (T_norm/T_norm.sum()).flatten().cpu()
    # p = torch.distributions.categorical.Categorical(probs = T_norm)
    # samp = p.sample(sample_shape = torch.Size([N]))
    samp = np.random.choice(T_norm.shape[0], size = N, p = T_norm.cpu())
    out = torch.zeros(N, adata.obsm[coord_to_use].shape[1])
    for k in range(0, N):
        idx_i = samp[k] // T.shape[1]
        idx_j = samp[k] % T.shape[1]
        x0 = torch.from_numpy(adata.obsm[coord_to_use][adata.obs.day == t0, :][idx_i.item(), :])
        x1 = torch.from_numpy(adata.obsm[coord_to_use][adata.obs.day == t1, :][idx_j.item(), :])
        out[k, :] = x0 + interp_frac*(x1 - x0)
    return out
