import numpy as np

def sample_brownian_bridge(t0, x0, t1, x1, sigma, N):
    def samp(t0, x0, t1, x1, sigma, t):
        return np.random.normal(x0 + (t - t0)/(t1 - t0)*(x1 - x0), (t1 - t)*(t - t0)/(t1 - t0)*sigma**2 * np.eye(x0.shape[0]))
    if N == 0:
        return np.array([]), np.array([])
    x_mid = np.array(samp(t0, x0, t1, x1, sigma, (t0 + t1)/2))
    t_mid = np.array([0.5*(t0 + t1), ])
    if N == 1:
        return x_mid, t_mid
    x_left, t_left = sample_brownian_bridge(t0, x0, 0.5*(t0 + t1), x_mid, sigma, N//2)
    x_right, t_right = sample_brownian_bridge(0.5*(t0 + t1), x_mid, t1, x1, sigma, N//2)
    return np.concatenate([x_left, x_mid, x_right]), np.concatenate([t_left, t_mid, t_right])

def sample_schrodinger_bridge(t0, t1, gamma, mu_spt, nu_spt, sigma, N, M):
    gamma_norm = (gamma/gamma.sum()).flatten()
    samp = np.random.choice(gamma_norm.shape[0], size = M, p = gamma_norm)
    path_x_all = []
    path_t = None
    for k in range(len(samp)):
        idx_i = samp[k] // gamma.shape[1]
        idx_j = samp[k] % gamma.shape[1]
        path_x, path_t = sample_brownian_bridge(t0, mu_spt[idx_i, :].reshape(1, -1), t1, nu_spt[idx_j, :].reshape(1, -1), sigma = sigma, N = N)
        path_x_all += [path_x, ]
    return path_x_all, path_t, samp // gamma.shape[1], samp % gamma.shape[1]

def sample_coupling(gamma, N = 1, norm = True):
    gamma_norm = gamma/gamma.sum() if norm else gamma
    gamma_norm = gamma_norm.flatten()
    samp = np.random.choice(gamma_norm.shape[0], size = N, p = gamma_norm)
    return samp // gamma.shape[1], samp % gamma.shape[1]

def sample_paths(gamma_all, N = 1, coord = False, x_all = None, get_gamma_fn = None, num_couplings = None):
    def path_idx_to_coord(paths, x_all):
        return np.array([[x_all[j][paths[i, j]] for j in range(paths.shape[1])] for i in range(paths.shape[0])])
    def sample_conditional(idx, gamma):
        return np.array([np.random.choice(gamma.shape[1], size = 1, p = (gamma[i, :]/gamma[i, :].sum()).flatten()) for i in idx]).flatten()
    idx_all = []
    if gamma_all is not None:
        num_couplings = len(gamma_all)
    gamma = gamma_all[0] if gamma_all is not None else get_gamma_fn(0)
    idx0, idx1 = sample_coupling(gamma, N = N)
    idx_all += [idx0, idx1]
    for i in range(1, num_couplings):
        gamma = gamma_all[i] if gamma_all is not None else get_gamma_fn(i)
        idx_all += [sample_conditional(idx_all[-1], gamma), ]
    paths = np.array(idx_all).T
    if coord:
        return path_idx_to_coord(paths, x_all)
    else:
        return paths
