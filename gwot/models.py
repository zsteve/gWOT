import numpy as np
import torch
from torch.autograd import grad, Variable
import pykeops
from pykeops.torch import LazyTensor, Vi, Vj
import math
import sklearn
import ot
import os
import dill
from scipy.sparse.linalg import aslinearoperator, eigs, LinearOperator
from scipy import matrix

from gwot.lambertw import lambertw
from gwot.ts import TimeSeries
from gwot.sim import Simulation

class OTModel(torch.nn.Module):
    """Core gWOT model class for inference and manipulating outputs.
    Forms and solves the optimisation problem with 
    general form
    :math:`\\min_{\\mathbf{R}_{t_1}, \ldots, \\mathbf{R}_{t_T}} \
           \\lambda \\mathrm{Reg}(\\mathbf{R}_{t_1}, \ldots, \\mathbf{R}_{t_T}) \
         + \\mathrm{Fit}(\\mathbf{R}_{t_1}, \ldots, \\mathbf{R}_{t_T})`,
    where :math:`\\mathbf{R}_{t_1}, \ldots, \\mathbf{R}_{t_T}` are the reconstructed marginals
    at times :math:`t_1, \ldots, t_T`. 

    :param ts: TimeSeries object containing input data.
    :param lamda_reg: regularisation strength parameter :math:`\\lambda`.
    :param D: diffusivity :math:`D`.
    :param w: `torch.Tensor` of weights for data-fitting term at each timepoint. 
            If `None`, then we take :math:`w_i = N_i/\sum_j N_j` where :math:`N_i` is the number of particles at timepoint `i`.
    :param lamda: `torch.Tensor` of weights :math:`\\lambda_i` controlling tradeoff of cross-entropy vs OT in data-fitting term
            at each timepoint.
    :param eps: `torch.Tensor` of entropic regularisation parameters to use in the 
            regularising functional OT term. If `None`, then we take the entries to be :math:`2D\Delta t_i` 
            (the theoretically correct value).
    :param eps_df: `torch.Tensor` of entropic regularisation parameters :math:`\\varepsilon_i` 
            to use in the OT component of the data-fitting functional. 
    :param c_scale: `torch.Tensor` of cost matrix scalings :math:`\\overline{C}_i` to use in the regularising functional.
            That is, the cost matrix for the pair of timepoints :math:`(t_i, t_{i+1})` will be 
            :math:`C^{(i)}_{jk} = C_{jk}/\\overline{C}_i`.
    :param c_scale_df: `torch.Tensor` of cost matrix scalings to use in the OT component of the 
            data-fitting functional. Defined in the same way as `c_scale`. 
    :param m: `torch.Tensor` of estimates of the total mass :math:`m_i` at each timepoint :math:`t_i`.
    :param g: `torch.Tensor` of growth rates, where :math:`g_{ij}` denotes the growth rate at 
            time :math:`t_i` and spatial location :math:`x_j`.
    :param kappa: (Only used for soft branching constraint) `torch.Tensor` of penalty weights :math:`\\kappa_i` 
            corresponding :math:`G_{i}(\\overline{\\mathbf{R}}_{t_i}, \\mathbf{R}_{t_i})`.
    :param growth_constraint: "exact" for exact branching constraint, and "KL" for soft branching constraint.
    :param u_hat: `torch.Tensor` of initial values for dual variables :math:`\\{\\hat{u}_i\\}_{i=1}^T`. 
            If `None`, we initialise with zeros.
    :param v_hat: `torch.Tensor` of initial values for :math:`\\{\\hat{v}_i\\}_{i=1}^T`. 
            If `None`, we initialise with zeros.
    :param pi_0: `torch.Tensor` of initial distribution :math:`\\pi_0` to use, 
            or else a choice of "uniform" (uniform on the space :math:`\\overline{\\mathcal{X}}`) 
            or "stationary" (stationary distribution of heat kernel on :math:`\\overline{\\mathcal{X}}`)
    :param device: Device to use with PyTorch (e.g. `torch.device("cuda:0")` in the case of GPU).
            If `None`, we use `torch.device("cpu")`. 
    :param use_keops: `True` to use KeOps for on-the-fly kernel reductions. Otherwise all kernels
            are precomputed and stored in memory. 
    """
    def __init__(self, ts, 
                 lamda_reg, 
                 D = None, 
                 w = None, 
                 lamda = None, 
                 eps = None, 
                 eps_df = None, 
                 c_scale = None, 
                 c_scale_df = None,
                 m = None, 
                 g = None, 
                 kappa = None, 
                 growth_constraint = 'exact', 
                 u_hat = None, 
                 v_hat = None,
                 pi_0 = 'stationary',
                 device = None,
                 use_keops = True):
        super(OTModel, self).__init__()
        # if no device specified, use CPU
        self.device = torch.device("cpu") if device is None else device
        self.use_keops = use_keops
        # initialise basic parameters 
        self.w = torch.from_numpy(ts.N).to(self.device)/ts.N.sum() if w is None else w
        self.lamda = torch.ones(ts.T, device = self.device) if lamda is None else lamda
        self.ts = ts
        self.lamda_reg = lamda_reg
        self.D = ts.D if D is None else D
        self.t_idx = torch.from_numpy(ts.t_idx).to(self.device)
        self.dt = torch.from_numpy(ts.dt).to(self.device)
        self.eps = torch.from_numpy(2*self.D*ts.dt).to(self.device) if eps is None else eps
        self.eps_df = eps_df
        self.x = torch.from_numpy(ts.x).to(self.device)
        self.c_scale = torch.ones(ts.T, device = self.device) if c_scale is None else c_scale
        self.c_scale_df = torch.ones(ts.T, device = self.device) if c_scale_df is None else c_scale_df
        self.C = None
        # parameters related to growth
        self.m = torch.ones(ts.T, device = self.device) if m is None else m 
        self.g = torch.ones(ts.T, self.x.shape[0], device = self.device) if g is None else g 
        self.kappa = kappa 
        if growth_constraint not in ["exact", "KL"]:
            raise ValueError("growth_constraint must be one of 'exact' or 'KL'")
        self.growth_constraint = growth_constraint
        # pi_0 (initial distribution for reference process)
        if type(pi_0) is not str:
            self.pi_0 = pi_0 
        elif pi_0 == 'uniform':
            self.pi_0 = torch.ones(self.x.shape[0], device = self.device)/self.x.shape[0]
        elif pi_0 == 'stationary':
            self.pi_0 = None
        else:
            raise ValueError("pi0 must either be a `torch.Tensor`, or one of 'uniform' or 'stationary'")
        # init kernels
        self.kernel_init()
        self.uv_init(u_hat, v_hat)

    def kernel_init(self,):
        """Initialise kernel matrices (as either `torch.Tensor` or `LazyTensor`) for use later.
        
        :param use_keops: If `True`, then all kernel matrices are initialised as `LazyTensor`s.
        """
        def get_pi0_unif(use_keops):
            # not recommended to use this in practice, since the eigengap is generally very small
            if use_keops:
                x_cpu = self.x.cpu() # scipy needs CPU location
                x_cpu_i = LazyTensor(x_cpu.view(1, x_cpu.shape[0], x_cpu.shape[1]).numpy())
                x_cpu_j = LazyTensor(x_cpu.view(x_cpu.shape[0], 1, x_cpu.shape[1]).numpy())
                D_ij = ((x_cpu_i - x_cpu_j)**2/(self.eps[0]*self.c_scale[0]).item()).sum(2) # scaling doesn't matter
                Z_ij = (-D_ij).exp()
                K_ij = Z_ij*LazyTensor(1/Z_ij.sum(dim = 1), axis = 1).T
                K_op = aslinearoperator(K_ij.T)
                self.K_op = K_op
                eigvals, eigvecs = eigs(K_op, k = 1)
                self.pi_0 = torch.from_numpy(eigvecs.real/eigvecs.real.sum()).to(self.device)
            else:
                K_op = aslinearoperator(matrix(K).T)
                self.K_op = K_op
                eigvals, eigvecs = eigs(K_op, k = 1)
                self.pi_0 = torch.from_numpy(eigvecs.real/eigvecs.real.sum()).to(self.device)
        if self.use_keops:
            # construct LazyTensor kernel for reg functional
            x_i = LazyTensor(self.x.view(1, self.x.shape[0], self.x.shape[1]))
            x_j = LazyTensor(self.x.view(self.x.shape[0], 1, self.x.shape[1]))
            D_ij = [((x_i - x_j)**2/(self.eps[i]*self.c_scale[i])).sum(2) for i in range(0, self.ts.T-1)]
            Z_ij = [(-D).exp() for D in D_ij] # Kernel matrix
            M_sums_i = [LazyTensor(1/M.sum(dim = 1), axis = 1).T for M in Z_ij]
            self.K_ij = [Z_ij[i]*M_sums_i[i] for i in range(0, len(Z_ij))]
            if self.pi_0 is None:
                get_pi0_unif()
            self.K_ij[0] = LazyTensor(self.pi_0.view(-1, 1, 1))*self.K_ij[0]
            # construct LazyTensor kernel for data-fitting functional
            D_ij_df = [((x_i - x_j)**2/(self.eps_df[i]*self.c_scale_df[i])).sum(2) for i in range(0, self.ts.T)]
            self.K_ij_df = [(-D).exp() for D in D_ij_df] 
        else:
            # do not use keops
            C = torch.from_numpy(sklearn.metrics.pairwise_distances(self.x.cpu(), metric = "sqeuclidean")).to(self.device)
            # construct kernel for reg functional
            D_ij = [C/(self.eps[i]*self.c_scale[i]) for i in range(0, self.ts.T-1)]
            Z_ij = [(-D).exp() for D in D_ij]
            self.K_ij = [(M.T/M.sum(dim = 1)).T for M in Z_ij]
            if self.pi_0 is None:
                get_pi0_unif(self.K_ij[0].cpu())
            self.K_ij[0] = self.pi_0.reshape(-1, 1) * self.K_ij[0]
            # now construct kernel for data fitting 
            D_ij_df = [C/(self.eps_df[i]*self.c_scale_df[i]) for i in range(0, self.ts.T)]
            self.K_ij_df = [(-D).exp() for D in D_ij_df]
       
    def uv_init(self, u_hat = None, v_hat = None):
        """Initialise dual variables :math:`\\{\\hat{u}_i, \\hat{v}_i\\}_i` for model. 
        
        :param u_hat: initial value of `u_hat` to use. If `None`, then initialise with zeros.
        :param v_hat: initial value of `v_hat` to use. If `None`, then initialise with zeros. 
        """
        if u_hat is None:
            u_hat = torch.zeros(self.ts.T, self.ts.x.shape[0], device = self.device)
        if v_hat is None:
            v_hat = torch.zeros(self.ts.T, self.ts.x.shape[0], device = self.device)
        for i in range(0, self.ts.T):
            v_hat[i, i != self.ts.t_idx] = -self.lamda[i]
        self.register_parameter(name = 'u_hat', param = torch.nn.Parameter(Variable(u_hat, requires_grad = True)))
        self.register_parameter(name = 'v_hat', param = torch.nn.Parameter(Variable(v_hat, requires_grad = True)))
        
    def forward(self):
        return self.dual_obj()

    def crossent_star(self, u, i):
        """Legendre transform :math:`u \\mapsto \mathrm{KL}^*(\\rho_{t_i} | u)` of generalised cross-entropy 
                in its second argument :math:`x \\mapsto \\mathrm{KL}(\\rho_{t_i} | x)`, where the first argument
                is the observed sample :math:`\\rho_{t_i}` at time `i`.
        
        :param u: dual variable
        :param i: timepoint index
        """
        val = -torch.sum(torch.log(1 - u[self.t_idx == i]))/torch.sum(self.t_idx == i)
        if torch.isnan(val): 
            val = torch.tensor(float("Inf"), device = self.device)
        return val
    
    def logKexp(self, K, x):
        """Compute kernel reduction of the form :math:`\\log(K\\exp(x))`
        """
        if self.use_keops:
            if type(x) is not pykeops.torch.LazyTensor:
                x = LazyTensor(x.view(1, -1, 1))
            return x.logsumexp(dim = 1, weight = K).view(-1)
        else:
            scale = x.max()
            return (K @ (x - scale).exp()).log() + scale

    def logsumexp_weight(self, w, x, dim = 1):
        """Compute kernel reduction of the form :math:`\\log(\\langle w, \\exp(x) \\rangle)`
        """
        if self.use_keops:
            if type(x) is not pykeops.torch.LazyTensor:
                x = LazyTensor(x)
            if type(w) is not pykeops.torch.LazyTensor:
                w = LazyTensor(w)
            return x.logsumexp(weight = w, dim = dim).view(-1)
        else:
            return torch.log((w * torch.exp(x)).sum())

    def compute_phi(self, i = None, out_arr = None):
        """Compute auxiliary dual variable :math:`\\phi_i` 

        :param i: index of which :math:`\\phi_i` variable we want to compute. 
                Set to be `None` if we want to compute all of them, but then `out_arr` 
                must be specified.
        :param out_arr: preallocated `torch.Tensor` in which to output `phi`. 
        :return: if `out_arr == None`, returns the value of :math:`\\phi_i` as a `torch.Tensor`.
                Otherwise returns `None` and the result is written to `out_arr`.
        """
        if i is None:
            self.compute_phi(i = 0, out_arr = out_arr)
            return None
        elif i == self.ts.T-2:
            psi_next = (-self.w[i+1]*self.u_hat[i+1, :])/self.lamda_reg
        else:
            phi_next = self.compute_phi(i = i+1, out_arr = out_arr)
            v_next = (-self.m[i+1]/self.m[i+2])*self.dt[i+1]*phi_next
            u_next = (-self.eps[i+1])*self.logKexp(self.K_ij[i+1], v_next/self.eps[i+1]).view(-1)
            psi_next = (-1/self.lamda_reg)*(self.lamda_reg*u_next/self.dt[i+1] + self.w[i+1]*self.u_hat[i+1, :])
        if self.growth_constraint == "KL":
            phi = self.kappa[i]*self.g[i, :]*(1 - torch.exp(psi_next/self.kappa[i]))
        elif self.growth_constraint == "exact":
            phi = -self.g[i, :]*psi_next
        else:
            raise Exception("growth_constraint not 'KL' or 'exact'")
        if out_arr is not None:
            out_arr[i, :] = phi
        return phi

    def dual_obj(self):
        """Evaluate dual objective (see Eq. 3.16 in manuscript)

        """
        u0 = (-self.w[0]*self.dt[0]*self.u_hat[0, :])/self.lamda_reg
        v0 = (-self.m[0]/self.m[1])*self.dt[0]*self.compute_phi(i = 0)
        w = self.logKexp(self.K_ij[0], v0/self.eps[0])
        reg = self.eps[0]*self.logsumexp_weight((u0/self.eps[0]).exp().view(1, -1, 1), w.view(1, -1, 1), dim = 1)
        return (self.lamda_reg/(self.m[0]*self.dt[0]))*reg + \
                torch.dot(self.w/self.m, 
                        torch.stack([self.eps_df[i] * torch.dot(torch.exp(self.u_hat[i, :]/self.eps_df[i]), 
                                                                self.K_ij_df[i] @ torch.exp(self.v_hat[i, :]/self.eps_df[i])) 
                                    for i in range(0, self.ts.T)])) + \
                torch.dot(self.lamda * self.w, 
                          torch.stack([self.crossent_star(-self.v_hat[i, :]/self.lamda[i], i) 
                                       for i in range(0, self.ts.T)]))

    def primal_obj(self, terms = False):
        """Evaluate primal objective (see Eq. C.3 in manuscript)

        """
        # Define first some helper functions. 
        def eval_primal_OT(model, phi_all):
            u = (-model.w[0]*model.dt[0]/model.lamda_reg)*model.u_hat[0, :]
            v = (-model.m[0]/model.m[1])*model.dt[0]*phi_all[0, :].to(self.device)
            Z = self.logsumexp_weight((model.K_ij[0] @ (v/self.eps[0]).exp()).view(-1, 1, 1), 
                                        (u/model.eps[0]).view(-1, 1, 1), dim = 0).item()
            return torch.dot(u*torch.exp(u/self.eps[0] - Z), model.K_ij[0] @ (v/self.eps[0]).exp()) + \
                    torch.dot(v*torch.exp(v/self.eps[0] - Z), model.K_ij[0].T @ (u/self.eps[0]).exp()) - \
                    self.eps[0]*Z
        def eval_primal_OT_tilde(model, i, phi_all):
            p = model.get_R(i)
            v = (-model.m[i]/model.m[i+1])*model.dt[i]*phi_all[i, :].to(self.device)
            u = -self.eps[i]*torch.log(model.K_ij[i] @ torch.exp(v/self.eps[i]))
            alpha = p 
            beta = (v/self.eps[i]).exp() * (model.K_ij[i].T @ (alpha/(model.K_ij[i] @ (v/self.eps[i]).exp())))
            return torch.dot(alpha, u) + torch.dot(beta, v)
        def eval_primal_OT_df(model, i):
            u = model.u_hat[i, :]
            v = model.v_hat[i, :]
            return torch.dot(u * torch.exp(u/model.eps_df[i]), model.K_ij_df[i] @ torch.exp(v/model.eps_df[i])) +  \
                    torch.dot(v * torch.exp(v/model.eps_df[i]), model.K_ij_df[i].T @ torch.exp(u/model.eps_df[i])) -  \
                    model.eps_df[i] * torch.exp(u/model.eps_df[i]).T @ (model.K_ij_df[i] @ torch.exp(v/model.eps_df[i]))
        def KL(alpha, beta):
            return (alpha*(torch.log(alpha/beta)) - alpha + beta).sum()
        def eval_primal_crossent_df(model, i):
            p_hat = model.get_R_hat(i)
            N = 1.0*(model.t_idx == i).sum()
            return ((1/N)*(-torch.log(p_hat[model.t_idx == i]/model.m[i]) - torch.log(N))).sum() - 1 + \
                        p_hat.sum()/model.m[i]
        def eval_primal_growth_KL(model, i, R):
            r = R[i, :]
            p = model.get_R(i+1)
            return model.kappa[i]*KL(p, model.g[i, :]*r)
        # precompute all phi values for efficiency
        phi_all = torch.zeros(self.ts.T-1, self.x.shape[0], device = self.device)
        self.compute_phi(out_arr = phi_all)
        crossent_df_all = torch.stack([eval_primal_crossent_df(self, i) for i in range(0, self.ts.T)])
        if self.growth_constraint == 'KL':
            R_all = self.get_R_bar(phi_all = phi_all)
            growth = torch.stack([eval_primal_growth_KL(self, i, R_all) for i in range(0, self.ts.T-1)])
        spine_all = torch.stack([eval_primal_OT(self, phi_all)] + \
                            [eval_primal_OT_tilde(self, i, phi_all) for i in range(1, self.ts.T - 1)])
        branches_all = torch.stack([eval_primal_OT_df(self, i) for i in range(0, self.ts.T)])
        if terms:
            # return individual terms of the primal
            if self.growth_constraint == 'exact':
                return {"reg" : (1/(self.dt*self.m[0:-1]) * (spine_all)), 
                        "branches" : 1/self.m * branches_all, 
                        "df" : crossent_df_all}
            else:
                return {"reg" : (1/(self.dt*self.m[0:-1]) * (spine_all)), 
                        "branches" : 1/self.m * branches_all, 
                        "growth" : (growth/self.m[1:]),
                        "df" : crossent_df_all}
        else:
            # return the primal objective (scalar)
            if self.growth_constraint == 'exact':
                return (self.lamda_reg/(self.dt*self.m[0:-1]) * spine_all).sum() + \
                        (self.w * (1/self.m * branches_all + self.lamda * crossent_df_all)).sum()
            else:
                return self.lamda_reg * ((spine_all/(self.dt*self.m[0:-1])).sum() + (growth/self.m[1:]).sum()) + \
                        (self.w * (1/self.m * branches_all + self.lamda * crossent_df_all)).sum()

    def get_coupling_df(self, i):
        """Get the OT coupling :math:`\\gamma` for the `i`th data-fitting OT term
            :math:`\\mathrm{OT}_{\\varepsilon_i}(\\mathbf{R}_{t_i}, \\mathbf{\\hat{R}}_{t_i})`.
            
        :return: :math:`\\gamma` as a `torch.Tensor`.
        """
        return torch.diag(torch.exp(self.u_hat[i, :]/self.eps_df[i])) @ \
                (self.K_ij_df[i] @ torch.diag(torch.exp(self.v_hat[i, :]/self.eps_df[i])))

    def get_coupling_reg(self, i, K = None):
        """Get the OT coupling :math:`\\gamma` for the `i`th regularisation OT term
            :math:`\\mathrm{OT}_{\\sigma^2 \\Delta t_i}(\\mathbf{R}_{t_i}, \\mathbf{\\overline{R}}_{t_{i+1}})`.
        
        :param K: kernel matrix to use. If `K == None`, then use the kernels computed by `kernel_init`.
        :return: :math:`\\gamma` as either a `LazyTensor` or `torch.Tensor`, depending on the type of `K`.
        """
        K = self.K_ij[i] if K is None else K
        if i > 0:
            v = (-self.m[i]/self.m[i+1])*self.dt[i]*self.compute_phi(i)
            g = v/self.eps[i]
            x = self.logKexp(self.K_ij[i], g.view(1, -1, 1)).view(-1)
            alpha = self.get_R(i)
            if type(K) is pykeops.torch.LazyTensor:
                return LazyTensor(alpha.view(-1, 1), axis = 0) * K * \
                        (LazyTensor(g.view(1, -1, 1)) - LazyTensor(x.view(-1, 1, 1))).exp()
            else:
                return alpha.view(-1, 1) * K * (g.view(1, -1) - x.view(-1, 1)).exp()
        elif i == 0:
            u0 = (-self.w[0]*self.dt[0]*self.u_hat[0, :])/(self.lamda_reg)
            v0 = (-self.m[0]/self.m[1])*self.dt[0]*self.compute_phi(i = 0)
            x = self.logKexp(self.K_ij[0], (v0/self.eps[0]).view(1, -1, 1))
            logZ = self.logsumexp_weight(torch.exp(u0/self.eps[0]).view(-1, 1, 1), x.view(-1, 1, 1), dim = 0).item()
            if type(K) is pykeops.torch.LazyTensor:
                return LazyTensor((u0/self.eps[0]).view(-1, 1, 1)).exp() * K * \
                        (LazyTensor((v0/self.eps[0]).view(1, -1, 1)) - logZ).exp()
            else:
                return (((K @ torch.diag(torch.exp(v0/self.eps[0] - logZ))).T * torch.exp(u0/self.eps[0])).T)
 
    def get_R(self, i = None): 
        """Get reconstructed marginal :math:`\\mathbf{R}_{t_i}` at timepoint `i`.
        
        """
        if i is None:
            return torch.stack([self.get_R(i) for i in range(0, self.ts.T)])
        else:
            return torch.exp(self.u_hat[i, :]/self.eps_df[i]) * (self.K_ij_df[i] @ torch.exp(self.v_hat[i, :]/self.eps_df[i]))

    def get_R_hat(self, i = None):
        """Get intermediate marginal :math:`\\mathbf{\\hat{R}}_{t_i}` at timepoint `i`.
        
        """
        if i is None:
            return torch.stack([self.get_R_hat(i) for i in range(0, self.ts.T)])
        else:
            return torch.exp(self.v_hat[i, :]/self.eps_df[i]) * (self.K_ij_df[i].T @ torch.exp(self.u_hat[i, :]/self.eps_df[i]))

    def get_R_bar(self, i = None, phi_all = None):
        """Get intermediate growth marginal :math:`\\mathbf{\\overline{R}}_{t_i}` at timepoint `i`.
        
        """
        if phi_all is None:
            phi_all = torch.zeros(self.ts.T-1, self.x.shape[0], device = self.device)
            self.compute_phi(out_arr = phi_all) 
        v_all = (phi_all.T * ((-self.m[0:-1]/self.m[1:])*self.dt)).T
        p_all = self.get_R()
        if i is None:
            return torch.stack([(torch.exp(v_all[i, :]/self.eps[i])) * \
                                (self.K_ij[i].T @ (p_all[i, :] / (self.K_ij[i] @ torch.exp(v_all[i, :]/self.eps[i])))) \
                            for i in range(0, self.ts.T-1)])
        else:
            return (torch.exp(v_all[i, :]/self.eps[i])) * \
                    (self.K_ij[i].T @ (p_all[i, :] / (self.K_ij[i] @ torch.exp(v_all[i, :]/self.eps[i]))))

    def get_K(self, i):
        """Get Gibbs kernel as a `torch.Tensor` for regulariser OT term from timepoint `i` to `i+1`. 
            N.B. the main reason this exists is because there is not an easy way to convert between 
            `LazyTensor` and `torch.Tensor` (standard dense array). Instead, we need to recompute 
            from scratch.
            
        """
        if self.C is None:
            self.C = sklearn.metrics.pairwise_distances(self.ts.x, metric = 'sqeuclidean')
        K = torch.exp(-torch.from_numpy(self.C).to(self.device)/float(self.c_scale[i]*self.eps[i]))
        K = (K.T/K.sum(1)).T
        if i == 0 and self.pi_0 is not None:
            K = (K.T * self.pi_0).T
        return K
    
    def get_K_df(self, i):
        """Get Gibbs kernel as a `torch.Tensor` for data-fitting OT term from timepoint `i` to `i+1`.
            N.B. the main reason this exists is because there is not an easy way to convert between 
            `LazyTensor` and `torch.Tensor` (standard dense array). Instead, we need to recompute 
            from scratch.
        
        """
        if self.C is None:
            self.C = sklearn.metrics.pairwise_distances(self.ts.x, metric = 'sqeuclidean')
        K_df = torch.exp(-torch.from_numpy(self.C).to(self.device)/float(self.c_scale_df[i]*self.eps_df[i]))
        K_df = (K_df.T/K_df.sum(1)).T
        return K_df
    
    def solve_sinkhorn(self, steps = 1000, tol = 5e-3, precompute_K = False, print_interval = 25):
        """Solve using Sinkhorn-like scheme (only for case without branching/growth, i.e. `g = 1`)
        
        :param steps: number of Sinkhorn steps to use 
        :param tol: primal-dual tolerance for convergence 
        :param precompute_K: if `True`, store kernel matrices in dense form (as `torch.Tensor`)
        :param print_interval: iteration interval at which to print iteration info.
        """
        # Block-optimisation (Sinkhorn-like) optimisation
        if self.use_keops and precompute_K:
            K_prec = [self.get_K(i) for i in range(self.ts.T-1)]
            K_df_prec = [self.get_K_df(i) for i in range(self.ts.T)]
        elif ~self.use_keops and precompute_K:
            # ignore precompute if not using Keops
            precompute_K = False
        c = 0 # mass constraint dual variable
        def U(i, c):
            return (-float(i == 0)*c - self.w[i]*self.u_hat[i, :])/(self.lamda_reg*self.D)
        def z1(i, c):
            if i == 0:
                p = torch.ones(self.u_hat.shape[1], device = self.device)
                return p
            else:
                p = torch.ones(self.u_hat.shape[1], device = self.device)
                for j in range(i):
                    if precompute_K:
                        p = (K_prec[j].T @ (p*U(j, c).exp())).T
                    else:
                        p = (self.K_ij[j].T @ (p*U(j, c).exp())).T
                return p
        def z2(i, c):
            p = torch.ones(self.u_hat.shape[1], device = self.device)
            for j in range(0, self.ts.T-i-1):
                if precompute_K:
                    p = K_prec[self.ts.T-j-2] @ (U(self.ts.T-j-1, c).exp() * p)
                else:
                    p = self.K_ij[self.ts.T-j-2] @ (U(self.ts.T-j-1, c).exp() * p)
            return p 
        
        def K_df(i):
            if precompute_K:
                return K_df_prec[i]
            else:
                return self.K_ij_df[i]

        obj_vals = []
        # initialise as ones 
        self.u_hat[:, :] = 0
        self.v_hat[:, :] = 0
        with torch.no_grad():
            for i in range(steps):
                # update u_hat
                for j in range(self.ts.T):
                    self.u_hat[j, :] = 1/(1/self.eps_df[j] + self.w[j]/(self.lamda_reg*self.D)) * \
                                    (torch.log(z1(j, c)*z2(j, c)/(K_df(j) @ torch.exp(self.v_hat[j, :]/self.eps_df[j]))) \
                                        - float(j == 0)*c/(self.lamda_reg*self.D) )
                for j in range(self.ts.T):
                    self.v_hat[j, :] = self.eps_df[j]*lambertw((self.lamda[j]/self.eps_df[j]) * \
                                                                (1.0/(1.0*torch.sum(self.t_idx == j))) * \
                                                                1/(K_df(j).T @ torch.exp(self.u_hat[j]/self.eps_df[j])))
                    self.v_hat[j, self.t_idx != j] = 0
                c = self.lamda_reg*self.D*torch.log(torch.sum(torch.exp((-self.w[0]*self.u_hat[0, :])/(self.lamda_reg*self.D)) * z1(0, c) * z2(0, c)))
                if i % print_interval == 0:
                    with torch.no_grad():
                        dual_obj = -self.dual_obj().item()
                        primal_obj = self.primal_obj().item()
                    print("Iteration = ", i, " Dual obj = ", dual_obj, " Primal_obj = ", primal_obj, " Gap = ", primal_obj - dual_obj, flush = True)
                    obj_vals += [(i, dual_obj), ]
                    if abs(primal_obj - dual_obj) < tol:
                        break
        return obj_vals, c
    
    def solve_lbfgs(self, max_iter = 50, steps = 10, lr = 0.01, max_eval = None, 
                    history_size = 100, line_search_fn = 'strong_wolfe', 
                    factor = 1, retry_max = 1, tol = 5e-3):
        """Solve using LBFGS (works in the general case)

        :param max_iter: max LBFGS iterations per step (passed to `torch.optim.LBFGS`)
        :param steps: number of steps 
        :param lr: learning rate (passed to `torch.optim.LBFGS`)
        :param max_eval: maximum function evals (passed to `torch.optim.LBFGS`)
        :param history_size: history size (passed to `torch.optim.LBFGS`)
        :param line_search_fn: line search function to use (passed to `torch.optim.LBFGS`)
        :param factor: if `NaN` encountered, decrease `lr` by `factor` and retry
        :param retry_max: maximum number of restarts
        :param tol: primal-dual tolerance for convergence.
        """
        obj_vals = []
        optimizer = torch.optim.LBFGS(self.parameters(), lr = lr, 
                                        history_size = history_size, 
                                        max_iter = max_iter,
                                        max_eval = max_eval, 
                                        line_search_fn=line_search_fn)
        u_hat_last = self.u_hat.clone()
        v_hat_last = self.v_hat.clone()
        retries = retry_max
        for i in range(0, steps):
            def closure():
                optimizer.zero_grad()
                obj = self.dual_obj()
                obj.backward()
                for j in range(0, self.ts.T):
                    self.v_hat.grad[j, self.t_idx != j] = 0
                return obj
            with torch.no_grad():
                dual_obj = -self.dual_obj().item()
                primal_obj = self.primal_obj().item()
            print("Iteration = ", i, " Dual obj = ", dual_obj, " Primal_obj = ", primal_obj, 
                " Gap = ", primal_obj - dual_obj, "Sum = ", self.get_R(0).sum().item())
            if abs(primal_obj - dual_obj) < tol:
                break
            if math.isnan(dual_obj):
                if retries <= 0:
                    lr = lr/factor
                    retries = retry_max
                else:
                    retries -= 1
                print("Warning: NaN encountered, restarting and scaling down learning rate. lr_new = %f" % lr)
                # restart optimization
                self.uv_init(u_hat_last.clone(), v_hat_last.clone())
                optimizer = torch.optim.LBFGS(self.parameters(), 
                                                lr = lr, 
                                                history_size = history_size, 
                                                max_iter = max_iter, 
                                                max_eval = max_eval, 
                                                line_search_fn=line_search_fn)
            else:
                obj_vals += [(i*max_iter, dual_obj), ]
                u_hat_last = self.u_hat.clone()
                v_hat_last = self.v_hat.clone()
            optimizer.step(closure = closure)
        self.uv_init(u_hat_last.clone(), v_hat_last.clone())
        return obj_vals, u_hat_last, v_hat_last
    
    def interp(self, i, coord_orig = None, R = None, R_bar = None, N = 100, interp_frac = 0.5, method = "geo"):
        """Compute displacement interpolation at time `(1-interp_frac)*t[i] + interp_frac*t[i+1]`,
            with growth.
        
        :param i: index of timepoints to interpolate 
        :param coord_orig: if not `None`, a `torch.Tensor` of alternate coordinates with row-wise 
            correspondence to `x` in which to compute the interpolation. 
        :param R: `torch.Tensor` of precomputed marginals :math:`\\mathbf{R}_{t_i}`.
        :param R_bar: `torch.Tensor` of precomputed intermediate growth marginals :math:`\\overline{\\mathbf{R}}_{t_i}`.
        """
        if any([R is None, R_bar is None]):
            with torch.no_grad():
                R = self.get_R()
                R_bar = self.get_R_bar()
        if method == "geo":
            K = self.get_K(i)
            gamma = self.get_coupling_reg(i, K)
            T = gamma @ torch.diag(R[i+1, :]/R_bar[i, :])**interp_frac
            T_norm = (T/T.sum()).flatten()
        elif method == "indep":
            T = torch.ger(R[i, :], R[i+1, :]*(R[i+1, :]/R_bar[i, :])**(interp_frac-1))
            T_norm = (T/T.sum()).flatten()
        samp = np.random.choice(T_norm.shape[0], size = N, p = T_norm.detach().cpu())
        out = torch.zeros(N, coord_orig.shape[1])
        for k in range(0, N):
            idx_i = samp[k] // T.shape[1]
            idx_j = samp[k] % T.shape[1]
            if coord_orig is not None:
                x0 = coord_orig[idx_i, :]
                x1 = coord_orig[idx_j, :]
            else:
                x0 = self.x[idx_i, :]
                x1 = self.x[idx_j, :]
            out[k, :] = x0 + interp_frac*(x1 - x0)
        return out

    def save(self, path):
        """(Experimental) save OTModel using `dill`.
        
        """
        with open(path, "wb") as f:
            dill.dump(self, f)

    @staticmethod    
    def load(path):
        """(Experimental) load OTModel using `dill`.
        
        """
        with open(path, "rb") as f:
            model = dill.load(f)
            model.kernel_init()
            return model

class OTModel_kl(OTModel):
    """Alternative OTModel for the case where the data-fitting term is pure cross-entropy
    
    :param ts: TimeSeries object containing input data.
    :param lamda_reg: regularisation strength parameter :math:`\\lambda`.
    :param D: diffusivity :math:`D`.
    :param w: `torch.Tensor` of weights for data-fitting term at each timepoint. 
            If `None`, then we take :math:`w_i = N_i/\sum_j N_j` where :math:`N_i` is the number of particles at timepoint `i`.
    :param eps: `torch.Tensor` of entropic regularisation parameters to use in the 
            regularising functional OT term. If `None`, then we take the entries to be :math:`2D\Delta t_i` 
    :param c_scale: `torch.Tensor` of cost matrix scalings :math:`\\overline{C}_i` to use in the regularising functional.
            That is, the cost matrix for the pair of timepoints :math:`(t_i, t_{i+1})` will be 
            :math:`C^{(i)}_{jk} = C_{jk}/\\overline{C}_i`.
    :param u0: `torch.Tensor` of initial values for dual variables
    :param pi_0: `torch.Tensor` of initial distribution :math:`\\pi_0` to use, 
            or else a choice of "uniform" (uniform on the space :math:`\\overline{\\mathcal{X}}`) 
            or "stationary" (stationary distribution of heat kernel on :math:`\\overline{\\mathcal{X}}`)
    :param device: Device to use with PyTorch (e.g. `torch.device("cuda:0")` in the case of GPU).
            If `None`, we use `torch.device("cpu")`. 
    :param use_keops: `True` to use KeOps for on-the-fly kernel reductions. Otherwise all kernels
            are precomputed and stored in memory. 
    """
    def __init__(self, ts, lamda_reg, 
                 D = None, w = None, 
                 eps = None,
                 c_scale = None,
                 u0 = None, pi_0 = None,
                 device = None,
                 use_keops = True):
        super(OTModel_kl, self).__init__(ts, lamda_reg, D = D, w = w, eps = eps, eps_df = torch.ones(ts.T, device = device), 
                                        c_scale = c_scale, c_scale_df = torch.ones(ts.T, device = device), pi_0 = pi_0, 
                                        device = device, use_keops = use_keops)
        # initialise model params, with defaults if None. 
        self.C = None
        self.u_hat = None
        self.v_hat = None # just in case 
        self.u_init(u0)

    def u_init(self, u0 = None):
        """Initialise dual variable `u`
        
        :param u0: initial value of `u`. If `None`, then all ones.
        """
        if u0 is None:
            u = torch.ones(self.ts.T, self.ts.x.shape[0], device = self.device)
        else:
            u = u0
        for i in range(0, self.ts.T):
            u[i, i != self.ts.t_idx] = -self.w[i]/(self.lamda_reg*self.D) # 0
        self.register_parameter(name = 'u', param = torch.nn.Parameter(Variable(u, requires_grad = True)))
        
    def F_star(self, u, i):
        """Legendre dual :math:`u \\mapsto F^*(u)` of cross-entropy data-fitting term
        :math:`x \mapsto -\sum_{k \\in \\mathrm{supp}(\\hat{\\rho}_{t_i})} \\log(x_k)`
        """
        val = (-torch.sum(self.t_idx == i) - torch.sum(torch.log(-u[self.t_idx == i])))/torch.sum(self.t_idx == i)
        if torch.isnan(val):
            val = torch.tensor(float("Inf"), device = self.device)
        return val

    def v1(self, i, log = False):
        """Compute intermediate variable `v_1` (TODO)
        
        """
        idx = np.arange(0, i)
        if not(log):
            v = torch.ones((self.u.shape[1], 1), requires_grad=True, device = self.device)
            for k in idx:
                v = self.K_ij[k].T @ ((v[:, 0]*torch.exp(self.u[k, :])).view(-1, 1)) 
            return v[:, 0]
        else:
            v = torch.ones((self.u.shape[1], 1), requires_grad=True, device = self.device)
            shift = 0
            for k in idx:
                m = self.u[k, :].mean()
                shift = shift + m
                v = self.K_ij[k].T @ ((v[:, 0]*torch.exp(self.u[k, :] - m)).view(-1, 1))
            return shift + v[:, 0].log() 

    def v2(self, i, log = False):
        """Compute intermediate variable `v_2` (TODO)
        
        """
        idx = np.arange(i+1, self.ts.T)[::-1]
        if not(log):
            v = torch.ones((self.u.shape[1], 1), requires_grad=True, device = self.device)
            for k in idx:
                v = self.K_ij[k-1] @ ((v[:, 0]*torch.exp(self.u[k, :])).view(-1, 1))
            return v[:, 0]
        else:
            v = torch.ones((self.u.shape[1], 1), requires_grad=True, device = self.device)
            shift = 0
            for k in idx:
                m = self.u[k, :].mean()
                shift = shift + m
                v = self.K_ij[k-1] @ ((v[:, 0]*torch.exp(self.u[k, :] - m)).view(-1, 1))
        return shift + v[:, 0].log()
    
    def Z(self, log = False): 
        """Compute normalising constant `Z` (TODO)
        
        """
        if not(log):
            return torch.dot(torch.exp(self.u[0, :]), self.v1(0) * self.v2(0)) 
        else:
            return torch.logsumexp(self.u[0, :] + self.v1(0, log = True) + self.v2(0, log = True), dim = 0)
    
    def dual_obj(self):
        """Compute dual objective 
        
        """
        reg_dual = torch.logsumexp(self.u[0, :] + self.v1(0, log = True) + self.v2(0, log = True), dim = 0)
        return self.lamda_reg*self.D*reg_dual + torch.sum(torch.stack([self.w[i]*self.crossent_star(-(self.lamda_reg*self.D/self.w[i])*self.u[i, :], i) for i in range(self.ts.T)])) 
    
    def primal_obj(self):
        """Compute primal objective
        """
        def reg():
            norm_const = self.Z()
            return (1/norm_const)*torch.stack([torch.dot(self.u[i, :]*torch.exp(self.u[i, :]), self.v1(i)*self.v2(i)) for i in range(0, self.ts.T)]).sum() - torch.log(norm_const)
        def xent(a, b):
            out = a*torch.log(a/b)
            out[a == 0] = 0
            return out.sum() - a.sum() + b.sum()
        def normalise(x):
            return x/x.sum()
        with torch.no_grad():
            p_all = self.get_R()
        return self.lamda_reg*self.D*reg() + torch.dot(self.w, torch.stack([xent(normalise((self.t_idx == i)*1.0), p_all[i, :]) for i in range(self.ts.T)]))

    def get_gamma_branch(self, i):
        pass

    def get_gamma_spine(self, i, K = None):
        pass
    
    def get_R(self, i = None): 
        """Get reconstructed marginal :math:`\\mathbf{R}_{t_i}` at timepoint `i`.
        
        """
        if i is None:
            return torch.stack([self.get_R(i) for i in range(0, self.ts.T)])
        else:
            return ((self.u[i, :] + self.v1(i, log = True) + self.v2(i, log = True)) - self.Z(log = True)).exp()

    def solve_lbfgs(self, max_iter = 50, steps = 10, lr = 0.01, max_eval = None, history_size = 100, line_search_fn = 'strong_wolfe', factor = 1, retry_max = 1, tol = 5e-3):
        """Solve using LBFGS

        :param max_iter: max LBFGS iterations per step (passed to `torch.optim.LBFGS`)
        :param steps: number of steps 
        :param lr: learning rate (passed to `torch.optim.LBFGS`)
        :param max_eval: maximum function evals (passed to `torch.optim.LBFGS`)
        :param history_size: history size (passed to `torch.optim.LBFGS`)
        :param line_search_fn: line search function to use (passed to `torch.optim.LBFGS`)
        :param factor: if `NaN` encountered, decrease `lr` by `factor` and retry
        :param retry_max: maximum number of restarts
        :param tol: primal-dual tolerance for convergence.
        """
        obj_vals = []
        optimizer = torch.optim.LBFGS(self.parameters(), lr = lr, history_size = history_size, max_iter = max_iter, max_eval = max_eval, line_search_fn=line_search_fn)
        u_last = self.u.clone()
        retries = retry_max
        for i in range(0, steps):
            ## 
            def closure():
                optimizer.zero_grad()
                obj = self.dual_obj()
                obj.backward()
                for j in range(0, self.ts.T):
                    self.u.grad[j, self.t_idx != j] = 0
                return obj
            ## 
            with torch.no_grad():
                dual_obj = -self.dual_obj().item()
                primal_obj = self.primal_obj().item()
            print("Iteration = ", i*max_iter, " Dual obj = ", dual_obj, " Primal_obj = ", primal_obj, " Gap = ", primal_obj - dual_obj)
            if abs(primal_obj - dual_obj) < tol:
                break
            elif math.isnan(dual_obj):
                if retries <= 0:
                    lr = lr/factor
                    retries = retry_max
                else:
                    retries -= 1

                print("Warning: NaN encountered, restarting and scaling down learning rate. lr_new = %f" % lr)
                # restart optimization
                self.u_init(u_last.clone())
                optimizer = torch.optim.LBFGS(self.parameters(), lr = lr, history_size = history_size, max_iter = max_iter, max_eval = max_eval, line_search_fn=line_search_fn)
            else:
                obj_vals += [(i*max_iter, dual_obj), ]
                u_last = self.u.clone()
            optimizer.step(closure = closure)
        self.u_init(u_last.clone())
        return obj_vals, u_last
    
class OTModel_ot(OTModel):
    def __init__(self, *args, **kwargs):
        super(OTModel_ot, self).__init__(*args, **kwargs)

    def crossent_star(self, u, i):
        # Hack for OT-only data fitting (i.e. F(a, b) = indicator(a = b))
        p = (1.0*(self.t_idx == i))
        p = p/p.sum()
        return torch.sum(p * u)
    
