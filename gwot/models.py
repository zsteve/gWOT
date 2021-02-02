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
    def __init__(self, ts, 
                 lamda, 
                 D = None, 
                 w = None, lamda_i = None, 
                 eps = None, eps_df = None, 
                 c_scale = None, c_scale_df = None,
                 m_i = None, g_i = None, kappa = None, 
                 growth_constraint = 'exact', 
                 u_hat = None, v_hat = None, pi_0 = 'stationary',
                 device = None, use_keops = True):
        super(OTModel, self).__init__()
        # if no device specified, use CPU
        self.device = torch.device("cpu") if device is None else device
        self.use_keops = use_keops
        # initialise parameters
        # basic parameters
        self.w = (1/ts.T)*torch.from_numpy(ts.N).to(self.device)/ts.N.mean() if w is None else w
        self.lamda_i = torch.ones(ts.T, device = self.device) if lamda_i is None else lamda_i
        self.ts = ts
        self.lamda = lamda
        self.D = ts.D if D is None else D
        self.t_idx = torch.from_numpy(ts.t_idx).to(self.device)
        self.dt = torch.from_numpy(ts.dt).to(self.device)
        self.eps = torch.from_numpy(2*self.D*ts.dt).to(self.device) if eps is None else eps
        self.eps_df = eps_df
        self.x = torch.from_numpy(ts.x).to(self.device)
        self.c_scale = c_scale
        self.c_scale_df = c_scale_df
        self.C = None
        # parameters related to growth
        self.m_i = m_i # masses 
        self.g_i = g_i # growth rates 
        self.kappa = kappa # growth constraint scale
        if growth_constraint not in ["exact", "KL"]:
            raise ValueError("growth_constraint must be one of 'exact' or 'KL'")
        self.growth_constraint = growth_constraint
        # pi_0, initial condition for reference process.
        if type(pi_0) is not str:
            self.pi_0 = pi_0 
        elif pi_0 == 'uniform':
            self.pi_0 = torch.ones(self.x.shape[0], device = self.device)/self.x.shape[0]
        elif pi_0 == 'stationary':
            self.pi_0 = None
        else:
            raise ValueError("pi0 must either be a torch.Tensor, or one of 'uniform' or 'stationary'")
        # init kernels
        self.kernel_init()
        self.uv_init(u_hat, v_hat)

    def kernel_init(self,):
        def get_pi0_unif(use_keops):
            # not recommended to use this in practice, since the eigengap is generally very small
            if use_keops:
                x_cpu = self.x.cpu() # scipy needs CPU location
                x_cpu_i = LazyTensor(x_cpu.view(1, x_cpu.shape[0], x_cpu.shape[1]).numpy())
                x_cpu_j = LazyTensor(x_cpu.view(x_cpu.shape[0], 1, x_cpu.shape[1]).numpy())
                D_ij = ((x_cpu_i - x_cpu_j)**2/(self.eps[0]*self.c_scale[0]).item()).sum(2) # scaling doesn't matter
                M_ij = (-D_ij).exp()
                K_ij = M_ij*LazyTensor(1/M_ij.sum(dim = 1), axis = 1).T
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
            M_ij = [(-D).exp() for D in D_ij] # Kernel matrix
            M_sums_i = [LazyTensor(1/M.sum(dim = 1), axis = 1).T for M in M_ij]
            self.K_ij = [M_ij[i]*M_sums_i[i] for i in range(0, len(M_ij))]
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
            M_ij = [(-D).exp() for D in D_ij]
            self.K_ij = [(M.T/M.sum(dim = 1)).T for M in M_ij]
            if self.pi_0 is None:
                get_pi0_unif(self.K_ij[0].cpu())
            self.K_ij[0] = self.pi_0.reshape(-1, 1) * self.K_ij[0]
            # now construct kernel for data fitting 
            D_ij_df = [C/(self.eps_df[i]*self.c_scale_df[i]) for i in range(0, self.ts.T)]
            self.K_ij_df = [(-D).exp() for D in D_ij_df]
        
    def uv_init(self, u_hat = None, v_hat = None):
        # Initialise model variables (u_hat, v_hat).
        if u_hat is None:
            u_hat = torch.zeros(self.ts.T, self.ts.x.shape[0], device = self.device)
        if v_hat is None:
            v_hat = torch.zeros(self.ts.T, self.ts.x.shape[0], device = self.device)
        for i in range(0, self.ts.T):
            v_hat[i, i != self.ts.t_idx] = -self.lamda_i[i]
        self.register_parameter(name = 'u_hat', param = torch.nn.Parameter(Variable(u_hat, requires_grad = True)))
        self.register_parameter(name = 'v_hat', param = torch.nn.Parameter(Variable(v_hat, requires_grad = True)))
        
    def forward(self):
        return self.dual_obj()

    def Xent_star(self, u, i):
        # Legendre transform of cross-entropy data-fitting term
        val = -torch.sum(torch.log(1 - u[self.t_idx == i]))/torch.sum(self.t_idx == i)
        if torch.isnan(val):
            val = torch.tensor(float("Inf"), device = self.device)
        return val
    
    def logKexp(self, K, x):
        # returns log(K @ exp(x))
        if self.use_keops:
            if type(x) is not pykeops.torch.LazyTensor:
                x = LazyTensor(x.view(1, -1, 1))
            return x.logsumexp(dim = 1, weight = K).view(-1)
        else:
            scale = x.max()
            return (K @ (x - scale).exp()).log() + scale

    def logsumexp_weight(self, w, x, dim = 1):
        # returns log(w * exp(x))
        if self.use_keops:
            if type(x) is not pykeops.torch.LazyTensor:
                x = LazyTensor(x)
            if type(w) is not pykeops.torch.LazyTensor:
                w = LazyTensor(w)
            return x.logsumexp(weight = w, dim = dim).view(-1)
        else:
            return torch.log((w * torch.exp(x)).sum())

    def compute_phi(self, i = None, out_arr = None):
        # Compute recurrence for phi. This is used for gradient based methods.
        if i is None:
            # i = None, so we compute phi for all times.
            self.compute_phi(i = 0, out_arr = out_arr)
            return None
        elif i == self.ts.T-2:
            psi_next = (-self.w[i+1]*self.u_hat[i+1, :])/self.lamda
        else:
            phi_next = self.compute_phi(i = i+1, out_arr = out_arr)
            v_next = (-self.m_i[i+1]/self.m_i[i+2])*self.dt[i+1]*phi_next
            u_next = (-self.eps[i+1])*self.logKexp(self.K_ij[i+1], v_next/self.eps[i+1]).view(-1)
            psi_next = (-1/self.lamda)*(self.lamda*u_next/self.dt[i+1] + self.w[i+1]*self.u_hat[i+1, :])
        if self.growth_constraint == "KL":
            phi = self.kappa[i]*self.g_i[i, :]*(1 - torch.exp(psi_next/self.kappa[i]))
        elif self.growth_constraint == "exact":
            phi = -self.g_i[i, :]*psi_next
        else:
            raise Exception("growth_constraint not 'KL' or 'exact'")
        if out_arr is not None:
            out_arr[i, :] = phi
        return phi

    def dual_obj(self):
        # Evaluate dual objective
        u0 = (-self.w[0]*self.dt[0]*self.u_hat[0, :])/self.lamda
        v0 = (-self.m_i[0]/self.m_i[1])*self.dt[0]*self.compute_phi(i = 0)
        # v0_j = LazyTensor(v0.view(1, v0.shape[0], 1))
        # u0_j = LazyTensor(u0.view(1, u0.shape[0], 1))
        # w = (v0_j/self.eps[0]).logsumexp(dim = 1, weight = self.K_ij[0]).view(-1)
        w = self.logKexp(self.K_ij[0], v0/self.eps[0])
        # w_j = LazyTensor(w.view(1, w.shape[0], 1))
        # reg = self.eps[0]*w_j.logsumexp(dim = 1, weight = (u0_j/self.eps[0]).exp()).view(-1)[0]
        reg = self.eps[0]*self.logsumexp_weight((u0/self.eps[0]).exp().view(1, -1, 1), w.view(1, -1, 1), dim = 1)
        return (self.lamda/(self.m_i[0]*self.dt[0]))*reg + \
                torch.dot(self.w/self.m_i, 
                        torch.stack([self.eps_df[i] * torch.dot(torch.exp(self.u_hat[i, :]/self.eps_df[i]), 
                                                                self.K_ij_df[i] @ torch.exp(self.v_hat[i, :]/self.eps_df[i])) 
                                    for i in range(0, self.ts.T)])) + \
                torch.dot(self.lamda_i * self.w, 
                          torch.stack([self.Xent_star(-self.v_hat[i, :]/self.lamda_i[i], i) 
                                       for i in range(0, self.ts.T)]))

    def primal_obj(self, terms = False):
        # Evaluate primal objective
        # Define first some helper functions
        def eval_primal_OT(model, phi_all):
            u = (-model.w[0]*model.dt[0]/model.lamda)*model.u_hat[0, :]
            v = (-model.m_i[0]/model.m_i[1])*model.dt[0]*phi_all[0, :].to(self.device)
            # u_i = LazyTensor(u.view(u.shape[0], 1, 1))
            # v_i = LazyTensor((model.K_ij[0] @ (v/self.eps[0]).exp()).view(u.shape[0], 1, 1))
            # Z = (u_i/model.eps[0]).logsumexp(dim = 0, weight = v_i).item()
            Z = self.logsumexp_weight((model.K_ij[0] @ (v/self.eps[0]).exp()).view(-1, 1, 1), 
                                        (u/model.eps[0]).view(-1, 1, 1), dim = 0).item()
            return torch.dot(u*torch.exp(u/self.eps[0] - Z), model.K_ij[0] @ (v/self.eps[0]).exp()) + \
                    torch.dot(v*torch.exp(v/self.eps[0] - Z), model.K_ij[0].T @ (u/self.eps[0]).exp()) - \
                    self.eps[0]*Z
        def eval_primal_OT_tilde(model, i, phi_all):
            p = model.get_P(i)
            v = (-model.m_i[i]/model.m_i[i+1])*model.dt[i]*phi_all[i, :].to(self.device)
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
        def eval_primal_Xent_df(model, i):
            p_hat = model.get_P_hat(i)
            N = 1.0*(model.t_idx == i).sum()
            return ((1/N)*(-torch.log(p_hat[model.t_idx == i]/model.m_i[i]) - torch.log(N))).sum() - 1 + \
                        p_hat.sum()/model.m_i[i]
        def eval_primal_growth_KL(model, i, R):
            r = R[i, :]
            p = model.get_P(i+1)
            return model.kappa[i]*KL(p, model.g_i[i, :]*r)
        # precompute all phi values for efficiency
        phi_all = torch.zeros(self.ts.T-1, self.x.shape[0], device = self.device)
        self.compute_phi(out_arr = phi_all)
        Xent_df_all = torch.stack([eval_primal_Xent_df(self, i) for i in range(0, self.ts.T)])
        if self.growth_constraint == 'KL':
            R_all = self.get_R(phi_all = phi_all)
            growth = torch.stack([eval_primal_growth_KL(self, i, R_all) for i in range(0, self.ts.T-1)])
        spine_all = torch.stack([eval_primal_OT(self, phi_all)] + \
                            [eval_primal_OT_tilde(self, i, phi_all) for i in range(1, self.ts.T - 1)])
        branches_all = torch.stack([eval_primal_OT_df(self, i) for i in range(0, self.ts.T)])
        if terms:
            # return individual terms of the primal
            if self.growth_constraint == 'exact':
                return {"reg" : (1/(self.dt*self.m_i[0:-1]) * (spine_all)), 
                        "branches" : 1/self.m_i * branches_all, 
                        "df" : Xent_df_all}
            else:
                return {"reg" : (1/(self.dt*self.m_i[0:-1]) * (spine_all)), 
                        "branches" : 1/self.m_i * branches_all, 
                        "growth" : (growth/self.m_i[1:]),
                        "df" : Xent_df_all}
        else:
            # return the primal objective (scalar)
            if self.growth_constraint == 'exact':
                return (self.lamda/(self.dt*self.m_i[0:-1]) * spine_all).sum() + \
                        (self.w * (1/self.m_i * branches_all + self.lamda_i * Xent_df_all)).sum()
            else:
                return self.lamda * ((spine_all/(self.dt*self.m_i[0:-1])).sum() + (growth/self.m_i[1:]).sum()) + \
                        (self.w * (1/self.m_i * branches_all + self.lamda_i * Xent_df_all)).sum()

    def get_coupling_branch(self, i):
        # branch (data-fitting) coupling corresponding to time t_i.
        return torch.diag(torch.exp(self.u_hat[i, :]/self.eps_df[i])) @ \
                (self.K_ij_df[i] @ torch.diag(torch.exp(self.v_hat[i, :]/self.eps_df[i])))

    def get_coupling_spine(self, i, K = None):
        # spine (regulariser) coupling corresponding to times (t_i, t_(i+1))
        K = self.K_ij[i] if K is None else K
        if i > 0:
            v = (-self.m_i[i]/self.m_i[i+1])*self.dt[i]*self.compute_phi(i)
            g = v/self.eps[i]
            # g_j = LazyTensor(g.view(1, g.shape[0], 1))
            # x = g_j.logsumexp(dim = 1, weight =self.K_ij[i]).view(-1)
            x = self.logKexp(self.K_ij[i], g.view(1, -1, 1)).view(-1)
            # x_ = (K @ torch.exp(v/self.eps[i])).log()
            alpha = self.get_P(i)
            if type(K) is pykeops.torch.LazyTensor:
                # return LazyTensor(alpha.view(alpha.shape[0], 1), axis = 0) * K * LazyTensor((g.view(1, g.shape[0]) - x.view(x.shape[0], 1)).exp())
                return LazyTensor(alpha.view(-1, 1), axis = 0) * K * \
                        (LazyTensor(g.view(1, -1, 1)) - LazyTensor(x.view(-1, 1, 1))).exp()
            else:
                return alpha.view(-1, 1) * K * (g.view(1, -1) - x.view(-1, 1)).exp()
        elif i == 0:
            u0 = (-self.w[0]*self.dt[0]*self.u_hat[0, :])/(self.lamda)
            v0 = (-self.m_i[0]/self.m_i[1])*self.dt[0]*self.compute_phi(i = 0)
            # g_j = LazyTensor((v0/self.eps[0]).view(1, v0.shape[0], 1))
            # f_i = LazyTensor((u0/self.eps[0]).view(u0.shape[0], 1, 1))
            # x = g_j.logsumexp(dim = 1, weight = self.K_ij[0])
            x = self.logKexp(self.K_ij[0], (v0/self.eps[0]).view(1, -1, 1))
            # x_i = LazyTensor(x.view(u0.shape[0], 1, 1))
            # logZ = x_i.logsumexp(dim = 0, weight = f_i.exp())           
            logZ = self.logsumexp_weight(torch.exp(u0/self.eps[0]).view(-1, 1, 1), x.view(-1, 1, 1), dim = 0).item()
            if type(K) is pykeops.torch.LazyTensor:
                return LazyTensor((u0/self.eps[0]).view(-1, 1, 1)).exp() * K * \
                        (LazyTensor((v0/self.eps[0]).view(1, -1, 1)) - logZ).exp()
            else:
                return (((K @ torch.diag(torch.exp(v0/self.eps[0] - logZ))).T * torch.exp(u0/self.eps[0])).T)
 
    def get_P_mass(self, i = None): 
        # get total mass of reconstructed marginal at time t_i
        if i is None:
            return torch.stack([self.get_P_mass(i) for i in range(0, self.ts.T)])
        else:
            return torch.sum(torch.exp(self.u_hat[i, :]/self.eps_df[i]) * \
                                (self.K_ij_df[i] @ torch.exp(self.v_hat[i, :]/self.eps_df[i])))

    def get_P(self, i = None): 
        # get reconstructed marginal at time t_i
        if i is None:
            return torch.stack([self.get_P(i) for i in range(0, self.ts.T)])
        else:
            return torch.exp(self.u_hat[i, :]/self.eps_df[i]) * (self.K_ij_df[i] @ torch.exp(self.v_hat[i, :]/self.eps_df[i]))

    def get_P_hat(self, i = None):
        # get intermediate marginal at time t_i
        if i is None:
            return torch.stack([self.get_P_hat(i) for i in range(0, self.ts.T)])
        else:
            return torch.exp(self.v_hat[i, :]/self.eps_df[i]) * (self.K_ij_df[i].T @ torch.exp(self.u_hat[i, :]/self.eps_df[i]))

    def get_R(self, i = None, phi_all = None):
        if phi_all is None:
            phi_all = torch.zeros(self.ts.T-1, self.x.shape[0], device = self.device)
            self.compute_phi(out_arr = phi_all) 
        v_all = (phi_all.T * ((-self.m_i[0:-1]/self.m_i[1:])*self.dt)).T
        p_all = self.get_P()
        if i is None:
            return torch.stack([(torch.exp(v_all[i, :]/self.eps[i])) * \
                                (self.K_ij[i].T @ (p_all[i, :] / (self.K_ij[i] @ torch.exp(v_all[i, :]/self.eps[i])))) \
                            for i in range(0, self.ts.T-1)])
        else:
            return (torch.exp(v_all[i, :]/self.eps[i])) * \
                    (self.K_ij[i].T @ (p_all[i, :] / (self.K_ij[i] @ torch.exp(v_all[i, :]/self.eps[i]))))

    def get_K(self, i):
        if self.C is None:
            self.C = sklearn.metrics.pairwise_distances(self.ts.x, metric = 'sqeuclidean')
        K = torch.exp(-torch.from_numpy(self.C).to(self.device)/float(self.c_scale[i]*self.eps[i]))
        K = (K.T/K.sum(1)).T
        if i == 0 and self.pi_0 is not None:
            K = (K.T * self.pi_0).T
        return K
    
    def get_K_df(self, i):
        if self.C is None:
            self.C = sklearn.metrics.pairwise_distances(self.ts.x, metric = 'sqeuclidean')
        K_df = torch.exp(-torch.from_numpy(self.C).to(self.device)/float(self.c_scale_df[i]*self.eps_df[i]))
        K_df = (K_df.T/K_df.sum(1)).T
        return K_df
    
    def solve_sinkhorn(self, steps = 1000, tol = 5e-3, precompute_K = False, print_interval = 25):
        # Block-optimisation (Sinkhorn-like) optimisation
        if self.use_keops and precompute_K:
            K_prec = [self.get_K(i) for i in range(self.ts.T-1)]
            K_df_prec = [self.get_K_df(i) for i in range(self.ts.T)]
        elif ~self.use_keops and precompute_K:
            # ignore precompute if not using Keops
            precompute_K = False
        c = 0 # mass constraint dual variable
        def U(i, c):
            return (-float(i == 0)*c - self.w[i]*self.u_hat[i, :])/(self.lamda*self.D)
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
        # disable gradient
        # self.u_hat.require_grad = False
        # self.v_hat.require_grad = False
        # 
        with torch.no_grad():
            for i in range(steps):
                # update u_hat
                for j in range(self.ts.T):
                    self.u_hat[j, :] = 1/(1/self.eps_df[j] + self.w[j]/(self.lamda*self.D)) * \
                                    (torch.log(z1(j, c)*z2(j, c)/(K_df(j) @ torch.exp(self.v_hat[j, :]/self.eps_df[j]))) \
                                        - float(j == 0)*c/(self.lamda*self.D) )
                for j in range(self.ts.T):
                    self.v_hat[j, :] = self.eps_df[j]*lambertw((self.lamda_i[j]/self.eps_df[j]) * \
                                                                (1.0/(1.0*torch.sum(self.t_idx == j))) * \
                                                                1/(K_df(j).T @ torch.exp(self.u_hat[j]/self.eps_df[j])))
                    self.v_hat[j, self.t_idx != j] = 0
                c = self.lamda*self.D*torch.log(torch.sum(torch.exp((-self.w[0]*self.u_hat[0, :])/(self.lamda*self.D)) * z1(0, c) * z2(0, c)))
                if i % print_interval == 0:
                    with torch.no_grad():
                        dual_obj = -self.dual_obj().item()
                        primal_obj = self.primal_obj().item()
                    print("Iteration = ", i, " Dual obj = ", dual_obj, " Primal_obj = ", primal_obj, " Gap = ", primal_obj - dual_obj, flush = True)
                    obj_vals += [(i, dual_obj), ]
                    if abs(primal_obj - dual_obj) < tol:
                        break
        return obj_vals, c
    
    def solve_lbfgs(self, max_iter = 50, steps = 10, lr = 0.01, max_eval = None, history_size = 100, line_search_fn = 'strong_wolfe', factor = 1, retry_max = 1, tol = 5e-3):
        # Gradient-based solution with L-BFGS
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
            ## 
            def closure():
                optimizer.zero_grad()
                obj = self.dual_obj()
                obj.backward()
                for j in range(0, self.ts.T):
                    self.v_hat.grad[j, self.t_idx != j] = 0
                return obj
            ## 
            with torch.no_grad():
                dual_obj = -self.dual_obj().item()
                primal_obj = self.primal_obj().item()
            print("Iteration = ", i, " Dual obj = ", dual_obj, " Primal_obj = ", primal_obj, 
                " Gap = ", primal_obj - dual_obj, "Sum = ", self.get_P(0).sum().item())
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
    
    def interp(self, i, coord_orig = None, P = None, R = None, N = 100, interp_frac = 0.5, method = "geo"):
        if any([P is None, R is None]):
            with torch.no_grad():
                P = self.get_P()
                R = self.get_R()
        if method == "geo":
            K = self.get_K(i)
            gamma = self.get_coupling_spine(i, K)
            T = gamma @ torch.diag(P[i+1, :]/R[i, :])**interp_frac
            T_norm = (T/T.sum()).flatten()
        elif method == "indep":
            T = torch.ger(P[i, :], P[i+1, :]*(P[i+1, :]/R[i, :])**(interp_frac-1))
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
        with open(path, "wb") as f:
            dill.dump(self, f)

    @staticmethod    
    def load(path):
        with open(path, "rb") as f:
            model = dill.load(f)
            model.kernel_init()
            return model



class OTModel_kl(OTModel):
    def __init__(self, ts, lamda, 
                 D = None, w = None, 
                 eps = None,
                 c_scale = None,
                 u0 = None, pi_0 = None,
                 device = None,
                 use_keops = True):
        super(OTModel_kl, self).__init__(ts, lamda, D = D, w = w, eps = eps, eps_df = torch.ones(ts.T, device = device), 
                                        c_scale = c_scale, c_scale_df = torch.ones(ts.T, device = device), pi_0 = pi_0, 
                                        device = device, use_keops = use_keops)
        # initialise model params, with defaults if None. 
        self.C = None
        self.u_hat = None
        self.v_hat = None # just in case 
        self.u_init(u0)

    def u_init(self, u0 = None):
        # Initialise model variables (u_hat, v_hat).
        if u0 is None:
            u = torch.ones(self.ts.T, self.ts.x.shape[0], device = self.device)
        else:
            u = u0
        for i in range(0, self.ts.T):
            u[i, i != self.ts.t_idx] = -self.w[i]/(self.lamda*self.D) # 0
        self.register_parameter(name = 'u', param = torch.nn.Parameter(Variable(u, requires_grad = True)))
        
    def F_star(self, u, i):
        val = (-torch.sum(self.t_idx == i) - torch.sum(torch.log(-u[self.t_idx == i])))/torch.sum(self.t_idx == i)
        if torch.isnan(val):
            val = torch.tensor(float("Inf"), device = self.device)
        return val

    def v1(self, i, log = False):
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
        if not(log):
            return torch.dot(torch.exp(self.u[0, :]), self.v1(0) * self.v2(0)) 
        else:
            return torch.logsumexp(self.u[0, :] + self.v1(0, log = True) + self.v2(0, log = True), dim = 0)
    
    def dual_obj(self):
        # this will be numerically unstable 
        # reg_dual = torch.ones(self.x.shape[0]).cuda()
        # for i in range(self.ts.T-1, 0, -1):
        #     reg_dual = self.K_ij[i-1] @ (self.u[i, :].exp() * reg_dual)
        # reg_dual = self.u[0, :].exp().T @ reg_dual
        # reg_dual = reg_dual.log()
        reg_dual = torch.logsumexp(self.u[0, :] + self.v1(0, log = True) + self.v2(0, log = True), dim = 0)
        return self.lamda*self.D*reg_dual + torch.sum(torch.stack([self.w[i]*self.Xent_star(-(self.lamda*self.D/self.w[i])*self.u[i, :], i) for i in range(self.ts.T)])) 
    
    def primal_obj(self, terms = False):
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
            p_all = self.get_P()
        return self.lamda*self.D*reg() + torch.dot(self.w, torch.stack([xent(normalise((self.t_idx == i)*1.0), p_all[i, :]) for i in range(self.ts.T)]))

    def get_gamma_branch(self, i):
        pass

    def get_gamma_spine(self, i, K = None):
        pass
    
    def get_P(self, i = None): 
        # get reconstructed marginal at time t_i
        if i is None:
            return torch.stack([self.get_P(i) for i in range(0, self.ts.T)])
        else:
            return ((self.u[i, :] + self.v1(i, log = True) + self.v2(i, log = True)) - self.Z(log = True)).exp()

    def get_K(self, i):
        if self.C is None:
            self.C = sklearn.metrics.pairwise_distances(self.ts.x, metric = 'sqeuclidean')
        K = torch.exp(-torch.Tensor(self.C, device = self.device)/float(self.c_scale[i]*self.eps[i]))
        K = (K.T/K.sum(1)).T
        if i == 0 and self.pi_0 is not None:
            K = (K.T * self.pi_0).T
        return K
    
    def get_K_df(self, i):
        if self.C is None:
            self.C = sklearn.metrics.pairwise_distances(self.ts.x, metric = 'sqeuclidean')
        K_df = torch.exp(-torch.Tensor(self.C, device = self.device)/float(self.c_scale_df[i]*self.eps_df[i]))
        K_df = (K_df.T/K_df.sum(1)).T
        return K_df
    
    def solve_lbfgs(self, max_iter = 50, steps = 10, lr = 0.01, max_eval = None, history_size = 100, line_search_fn = 'strong_wolfe', factor = 1, retry_max = 1, tol = 5e-3):
        # Gradient-based solution with L-BFGS
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

    def Xent_star(self, u, i):
        # Code for OT-only data fitting (i.e. F(a, b) = indicator(a = b))
        p = (1.0*(self.t_idx == i))
        p = p/p.sum()
        return torch.sum(p * u)
    
