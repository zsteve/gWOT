# SDE couplings estimation implementation
# Stephen Zhang (syz@math.ubc.ca)

import matplotlib.pyplot as plt
from pykeops.numpy import Genred
from pykeops.torch import LazyTensor, Vi, Vj
import torch
from torch.autograd import grad, Variable
import autograd
import autograd.numpy as np
import math
import sklearn
import ot
import os
from lambertw import lambertw
from pathos.multiprocessing import ProcessingPool as Pool
import dill
from scipy.sparse.linalg import aslinearoperator, eigsh, LinearOperator

import gwot

class OTModel_kl(gwot.OTModel):
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
            u[i, i != self.ts.time_idx] = -self.w[i]/(self.lamda*self.D) # 0
        self.register_parameter(name = 'u', param = torch.nn.Parameter(Variable(u, requires_grad = True)))
        
    def F_star(self, u, i):
        val = (-torch.sum(self.time_idx == i) - torch.sum(torch.log(-u[self.time_idx == i])))/torch.sum(self.time_idx == i)
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
        return self.lamda*self.D*reg() + torch.dot(self.w, torch.stack([xent(normalise((self.time_idx == i)*1.0), p_all[i, :]) for i in range(self.ts.T)]))
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
            self.C = sklearn.metrics.pairwise_distances(self.ts.x, self.ts.x, metric = 'sqeuclidean')
        K = torch.exp(-torch.Tensor(self.C, device = self.device)/float(self.c_scale[i]*self.eps[i]))
        K = (K.T/K.sum(1)).T
        if i == 0 and self.pi_0 is not None:
            K = (K.T * self.pi_0).T
        return K
    
    def get_K_df(self, i):
        if self.C is None:
            self.C = sklearn.metrics.pairwise_distances(self.ts.x, self.ts.x, metric = 'sqeuclidean')
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
                    self.u.grad[j, self.time_idx != j] = 0
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
    
