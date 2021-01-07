# Stephen Zhang (syz@math.ubc.ca)
import torch
from torch.autograd import grad, Variable
from pykeops.torch import LazyTensor, Vi, Vj
import math
import sklearn
import ot
import os
from lambertw import lambertw
import dill
from scipy.sparse.linalg import aslinearoperator, eigs, LinearOperator
from scipy import matrix

import gwot

class OTModel_ot(gwot.OTModel):
    def __init__(self, *args, **kwargs):
        super(OTModel_ot, self).__init__(*args, **kwargs)

    def Xent_star(self, u, i):
        # Code for OT-only data fitting (i.e. F(a, b) = indicator(a = b))
        p = (1.0*(self.time_idx == i))
        p = p/p.sum()
        return torch.sum(p * u)
    
