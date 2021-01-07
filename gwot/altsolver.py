import torch
from torch.autograd import grad, Variable
import math

def solve_adam(self, steps = 250, print_steps = 10, lr = 0.001, factor = 1, retry_max = 1, tol = 5e-3):
    obj_vals = []
    optimizer = torch.optim.Adam(self.parameters(), lr = lr)
    # optimizer = torch.optim.SGD(self.parameters(), lr = lr, momentum = 0.9)
    u_hat_last = self.u_hat.clone()
    v_hat_last = self.v_hat.clone()
    retries = retry_max

    lamda_i_mask = torch.stack([self.lamda_i, ]*self.x.shape[0]).T

    for i in range(0, steps):
        ## 
        def closure():
            optimizer.zero_grad()
            obj = self.dual_obj()
            obj.backward()
            for j in range(0, self.sim.T):
                self.v_hat.grad[j, self.time_idx != j] = 0
            return obj
        ## 
        if i % print_steps == 0:
            with torch.no_grad():
                dual_obj = -self.dual_obj().item()
                primal_obj = self.primal_obj().item()
            print("Iteration = ", i, " Dual obj = ", dual_obj, " Primal_obj = ", primal_obj, " Gap = ", primal_obj - dual_obj, "Sum = ", self.get_P(0).sum().item())
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
                optimizer = torch.optim.Adam(self.parameters(), lr = lr)
                # optimizer = torch.optim.SGD(self.parameters(), lr = lr, momentum = 0.9)
            else:
                obj_vals += [(i, dual_obj), ]
                u_hat_last = self.u_hat.clone()
                v_hat_last = self.v_hat.clone()

        optimizer.step(closure = closure)

    self.uv_init(u_hat_last.clone(), v_hat_last.clone())
    return obj_vals, u_hat_last, v_hat_last
