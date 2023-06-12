import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spl
from time import perf_counter

class DASAExp2(object):

    def __init__(self, solvefun, objfun, obj_sens_state, obj_sens_param, res_sens_state, res_sens_param):
        self.objfun = objfun
        self.solvefun = solvefun
        self.obj_sens_state = obj_sens_state
        self.obj_sens_param = obj_sens_param
        self.res_sens_state = res_sens_state
        self.res_sens_param = res_sens_param
        self.reset_timer()

    def reset_timer(self):
        self.solve_time = 0.0
        self.obj_time = 0.0
        self.dhdu_time = 0.0
        self.dhdp_time = 0.0
        self.dLdu_time = 0.0
        self.dLdp_time = 0.0
        self.adj_time = 0.0
        self.sens_time = 0.0
        self.grad_time = 0.0
        self.jac_time = 0.0

    def obj(self, p):
        time_start = perf_counter()
        u = self.solvefun(p)
        self.solve_time += perf_counter() - time_start
        time_start = perf_counter()
        obj = self.objfun(u, p)
        self.obj_time += perf_counter() - time_start
        return obj

    def grad(self, p):
        #u = self.solvefun(p)
        dhdu = self.obj_sens_state(u, p)
        dhdp = self.obj_sens_param(u, p)
        dLdu = self.res_sens_state(u, p)
        dLdp = self.res_sens_param(u, p)
        adj = -spl.spsolve(dLdu, dhdu)
        sens = dLdp.dot(adj)
        sens = sens + dhdp
        return sens


class DASAExp(object):

    def __init__(self, objfun, obj_sens_state, obj_sens_param, solvefun, res_sens_state, res_sens_param):
        self.objfun   = objfun
        self.solvefun = solvefun
        self.obj_sens_state = obj_sens_state
        self.obj_sens_param = obj_sens_param
        self.res_sens_state = res_sens_state
        self.res_sens_param = res_sens_param
        self.reset_timer()

    def reset_timer(self):
        self.solve_time = 0.0
        self.obj_time = 0.0
        self.dhdu_time = 0.0
        self.dhdp_time = 0.0
        self.dLdu_time = 0.0
        self.dLdp_time = 0.0
        self.adj_time = 0.0
        self.sens_time = 0.0
        
    def obj(self, p):
        time_start = perf_counter()
        u = self.solvefun(p)
        self.solve_time += perf_counter() - time_start
        time_start = perf_counter()
        obj = self.objfun(u, p)
        self.obj_time += perf_counter() - time_start
        return obj

    def grad(self, p):
        u = self.solvefun(p)
        dhdu = self.obj_sens_state(u, p)
        dhdp = self.obj_sens_param(u, p)
        dLdu = self.res_sens_state(u, p)
        dLdp = self.res_sens_param(u, p)
        adj  = -spl.spsolve((dLdu.T).tocsc(), dhdu)
        sens = dLdp.dot(adj)
        sens = sens + dhdp
        return sens

class DASAExpLM2(DASAExp2):

    def __init__(self, solvefun, objfun, obj_sens_state, obj_sens_param, res_sens_state, res_sens_param, jac_size, top_size, init_jac=True):
        super().__init__(solvefun, objfun, obj_sens_state,
                         obj_sens_param, res_sens_state, res_sens_param)
        self.top_size = top_size
        self.jac = np.zeros(jac_size)
        print(self.jac.shape)
        print(self.top_size)
        print(self.obj_sens_param(0,0).todense().shape)
        if init_jac:
            self.jac[self.top_size:, :] = self.obj_sens_param(0, 0).todense()

    def grad(self, p):
        u = self.solvefun(p)
        time_start_all = perf_counter()
        time_start = perf_counter()
        dhdu = self.obj_sens_state(u, p)
        self.dhdu_time += perf_counter() - time_start
        time_start = perf_counter()
        dLdu = self.res_sens_state(u, p)
        self.dLdu_time += perf_counter() - time_start
        time_start = perf_counter()
        dLdp = self.res_sens_param(u, p)
        self.dLdp_time += perf_counter() - time_start
        time_start = perf_counter()
        # dLdu = prob.A is symmetric
        adj = -spl.spsolve(dLdu, dhdu.T)
        sens = dLdp.T @ adj
        self.grad_time += perf_counter() - time_start
        self.jac[:self.top_size, :] = sens.T.todense()
        self.jac_time += perf_counter() - time_start_all
        return self.jac

class DASAExpLM(DASAExp):

    def grad(self, p):
        u = self.solvefun(p)
        time_start = perf_counter()
        dhdu = self.obj_sens_state(u, p)
        self.dhdu_time += perf_counter() - time_start
        time_start = perf_counter()
        dhdp = self.obj_sens_param(u, p)
        self.dhdp_time += perf_counter() - time_start
        time_start = perf_counter()
        dLdu = self.res_sens_state(u, p)
        self.dLdu_time += perf_counter() - time_start
        time_start = perf_counter()
        dLdp = self.res_sens_param(u, p)
        self.dLdp_time += perf_counter() - time_start
        time_start = perf_counter()
        # dLdu = prob.A is symmetric
        adj  = -spl.spsolve(dLdu, dhdu.T)
        self.adj_time += perf_counter() - time_start
        time_start = perf_counter()
        sens = dLdp.dot(adj)
        self.sens_time += perf_counter() - time_start
        return sps.vstack([sens.T, dhdp]).todense()

class DASAExpKL(DASAExpLM):
    
    def __init__(self, objfun, obj_sens_state, obj_sens_param, solvefun, res_sens_state, res_sens_param, const_term, param_sens_coeff):
        super().__init__(objfun, obj_sens_state, obj_sens_param, solvefun, res_sens_state, res_sens_param)
        self.const_term = const_term
        self.param_sens_coeff = param_sens_coeff
    
    def obj(self, xi):
        p = self.const_term + self.param_sens_coeff @ xi
        return super().obj(p)

    def grad(self, xi):
        p = self.const_term + self.param_sens_coeff @ xi
        dfdp = super().grad(p)
        return dfdp.dot(self.param_sens_coeff)

class DASAExpKL2(DASAExpLM2):

    def __init__(self, solvefun, objfun, obj_sens_state, obj_sens_param, res_sens_state, res_sens_param, jac_size, top_size, const_term, param_sens_coeff,):
        super().__init__(solvefun, objfun, obj_sens_state, obj_sens_param,
                         res_sens_state, res_sens_param, jac_size, top_size, False)
        self.const_term = const_term
        self.param_sens_coeff = param_sens_coeff.copy()
        #self.jac[self.top_size:, :] = self.obj_sens_param(0, 0) @ self.param_sens_coeff

    def obj(self, xi):
        p = self.const_term + self.param_sens_coeff @ xi
        return super().obj(p)

    def grad(self, xi):
        p = self.const_term + self.param_sens_coeff @ xi
        u = self.solvefun(p)
        time_start_all = perf_counter()
        time_start = perf_counter()
        dLdp = self.res_sens_param(u, p)
        self.dLdp_time += perf_counter() - time_start
        time_start = perf_counter()
        time_start_grad = perf_counter()
        dhdu = self.obj_sens_state(u, p)
        self.dhdu_time += perf_counter() - time_start
        #time_start = perf_counter()
        dLdu = self.res_sens_state(u, p)
        #self.dLdu_time += perf_counter() - time_start
        time_start = perf_counter()
        # dLdu = prob.A is symmetric
        adj = -spl.spsolve(dLdu, dhdu.T)
        self.adj_time += perf_counter() - time_start
        time_start = perf_counter()
        sens = adj.T @ dLdp
        self.sens_time += perf_counter() - time_start
        self.grad_time += perf_counter() - time_start_grad
        #print(f'grad time elasped: {perf_counter() - time_start_grad}')
        #self.jac[:self.top_size, :] = sens.todense() @ self.param_sens_coeff
        self.jac = sens @ self.param_sens_coeff
        self.jac_time += perf_counter() - time_start_all
        return self.jac
