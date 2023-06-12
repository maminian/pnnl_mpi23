import numpy as np
import scipy.sparse.linalg as spl
import scipy.sparse as sps
from sdfs.tpfa import TPFA
from sdfs.partial_solve import PartialSolve
from time import perf_counter


class DarcyExp(object):

    def __init__(self, tpfa, iuobs=None, ssv=None):
        self.tpfa = tpfa
        self.ssv = range(self.tpfa.geom.cells.num) if ssv is None else ssv
        self.Nc = self.tpfa.geom.cells.num
        self.Nc_range = np.arange(self.Nc)
        self.cells_neighbors = self.tpfa.cell_neighbors
        self.keep = np.concatenate(
            (self.Nc_range, np.flatnonzero(self.cells_neighbors >= 0) + self.Nc))
        self.cols = np.concatenate(
            (self.Nc_range, np.tile(self.Nc_range, 4)))[self.keep]
        self.rows = np.concatenate(
            (self.Nc_range, self.cells_neighbors.ravel()))[self.keep]
        neumann_bc = (self.tpfa.bc.kind == 'N')
        Nq = np.count_nonzero(neumann_bc)
        self.dLdq = sps.csc_matrix(
            (-np.ones(Nq), (np.arange(Nq), self.tpfa.geom.cells.to_hf[2*self.tpfa.Ni:][neumann_bc])), shape=(Nq, self.Nc))
        self.iuobs = iuobs
        if self.iuobs is not None:
            self.ps = PartialSolve(sps.csc_matrix((np.ones(
                2 * self.tpfa.Ni + self.Nc), (self.tpfa.rows, self.tpfa.cols)), shape=(self.Nc, self.Nc)), iuobs)

    def randomize_bc(self, kind, scale):
        self.tpfa.bc.randomize(kind, scale)
        self.tpfa.update_rhs(kind)
        return self

    def increment_bc(self, kind, value):
        self.tpfa.bc.increment(kind, value)
        self.tpfa.update_rhs(kind)
        return self

    def solve(self, Y, q=None):
        self.K = np.exp(Y)
        self.A, b = self.tpfa.ops(self.K, q)
        return spl.spsolve(self.A, b)

    def partial_solve(self, Y, idx=None):
        if self.iuobs is None:
            raise Exception("Partial solve needs indices.")

        self.K = np.exp(Y)
        self.A, b = self.tpfa.ops(self.K)
        self.ps.update(self.A)

        if idx is not None:
            return self.ps.solve(b, idx)
        else:
            return self.ps.solve_all(b)

    def residual(self, u, Y):
        self.K = np.exp(Y)
        self.A, b = self.tpfa.ops(self.K)
        return self.A @ u - b

    def residual_sens_Y(self, u, Y):
        # call residual(self, u, Y) before residual_sens_Y(self, u, Y)
        offdiags = (u[self.cells_neighbors] - u[None, :]) * self.tpfa.sens()
        vals = np.vstack(((self.tpfa.alpha_dirichlet * u - self.tpfa.rhs_dirichlet -
                           offdiags.sum(axis=0))[None, :], offdiags)) * self.K[None, :]
        return sps.csr_matrix((vals.ravel()[self.keep], (self.rows, self.cols)), shape=(self.Nc, self.Nc))

    def residual_sens_u(self, u, Y):
        # call residual(self, u, Y) before residual_sens_u(self, u, Y)
        return self.A

    def residual_sens_p(self, u, p):
        # call residual(self, u, Y) before residual_sens_p(self, u, p)
        return sps.vstack([self.residual_sens_Y(u, p[:self.tpfa.geom.cells.num]), self.dLdq])
