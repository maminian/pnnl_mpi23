from os import pathsep
from typing import Optional
from numba import njit, prange
from numba.np.ufunc import parallel
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spl
import sksparse.cholmod as cm
from time import perf_counter


@njit
def partial_spsolve(L, iL, jL, b, cls_left):
    for j in range(b.shape[0]):
        b[j] /= L[jL[j]]
        b[iL[jL[j]+1:jL[j+1]]] -= L[jL[j]+1:jL[j+1]] * b[j]

    for j in cls_left:
        b[j] -= np.dot(L[jL[j]+1:jL[j+1]], b[iL[jL[j]+1:jL[j+1]]])
        b[j] /= L[jL[j]]


class PartialSolve(object):

    def __init__(self, A, idx: np.ndarray) -> None:
        self.idx = idx
        self.factor = cm.analyze(
            A, mode="auto", ordering_method="nesdis", use_long=False)
        self.n = A.shape[0]
        self.P = self.factor.P()
        self.Pt = np.argsort(self.P)[self.idx]
        self.L = None
        self.x = np.zeros(self.n)
        self.closure_left = np.zeros((self.n, idx.size), dtype=bool)
        self.closure_left[self.Pt, np.arange(idx.size)] = True
        self.closure_ready = False
        partial_spsolve(np.ones(3), np.arange(2, dtype=np.int32), np.arange(
            3, dtype=np.int32), np.ones(2), np.flip(np.zeros(2, dtype=np.int64)))

    def update(self, A):
        self.factor.cholesky_inplace(A)
        self.L = self.factor.L()
        if not self.closure_ready:
            for j in range(self.n-1):
                self.closure_left[self.L.indices[self.L.indptr[j]+1]
                                  ] |= self.closure_left[j]
            self.sparsity_left = sps.csc_matrix(self.closure_left)
            self.sparsity_left_all = np.flip(np.flatnonzero(
                self.sparsity_left @ np.ones(self.idx.size)))
            self.closure_ready = True
        return self

    def solve(self, b: np.ndarray, j: int):
        self.x = b[self.P]
        partial_spsolve(self.L.data, self.L.indices, self.L.indptr,
                        self.x, np.flip(self.sparsity_left.indices[self.sparsity_left.indptr[j]:self.sparsity_left.indptr[j+1]]))
        return self.x[self.Pt[j]]

    def solve_all(self, b: np.ndarray) -> np.ndarray:
        self.x = b[self.P]
        partial_spsolve(self.L.data, self.L.indices, self.L.indptr,
                        self.x, self.sparsity_left_all)
        return self.x[self.Pt]
