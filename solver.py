import numpy as np
from numpy import ndarray, eye as I, diag
from typing import Union, Callable
import networkx as nx
from scipy.sparse import spmatrix
from cgm import conjugate_gradient
from utils import vec, mat
from numpy.linalg import eigh, solve
from scipy.optimize import minimize

from cvxpy import Variable, Minimize, Problem
from cvxpy import norm as cvxnorm
from cvxpy import SCS
from cvxpy import vec as cvxvec

class CGMFSolver:

    def __init__(self,
                 UT: ndarray,
                 UN: ndarray,
                 G: ndarray,
                 S: ndarray,
                 gamma: float):
        """
        This class is designed to solve the problem

        (diag(vec(S)) + γ H^{-2}) vec(F) = vec(Y)

        for F exactly, where H = (UT ⊗ UN) diag(vec(G)) (UT.T ⊗ UN.T)
        """

        self.UT = UT
        self.UN = UN
        self.G = G
        self.S = S
        self.gamma = gamma

    def set_gamma(self, gamma: float):
        self.gamma = gamma

    def A(self, x: ndarray) -> ndarray:
        """
        Efficicent multiplication of system matrix by vector x. Includes effect of preconditioners
        """
        return self.gamma * x + vec(self.G * (self.UN.T @ (self.S * (self.UN @ (self.G * mat(x, like=self.G)) @ self.UT.T)) @ self.UT))

    def Phi(self, x: ndarray) -> ndarray:
        """
        Efficicent multiplication of right preconditioner by vector x
        """
        return vec(self.UN @ (self.G * mat(x, like=self.G)) @ self.UT.T)

    def PhiT(self, x: ndarray) -> ndarray:
        """
        Efficicent multiplication of left preconditioner by vector x
        """
        return vec(self.G * (self.UN.T @ mat(x, like=self.G) @ self.UT))

    def solve(self, Y: ndarray) -> ndarray:
        """
        Solve the problem (diag(vec(S)) + γ H^{-2}) vec(F) = vec(Y) for a given Y. Returns
        F in matrix form
        """
        return mat(conjugate_gradient(self.A, vec(Y), Phi=self.Phi, PhiT=self.PhiT), like=self.G)


class VarSolver:

    def __init__(self, Omega_Q: ndarray, Q: ndarray, X: ndarray, lam: float = 0.005):

        self.Omega_Q = Omega_Q
        self.X = X
        self.Q = Q.astype(bool)
        self.lam = lam

        self.N, self.T = Q.shape

        self.params = None

    def rmse(self, Omega: ndarray):
        return (((Omega - self.predict()) ** 2).sum() / (self.N * self.T)) ** 0.5

    def r_squared(self, Omega: ndarray):
        return 1 - ((Omega - self.predict()) ** 2).sum() / ((Omega - Omega.mean()) ** 2).sum()

    def _get_params(self):
        pass

    def predict(self):
        pass
    

class NMMVarSolver(VarSolver):
    
    def __init__(self, Omega_Q: ndarray, Q: ndarray, X: ndarray, lam: float = 0.005):
        super().__init__(Omega_Q, Q, X, lam)
        
    def _get_params(self):
        
        q = np.argwhere(self.Q)
        
        self.params = Variable((self.N, self.T))

        obj = Minimize(cvxnorm(self.params, 'nuc'))

        constraints = [self.params[q[:, 0], q[:, 1]] == self.Omega_Q[q[:, 0], q[:, 1]]]

        prob = Problem(obj, constraints)

        prob.solve(solver=SCS, verbose=False)

    def predict(self):
        
        if self.params is None:
            
            self._get_params()
            
        return self.params.value


class RRVarSolver(VarSolver):

    def __init__(self, Omega_Q: ndarray, Q: ndarray, X: ndarray, lam: float = 0.005):
        super().__init__(Omega_Q, Q, X, lam)

    def _get_params(self):
        self.params = [solve(self.X[vec(self.Q)].T @ self.X[vec(self.Q)] + self.lam * I(self.X.shape[1]), self.X.T @ vec(self.Omega_Q))]

    def predict(self):
        if self.params is None:
            self._get_params()
        return mat(self.X @ self.params[0], like=self.Omega_Q)


class KRRVarSolver(VarSolver):

    def __init__(self, Omega_Q: ndarray, Q: ndarray, X: ndarray, lam: float = 0.005):
        super().__init__(Omega_Q, Q, X, lam)




class RNCVarSolver(VarSolver):

    def __init__(self,
                 Omega_Q: ndarray,
                 Q: ndarray,
                 X: ndarray,
                 lam: float,
                 UT: ndarray,
                 UN: ndarray,
                 G: ndarray,
                 gamma: float):

        super().__init__(Omega_Q, Q, X, lam)

        self.UT = UT
        self.UN = UN
        self.G = G
        self.gamma = gamma

        self.N, self.T = self.Omega_Q.shape

        lamX, self.V = eigh(X[vec(Q)].T @ X[vec(Q)])
        self.DX = diag((lamX + lam) ** -0.5)

        self.PP = self.X @ self.V @ self.DX
        self.GPs = [self.G * (self.UN.T @ (self.Q * mat(self.PP[:, i], like=self.G)) @ self.UT) for i in range(self.X.shape[1])]

    def A(self, x: ndarray):
        A = mat(x[:self.N * self.T], like=self.G)
        a = x[self.N * self.T:]
        B1 = self.G * (self.UN.T @ (self.Q * (self.UN @ (self.G * A) @ self.UT.T)) @ self.UT) + self.gamma * A
        B2 = sum(a[i] * self.GPs[i] for i in range(self.X.shape[1]))
        b1 = np.array([(A * self.GPs[i]).sum() for i in range(self.X.shape[1])])

        return np.block([vec(B1 + B2), b1 + a])

    def Phi(self, x: ndarray):
        return np.block([vec(self.UN @ (self.G * mat(x[:self.N * self.T], like=self.G)) @ self.UT.T), self.V @ self.DX @ x[self.N * self.T:]])

    def PhiT(self, x: ndarray):
        return np.block([vec(self.G * (self.UN.T @ mat(x[:self.N * self.T], like=self.G) @ self.UT)), self.DX @ self.V.T @ x[self.N * self.T:]])

    def _get_params(self, verbose=False):

        y0 = np.block([vec(self.Omega_Q), self.X.T @ vec(self.Omega_Q)])
        out = conjugate_gradient(self.A, y0, Phi=self.Phi, PhiT=self.PhiT, verbose=verbose)
        self.params = [mat(out[:self.N * self.T], like=self.G), out[self.N * self.T:]]

    def predict(self, verbose=False):
        if self.params is None:
            self._get_params(verbose=verbose)
        return self.params[0] + mat(self.X @ self.params[1], like=self.G)


class LFPVarSolver(VarSolver):

    def __init__(self,
                 Omega_Q: ndarray,
                 Q: ndarray,
                 X: ndarray,
                 lam: float,
                 UT: ndarray,
                 UN: ndarray,
                 Lam: np.ndarray,
                 S_: ndarray,
                 A_: ndarray,
                 eta: Callable[[ndarray, float], ndarray],
                 beta0: float,
                 ):
        super().__init__(Omega_Q, Q, X, lam)

        self.UT = UT
        self.UN = UN
        self.Lam = Lam

        self.S_ = S_
        self.A_ = A_
        self.eta = eta

        self.x0 = np.array([0, 0, beta0, 0, beta0])

    def H(self, Y: ndarray, beta: float):
        """
        Apply the operation Y -> mat(H @ vec(Y)) efficiently for a filter defined by λ -> η(λ; β)
        """
        return self.UN @ (self.eta(self.Lam, beta) * (self.UN.T @ Y @ self.UT)) @ self.UT.T

    def Omega(self, v: ndarray):
        """
        Return the estimator for Omega for a given objective vector
        """
        return v[0] + v[1] * self.H(self.S_, v[2]) + v[3] * self.H(self.A_, v[4])

    def objective(self, v: ndarray):
        """
        The objective function to minimise
        """
        return ((self.Omega_Q - self.Q * self.Omega(v)) ** 2).sum() + self.lam * ((v - self.x0) ** 2).sum()

    def _get_params(self, verbose=False):

        self.result = minimize(self.objective, x0=self.x0, bounds=[(None, None), (None, None), (-1, None), (None, None), (-1, None)])

        if verbose:
            print(self.result)

        self.params = [self.result.x]

    def predict(self, verbose=False):
        if self.params is None:
            self._get_params(verbose=verbose)
        return self.Omega(self.params[0])

#
# class ANOSolver:
#
#     def __init__(self,
#                  Y: Union[ndarray, spmatrix],
#                  S: Union[ndarray, spmatrix],
#                  LT: Union[ndarray, spmatrix, nx.Graph],
#                  LN: Union[ndarray, spmatrix, nx.Graph],
#                  eta: Callable[[ndarray, float], ndarray],
#                  beta: float,
#                  gamma: float):
#
#         """
#         Create an instance of an ANOSolver. This class is designed to solve the problem
#
#         (diag(vec(S)) + γ H^{-2}) vec(F) = vec(Y)
#
#         for F exactly, and to estimate the log-diagonal of inv(diag(vec(S)) + γ H^{-2}).
#
#         H is the graph filter function, defined on the product graph with time-like and
#         space-like Laplacians LT and LN as
#
#         H = eta(LT ⊕ LN)
#
#         Parameters
#         ----------
#         Y           (N, T) numpy array of observations. Zeros where no observation was made.
#         S           (N, T) binary array. Contains 1s where observations were made and zeros elsewhere.
#         LT          (T, T) time-like graph Laplacian. Can be ndarray, spmatrix or nx.Graph
#         LN          (N, N) space-like graph Laplacian. Can be ndarray, spmatrix or nx.Graph
#         eta         Spectral filter function mapping Laplacian eigenvalues λ -> η(λ; β)
#         gamma       The regularisation parameter
#         """
#
#
#         self.Y = Y
#         self.S = S
#
#         assert Y.ndim == 2 and S.ndim == 2
#         assert Y.shape == S.shape
#
#
#
#         self.N, self.T = Y.shape
#
#         def to_numpy_array(L: Union[ndarray, spmatrix, nx.Graph]) -> ndarray:
#
#             if isinstance(L, nx.Graph):
#                 return nx.linalg.laplacian_matrix(L).toarray()
#             elif isinstance(L, spmatrix):
#                 return L.toarray()
#             elif isinstance(L, ndarray):
#                 return L
#
#         self.LT = to_numpy_array(LT)
#         self.LN = to_numpy_array(LN)
#
#         assert self.LT.shape == (self.T, self.T)
#         assert self.LN.shape == (self.N, self.N)
#
#         self.eta = eta
#         self.beta = beta
#         self.gamma = gamma
#
#         assert gamma > 0
#
#         self.decompose()
#
#         self.f_solver = FSolver(self.UT, self.UN, self.G, self.S, self.gamma)
#
#
#     def decompose(self):
#
#         lamLT, self.UT = eigh(self.LT)
#         lamLN, self.UN = eigh(self.LN)
#         self.Lam = lamLN[:, None] + lamLT[None, :]
#
#         self.G = self.eta(self.Lam).astype(float)
#
#
