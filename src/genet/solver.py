import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from warnings import warn


class RegRelSolver:
    """Gene Regulartory Relation Solver."""

    def __init__(self, inv_k0: NDArray, inv_kt: NDArray, lam: float) -> None:
        n = inv_k0.shape[0]
        if inv_k0.shape != (n, n):
            raise ValueError("`inv_k0` shape doesn't match")
        if inv_kt.shape != (n, n):
            raise ValueError("`inv_kt` shape doesn't match")
        if lam < 0:
            raise ValueError("`lam` must be positive")

        self.n = n
        self.inv_k0 = inv_k0
        self.inv_kt = inv_kt
        self.lam = lam

        self.mask = self._get_mask()
        self._identity = np.identity(self.n)
        self.result = None

    def _get_mask(self) -> NDArray:
        return self.inv_k0 != 0.0 | self.inv_kt != 0.0

    def _a_slim_to_a_full(self, a_slim: NDArray) -> NDArray:
        a_full = np.zeros((self.n, self.n))
        a_full[self.mask] = a_slim
        return a_full

    def _a_full_to_a_slim(self, a_full: NDArray) -> NDArray:
        a_slim = a_full[self.mask]
        return a_slim

    # optimization interfaces
    def objective(self, a_slim: NDArray) -> float:
        a_full = self._a_slim_to_a_full(a_slim)
        exp_at = self._identity + a_full
        residual = exp_at.T.dot(self.inv_kt).dot(exp_at) - self.inv_k0

        return 0.5 * (residual**2).sum() + 0.5 * (a_slim**2).sum()

    def gradient(self, a_slim: NDArray) -> NDArray:
        a_full = self._a_slim_to_a_full(a_slim)
        exp_at = self._identity + a_full
        residual = exp_at.T.dot(self.inv_kt).dot(exp_at) - self.inv_k0

        grad = self._a_full_to_a_slim(2.0 * exp_at.T.dot(self.inv_kt).dot(residual))
        grad += self.lam * a_slim

        return grad

    def fit(
        self, a_slim_0: NDArray | None = None, options: dict | None = None
    ) -> NDArray:
        if a_slim_0 is None:
            a_slim_0 = np.zeros(self.mask.sum())
        self.result = minimize(
            fun=self.objective,
            x0=a_slim_0,
            method="BFGS",
            jac=self.gradient,
            options=options,
        )

        if not self.result.sccuess:
            warn("solver failed to converge")

        a_slim_opt = self.result.x
        a_full_opt = self._a_slim_to_a_full(a_slim_opt)
        return a_full_opt
