import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from warnings import warn


class RegRelSolver:
    """Gene Regulartory Relation Solver."""

    def __init__(
        self,
        k0: NDArray,
        kt: NDArray,
        lam: float = 0.0,
        weight: NDArray | None = None,
    ) -> None:
        k0 = np.asarray(k0)
        kt = np.asarray(kt)
        lam = float(lam)

        n = k0.shape[0]
        if k0.shape != (n, n):
            raise ValueError("`k0` shape doesn't match")
        if kt.shape != (n, n):
            raise ValueError("`kt` shape doesn't match")
        if lam < 0:
            raise ValueError("`lam` must be positive")

        if weight is None:
            weight = np.ones((n, n))
        weight = np.asarray(weight)

        if weight.shape != (n, n):
            raise ValueError("`weight` shape doesn't match")
        if (weight < 0).any() or (weight > 1).any():
            raise ValueError("`weight` elements have to between 0 or 1")

        self.n = n
        self.k0 = k0
        self.kt = kt
        self.inv_k0 = np.linalg.inv(k0)
        self.inv_kt = np.linalg.inv(kt)
        self.lam = lam
        self.weight = weight

        self.mask = self._get_mask()
        self.result = None
        self.m = self.mask.sum()
        self._identity = np.identity(self.n)

    def _get_mask(self) -> NDArray:
        return (~np.isclose(self.inv_k0, 0.0)) | (~np.isclose(self.inv_kt, 0.0))

    def _slim_to_full(self, slim: NDArray) -> NDArray:
        full = np.zeros((self.n, self.n), dtype=slim.dtype)
        full[self.mask] = slim
        return full

    def _full_to_slim(self, full: NDArray) -> NDArray:
        slim = full[self.mask]
        return slim

    # optimization interfaces
    def objective(self, at_slim: NDArray) -> float:
        at_full = self._slim_to_full(at_slim)
        exp_at = self._identity + at_full
        residual = self.kt - exp_at.dot(self.k0).dot(exp_at.T)

        return (
            0.5 * (self.weight * residual**2).sum()
            + 0.5 * self.lam * (at_slim**2).sum()
        )

    def gradient(self, at_slim: NDArray) -> NDArray:
        at_full = self._slim_to_full(at_slim)
        exp_at = self._identity + at_full
        residual = self.kt - exp_at.dot(self.k0).dot(exp_at.T)

        grad = self._full_to_slim(
            2.0 * self.weight * exp_at.dot(self.k0).dot(-residual)
        )
        grad += self.lam * at_slim

        return grad

    def fit(
        self, at_full_0: NDArray | None = None, options: dict | None = None
    ) -> NDArray:
        if at_full_0 is None:
            at_full_0 = np.zeros((self.n, self.n))
        at_slim_0 = self._full_to_slim(at_full_0)
        self.result = minimize(
            fun=self.objective,
            x0=at_slim_0,
            method="BFGS",
            jac=self.gradient,
            options=options,
        )

        if not self.result.success:
            warn(f"solver failed to converge, with status={self.result.status}")

        at_slim_opt = self.result.x
        at_full_opt = self._slim_to_full(at_slim_opt)
        return at_full_opt
