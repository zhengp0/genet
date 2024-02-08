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
        eta: float = 0.0,
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
        if eta < 0:
            raise ValueError("`eta` must be positive")

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
        self.eta = eta
        self.weight = weight

        self.mask = self._get_mask()
        self.mask_index = np.argwhere(self.mask)
        self.result = None
        self.m = self.mask.sum()
        self._identity = np.identity(self.n)

        self.mask_index_to_selection_weight_index, self.selection_weight = (
            self._get_selection_weight()
        )

    def _get_mask(self) -> NDArray:
        return (~np.isclose(self.inv_k0, 0.0)) | (~np.isclose(self.inv_kt, 0.0))

    def _get_selection_weight(self) -> NDArray:
        mask_index_to_selection_weight_index, p = {}, 0
        for i, j in self.mask_index:
            if i < j:
                mask_index_to_selection_weight_index[(i, j)] = p
                p += 1

        size = len(mask_index_to_selection_weight_index)
        return mask_index_to_selection_weight_index, np.repeat(0.5, size)

    def _slim_to_full(self, slim: NDArray) -> NDArray:
        full = np.zeros((self.n, self.n), dtype=slim.dtype)
        full[self.mask] = slim
        return full

    def _full_to_slim(self, full: NDArray) -> NDArray:
        slim = full[self.mask]
        return slim

    def _selection_weight_to_slim(self, selection_weight: NDArray) -> NDArray:
        slim = np.zeros(self.m)
        for k, (i, j) in enumerate(self.mask_index):
            if i < j:
                slim[k] = selection_weight[
                    self.mask_index_to_selection_weight_index[(i, j)]
                ]
            elif i > j:
                slim[k] = (
                    1.0
                    - selection_weight[
                        self.mask_index_to_selection_weight_index[(j, i)]
                    ]
                )
        return slim

    # optimization interfaces
    def objective_smooth(self, at_slim: NDArray) -> float:
        at_full = self._slim_to_full(at_slim)
        exp_at = self._identity + at_full
        residual = self.kt - exp_at.T.dot(self.k0).dot(exp_at)

        return (
            0.5 * (self.weight * residual**2).sum()
            + 0.5 * self.lam * (at_slim**2).sum()
        )

    def gradient_smooth(self, at_slim: NDArray) -> NDArray:
        at_full = self._slim_to_full(at_slim)
        exp_at = self._identity + at_full
        residual = self.kt - exp_at.T.dot(self.k0).dot(exp_at)

        grad = self._full_to_slim(
            2.0 * self.weight * exp_at.T.dot(self.k0).dot(-residual)
        )
        grad += self.lam * at_slim

        return grad

    def objective(self, at_slim: NDArray) -> float:
        objective_smooth = self.objective_smooth(at_slim)
        selection_penalty = (
            self._selection_weight_to_slim(self.selection_weight).dot(np.abs(at_slim))
        ).sum()
        return objective_smooth + self.eta * selection_penalty

    def gradient_selection_weight(self, at_slim: NDArray) -> NDArray:
        grad = np.zeros_like(self.selection_weight)
        for k, (i, j) in enumerate(self.mask_index):
            if i < j:
                grad[self.mask_index_to_selection_weight_index[(i, j)]] += np.abs(
                    at_slim[k]
                )
            elif i > j:
                grad[self.mask_index_to_selection_weight_index[(j, i)]] -= np.abs(
                    at_slim[k]
                )
        return grad

    def prox_at_slim(self, at_slim: NDArray, step: float) -> NDArray:
        threshold = (
            step * self.eta * self._selection_weight_to_slim(self.selection_weight)
        )
        return np.sign(at_slim) * np.maximum(np.abs(at_slim) - threshold, 0.0)

    def prox_selection_weight(self, at_slim: NDArray, step: float) -> NDArray:
        return np.clip(
            self.selection_weight - step * self.gradient_selection_weight(at_slim),
            0.0,
            1.0,
        )

    def fit(
        self, at_full_0: NDArray | None = None, options: dict | None = None
    ) -> NDArray:
        if at_full_0 is None:
            at_full_0 = np.zeros((self.n, self.n))
        at_slim_0 = self._full_to_slim(at_full_0)

        if self.eta == 0.0:
            self.result = minimize(
                fun=self.objective_smooth,
                x0=at_slim_0,
                method="BFGS",
                jac=self.gradient_smooth,
                options=options,
            )
        else:
            raise NotImplementedError("Directional prior is not implemented yet.")

        if not self.result.success:
            warn(f"solver failed to converge, with status={self.result.status}")

        at_slim_opt = self.result.x
        at_full_opt = self._slim_to_full(at_slim_opt)
        return at_full_opt
