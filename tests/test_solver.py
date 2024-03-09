from typing import Callable
import numpy as np
from numpy.typing import NDArray
from genet.solver import RegRelSolver


def setup() -> tuple[NDArray, dict]:
    np.random.seed(0)
    n = 5

    at = np.random.randn(n, n)
    exp_at = np.identity(n) + at

    k0_half = np.random.randn(n, n)
    k0 = k0_half.T.dot(k0_half)

    kt = exp_at.T.dot(k0).dot(exp_at)

    lam = 0.0
    weight = np.random.rand(n, n)

    return at, dict(k0=k0, kt=kt, lam=lam, weight=weight)


def ad_gradient(objective: Callable, x: NDArray, eps: float = 1e-16) -> NDArray:
    z = x + 0j
    g = np.zeros_like(x)

    for i in range(x.size):
        z[i] += eps * 1j
        g[i] = objective(z).imag / eps
        z[i] -= eps * 1j

    return g


def test_gradient():
    _, solver_args = setup()
    solver = RegRelSolver(**solver_args)

    x = np.arange(solver.m, dtype=float)
    my_grad = solver.gradient(x)
    tr_grad = ad_gradient(solver.objective, x)

    assert np.allclose(my_grad, tr_grad)


def test_fit():
    tr_solution, solver_args = setup()
    solver = RegRelSolver(**solver_args)

    my_solution = solver.fit(at_full_0=tr_solution)

    assert np.allclose(my_solution, tr_solution)
