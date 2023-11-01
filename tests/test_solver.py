from typing import Callable
import numpy as np
from numpy.typing import NDArray
from genet.solver import RegRelSolver


def setup() -> tuple[NDArray, NDArray, float]:
    n = 5
    inv_kt = np.identity(n)
    inv_k0 = np.identity(n)

    return inv_k0, inv_kt


def ad_gradient(objective: Callable, x: NDArray, eps: float = 1e-16) -> NDArray:
    z = x + 0j
    g = np.zeros_like(x)

    for i in range(x.size):
        z[i] += eps * 1j
        g[i] = objective(z).imag / eps
        z[i] -= eps * 1j

    return g


def test_gradient():
    inv_k0, inv_kt = setup()
    solver = RegRelSolver(inv_k0, inv_kt)

    x = np.arange(solver.m, dtype=float)
    my_grad = solver.gradient(x)
    tr_grad = ad_gradient(solver.objective, x)

    assert np.allclose(my_grad, tr_grad)


def test_fit():
    inv_k0, inv_kt = setup()
    solver = RegRelSolver(inv_k0, inv_kt)

    my_solution = solver.fit()
    tr_solution = np.zeros((solver.n, solver.n))

    assert np.allclose(my_solution, tr_solution)
