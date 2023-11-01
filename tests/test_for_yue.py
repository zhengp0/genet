import numpy as np
from genet.solver import RegRelSolver


def test_on_simple_case():
    n = 5
    k0 = np.identity(n)

    at = np.zeros((n, n))
    at[0, 1] = 0.1
    exp_at = np.identity(n) + at

    kt = exp_at.dot(k0).dot(exp_at.T)

    inv_k0 = np.linalg.inv(k0)
    inv_kt = np.linalg.inv(kt)

    solver = RegRelSolver(inv_k0, inv_kt)
    my_at = solver.fit()
    my_exp_at = np.identity(n) + my_at

    # forward
    fresidual = kt - my_exp_at.dot(k0).dot(my_exp_at.T)
    # backward
    bresidual = my_exp_at.T.dot(inv_kt).dot(my_exp_at) - inv_k0

    assert np.allclose(fresidual, 0.0, atol=1e-7)
    assert np.allclose(bresidual, 0.0, atol=1e-7)

    # however we cannot gaurantee exp_at is close to my_exp_at
    # assert np.allclose(my_exp_at, exp_at) <- this will fail the test
