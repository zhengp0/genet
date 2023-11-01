# genet
Gene regulatory network inference with covariance dynamics

## Install

You can install this package through pip
```bash
pip install git+https://github.com/zhengp0/genet.git
```

## Example
After installed the package you can use the following code as an example.
```python
import numpy as np
from genet.solver import RegRelSolver

# problem setup
# true solution of this problem should be all 0
n = 5
inv_kt = np.identity(n)
inv_k0 = np.identity(n)
lam = 0.0

# solve the problem
solver = RegRelSolver(inv_kt, inv_k0, lam)
regrel = solver.fit()

assert np.allclose(regrel, 0.0)
```