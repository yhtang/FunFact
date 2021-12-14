First, we will define some tensors and indices:

``` py
import funfact as ff
import numpy as np
a = ff.tensor('a', 5, 2, 4)                      # a random 3-way tensor
b = ff.tensor('b', np.arange(10).reshape(2, 5))  # a 2-way tensor (matrix)
u = ff.tensor('u', 4)                            # a random vector
v = ff.tensor(5)                                 # an anonymous random vector
c = ff.tensor('c')                               # a scalar, i.e. 0-tensor
i, j, k = ff.indices('i, j, k')                  # define indices
```

Next, we create some tensor expressions. This only specifies the algebra but
does **not** carry out the computation immediately.

``` py
a[i, j, k] * b[j, i]  # (1) contraction
u[i] * v[j]           # (2) outer product
u[i] - v[j]           # (3) pairwise substraction
```

1. Equivalent to `numpy.einsum('ijk,ji', a, b)`
2. Equivalent to `numpy.outer(u, v)`
3. Equivalent to `np.subtract.outer(u, v)`

Repeated indices imply summation along the given axis. However, summation can be disabled by prepending an index with `~`.

``` py
b[i, j] * v[j]    # (1) inner product between b and v
b[i, ~j] * v[~j]  # (2) multiply v with each row of b
b[i,  j] * v[~j]  # only one `~` needed to suppress summation
b[i, ~j] * v[ j]  # same as above
```

1. Equivalent to `b @ v`, or `np.dot(b, v)`, or `numpy.einsum('ij,j', b, v)`.
2. Equivalent to `b * v[None, :]`, or `numpy.einsum('ij,j->ij', b, v)`.

!!! note 

    decorators only have a one-time effect that takes place upon the first pairing of the same indices.

``` py
b[k, ~j] * (u[i] * v[~j]) 
```