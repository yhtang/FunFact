# Tensors and Indices

## Abstract Indices

Abstract indices are symbolic objects that FunFact uses to denote how
computations should happen between tensors. They can be created using the
[`index`](../../../api/index_) and [`indices`](../../../api/indices)
methods.

``` py title="Examples"
import funfact as ff
i = ff.index('i')          # named index
j = ff.index()             # anonymous index
p, q = ff.indices('p, q')  # batch declaration
r, s, t = ff.indices(3)    # batch declaration of anonymous indices
```

## Abstract Tensors

Tensors can be created using the [`tensor`](../../../api/tensor) method.
These tensors are *abstract* in the sense that they only contain attributes
such as dimensionality, shape, and optionally a symbol.
Abstract tensor are not populated with numerical elements until we start to
solve a concrete factorization problem.

``` py title="Examples"
import funfact as ff
import numpy as np
a = ff.tensor('a', 5, 2, 4)                      # a random 3-way tensor
b = ff.tensor('b', np.arange(10).reshape(2, 5))  # a 2-way tensor (matrix)
u = ff.tensor('u', 4)                            # a random vector
v = ff.tensor(5)                                 # an anonymous random vector
c = ff.tensor('c')                               # a scalar, i.e. 0-tensor
```

## Special tensors

FunFact is shipped with a small set of predefined special tensors due to their
prevalence in linear algebra computations. These include:

- The [unit tensor](../../../api/ones) of all 1s.
- The [zero tensor](../../../api/zeros) of all 0s.
- The [identity/Kronecker delta tensor](../../../api/eye). 

## Initializers

By default, the abstract tensors are populated with random elements draw
from the normal distribution upon initialization of a factorization model. A
collection of alternative initializers are provided in the
[`initializers`](../../../api/initializers) module for customizing this behavior.
Users can also plug in any callable objects that accepts a `shape`
argument of a tuple of integers and returns a numerical tensor of the
corresponding shape.

``` py title="Examples"
import ff.initializers as ini
X = ff.tensor('X', 5, 2, 4, initializer=ini.Uniform)
Y = ff.tensor('Y', 5, 2, 4, initializer=ini.Uniform(scale=1.25))
Z = ff.tensor('Z', 5, 2, 4, initializer=ini.Normal(mean=-1, std=0.2))
```

## Preferences and conditions

Many factorization algorithms enforces constraints on the factors found in the solution.
For example, the eigencomposition requires that the matrix of eigenvectors to be
orthogonal/unitary. In FunFact, this can be specified on a tensor-by-tensor basis
using the `prefer=` argument to [`tensor`](../../../api/tensor). A number of
predefined conditions can be found in [`conditions`](../../../api/conditions).
Custom conditions can be created by subclassing `funfact.conditions._Condition`.

``` py title="Examples"
import ff.conditions as cond
X = ff.tensor('X', 6, 6, prefer=cond.UpperTriangular)
Y = ff.tensor('Y', 6, 6, prefer=cond.Unitary(weight=2.0))
Z = ff.tensor('Z', 6, 6, prefer=cond.Diagonal(elementwise='l1'))
```
