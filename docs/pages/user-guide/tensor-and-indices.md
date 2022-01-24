# Tensors and Indices

## Abstract Indices


## Abstract Tensors

Tensors can be created using the [tensor](funfact.tensor) method.

``` py
import funfact as ff
a = ff.tensor('a', 5, 2, 4)                      # a random 3-way tensor
```
!!! note
    The tensors that we create here are *abstract tensors*, which only have attributes such as dimensionality, shape, and a symbol. They have not been populated with numerical elements yet.





```
import numpy as np
b = ff.tensor('b', np.arange(10).reshape(2, 5))  # a 2-way tensor (matrix)
u = ff.tensor('u', 4)                            # a random vector
v = ff.tensor(5)                                 # an anonymous random vector
c = ff.tensor('c')                               # a scalar, i.e. 0-tensor
i, j, k = ff.indices('i, j, k')                  # define indices
```

