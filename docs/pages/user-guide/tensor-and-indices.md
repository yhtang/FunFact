# Tensors and Indices

!!! note

    This page assumes the readers are familiar with the Einstein notation. Many
    excellent introductary articles can be found at:

    - [Einstein notation - Wikipedia](https://en.wikipedia.org/wiki/Einstein_notation)
    - [Tensor - Wikipedia](https://en.wikipedia.org/wiki/Tensor#Notation)
    - [Einstein Summation - MathWorld](https://mathworld.wolfram.com/EinsteinSummation.html)
    - [Einstein Summation Convention: an Introduction - YouTube](https://www.youtube.com/watch?v=CLrTj7D2fLM)

## Abstract Indices

Abstract indices are symbolic objects that FunFact uses to denote how
computations should happen between tensors. They can be created using the
[`index`](../../../api/index_) and [`indices`](../../../api/indices)
methods.

``` py
import funfact as ff
i = ff.index('i')          # named index
j = ff.index()             # anonymous index
p, q = ff.indices('p, q')  # batch declaration
r, s, t = ff.indices(3)    # batch declaration of anonymous indices
```

## Abstract Tensors

Tensors can be created using the [`tensor`](../../../api/tensor) method.

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