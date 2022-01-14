# Tensor Expressions

[Tensors and indices](../tensor-and-indices) can be combined in *tensor 
expressions* (`TsrEx`) in three ways:

- [**Indexless expressions**](#indexless-expressions) that only involve tensors,
- [**Indexed expressions**](#indexed-expressions) that involve tensors indexed with indices,
- [**Hybrid expressions**](#hybrid-expressions) that are a combinition of both.

!!! note
    FunFact adopts a lazy evaluation model for tensor expressions.
    When a tensor expression is generated only [some basic analysis](#properties-of-tensor-expressions)
    is performed without evaluating the full expression. This only happens
    when a model created from the tensor expression is [evaluated](../eval).

## Thee Types of Tensor Expressions

### Indexless Expressions

An indexless expression only involves tensors. These operations are
equivalent to NumPy-style array operations.
The supported operations are:

#### Elementwise Operations

Elementwise operations act on tensors of the same shape.

```py
import funfact as ff
a = ff.tensor('a', 5, 2, 4, 6)  # a random 4-way tensor
b = ff.tensor('b', 5, 2, 4, 6)  # a random 4-way tensor with the same shape
tsrex = a + b                   # elementwise sum of a and b
tsrex = a - b                   # elementwise difference of a and b
tsrex = a * b                   # elementwise product of a and b
tsrex = a / b                   # elementwise division of a and b
tsrex = a**b                    # elementwise exponentiation: base a, exponents b
```
Elementwise operations can also be carried out between a tensor and a scalar value.
In this case the scalar is broadcasted to the same dimensions as the tensor
and the operations is performed elementwise.

```py
import funfact as ff
a = ff.tensor('a', 5, 2, 4, 6)  # a random 4-way tensor
tsrex = a + 1                   # elementwise sum of a and 1, equivalent: 1 + a
tsrex = a - 3                   # elementwise difference of a and 3
tsrex = 3 - a                   # additive inverse of above
tsrex = a * 2                   # elementwise product of a and 2, equivalent: 2 * a
tsrex = a / 4                   # elementwise division of a and 4
tsrex = 4 / a                   # multiplicative inverse of above
tsrex = a**3                    # elementwise cube of a
tsrex = 2**a                    # elementwise exponentiation: base 2, exponents a
tsrex = -a                      # elementwise unary minus
```

Elementwise operations can be concatenated and the usual order of operations is
taken into acount.

```py
import funfact as ff
import numpy as np
a = ff.tensor('a', np.array([1.]))
b = ff.tensor('b', np.array([2.]))
c = ff.tensor('c', np.array([3.]))
tsrex = a + b * c               # returns 7 after evaluation
tsrex = a + (b * c)             # equivalent to above
tsrex = (a + b) * c             # returns 9 after evaluation
```

#### Matrix Product

An indexless matrix inner product can be carried out on 2-way tensors
with compatible dimensions with the `@` operator.

```py
import funfact as ff
a = ff.tensor('a', 5, 2) 
b = ff.tensor('b', 2, 3)
tsrex = a @ b                   # matrix product of a and b with shape 5 x 3
```

Matrix products can be concatenated (`a @ b @ c @ ...`) and combined with 
elementwise operations (`(a @ b) + c`).

#### Matrix Kronecker Product

An indexless matrix Kronecker can be performed for 2-way tensors 
with the `&` operator.

```py
import funfact as ff
a = ff.tensor('a', 5, 2)
b = ff.tensor('b', 2, 3)
tsrex = a & b                   # Kronecker product of a and b with shape 10 x 6
```

Kronecker products can be concatenated (`a & b & c & ...`) and combined with
matrix multiplications and elementwise operations (`(a & b) @ c + d`).

### Indexed Expressions

FunFact tensor expressions can also be written as indexed expressions that are based on
Einstein notation, similar to [einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)
in NumPy, where repeated indices are as assumed contracting indices.
For example, the [indexless matrix inner product](#matrix-inner-product) can be written as follows
as an indexed expression:

```py
import funfact as ff
a = ff.tensor('a', 5, 2)
b = ff.tensor('b', 2, 3)
i, j, k = ff.indices('i, j, k')
tsrex = a[i, j] * b[j, k]       # equivalent: a @ b
```

Here the index `j` appears on both the left-hand and right-hand side and thus contracted over. 
The contraction is performed by an elementwise
multiplication (as indicated by `*`) followed by a reduction through summation along the contracting
dimension. The elementwise operation can be replaced by other operations:

```py
tsrex = a[i, j] + b[j, k]       # elementwise summation,   reduction by summation
tsrex = a[i, j] / b[j, k]       # elementwise division,    reduction by summation
tsrex = a[i, j] - b[j, k]       # elementwise subtraction, reduction by summation
```

!!! warning
    TODO: describe how to change the reduction operation, refer to semiring operators.

In contrast to the indexless matrix inner product, indexed expressions can be easily extended 
to higher-order tensors. For example, a [Tucker decomposition](https://en.wikipedia.org/wiki/Tucker_decomposition)
for a 3-way $10 \times 20 \times 30$ tensor can be written in FunFact as follows:

```py
T = ff.tensor('T', 4, 5, 6)
U1 = ff.tensor('U_1', 4, 10)
U2 = ff.tensor('U_2', 5, 20)
U3 = ff.tensor('U_3', 6, 30)
i, j, k, l, m, n = ff.indices('i, j, k, l, m, n')
tsrex = T[i, j, k] * U1[i, l] * U2[j, m] * U3[k, n] # l x m x n tensor
```

We will see later how we can write a [Tensor rank decomposition](https://en.wikipedia.org/wiki/Tensor_rank_decomposition) for a 3-way tensor using explicitly non-reducing indices.

#### Semiring Operators

!!! warning
    TODO: update the semiring implementation.

#### Explicitly Non-Reducing Indices

Indices can be made explicitly non-reducing by decorating them with a `~` which avoids a contraction
even when they are repeated. In that case only the elementwise operations will be carried out without
the reduction.
For example,
```py
import funfact as ff
a = ff.tensor('a', 5, 2)
b = ff.tensor('b', 2, 3)
i, j, k = ff.indices('i, j, k')
tsrex = a[i, ~j] * b[j, k]      # 5 x 2 x 3 tensor
``` 

!!! note
    The `~` decorator can be added to any of the occurences in a binary expression in order
    to be applied. For example, `a[i, ~j] * b[j, k]`, `a[i, j] * b[~j, k]`, and
    `a[i, ~j] * b[~j, k]` all give the same result.

We can use this functionality to write a rank-5 [tensor rank decomposition](https://en.wikipedia.org/wiki/Tensor_rank_decomposition) of a $10 \times 20 \times 30$ tensor as follows:

```py
r = 5
U1 = ff.tensor('U_1', 10, r)
U2 = ff.tensor('U_2', 20, r)
U3 = ff.tensor('U_3', 30, r)
i, j, k, l = ff.indices('i, j, k, l')
tsrex = (U1[i, ~l] * U2[j, l]) * U3[k, l] # i x j x k tensor
```
Here the `~` is required to avoid the reduction over the `l` index after the first contraction.

#### Kronecker indices

The `*` operator is a second kind of index decorator that indicates that a given index is a *Kronecker
index*, which means that the given index is considered as an *in-dimension outer product*. This 
functionality can be used to write the standard [Kronecker product](#matrix-kronecker-product) of
two matrices as an indexed expression:

```  py
import funfact as ff
a = ff.tensor('a', 5, 2)
b = ff.tensor('b', 2, 3)
i, j = ff.indices('i, j')
tsrex = a[[*i, *j]] & b[i, j]   # Kronecker product of a and b with shape 10 x 6
```

!!! warning
    Index notations with a Kronecker product index have to written with double brackets `[[*i, *j]]`

!!! note
     The `*` decorator can be added to any of the occurences in a binary expression in order
    to be applied. For example, `a[[*i, *j]] * b[i, j]`, `a[i, j] * b[[*i, j]]`, and
    `a[[*i, *j]] * b[[*i, j]]` all give the same result.

An interesting use-case of Kronecker indices is to write a FunFact tensor expression for a
[Khatri-Rao product](https://en.wikipedia.org/wiki/Khatri%E2%80%93Rao_product) of two matrices:

``` py
import funfact as ff
a = ff.tensor('a', 5, 2)
b = ff.tensor('b', 2, 2)
c = ff.tensor('c', 5, 4)
i, j = ff.indices('i, j')
tsrex = a[[*i, ~j]] & b[i, j]   # Khatri-Rao product of a and b with shape 10 x 2
tsrex = a[[~i, *j]] & c[i, j]   # row-wise Khatri-Rao product of a and c with shape 5 x 8
```

!!! warning
    This doesn't work!?

#### Tensor transposition

#### Index reassignment

### Hybrid expressions

## Non-linearities

## Properties of tensor expressions

## Examples