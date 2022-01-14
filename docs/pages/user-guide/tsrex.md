# Tensor Expressions

[Tensors and indices](../tensor-and-indices) can be combined in *tensor 
expressions* (`TsrEx`) in three ways:

- **Indexless expressions** that only involve tensors,
- **Indexed expressions** that involve tensors indexed with indices,
- **Hybrid expressions** that are a combinition of both.

!!! note
    FunFact adopts a lazy evaluation model for tensor expressions.
    When a tensor expression is generated only [some basic anlysis](#properties-of-tensor-expressions) is performed without evaluating
    the full expression. This only happens when a model created from the
     tensor expression is [evaluated](../eval).

## Thee types of tensor expressions

### Indexless expressions

An indexless expression only involves tensors. These operations are
equivalent to NumPy-style array operations.
The supported operations are:

#### Elementwise operations

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
tsrex = -a                      # unary minus
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

#### Matrix inner product

An indexless matrix inner product can be carried out on 2-way tensors
with compatible dimensions with the `@` operator.

```py
import funfact as ff
a = ff.tensor('a', 5, 2) 
b = ff.tensor('b', 2, 3)
c = a @ b                       # matrix product of a and b with shape 5 x 3
```

Matrix products can be concatenated (`a @ b @ c @ ...`) and combined with 
elementwise operations (`(a @ b) + c`).

#### Matrix Kronecker product

An indexless matrix Kronecker can be performed for 2-way tensors 
with the `&` operator.

```py
import ff as ff
a = ff.tensor('a', 5, 2)
b = ff.tensor('b', 2, 3)
c = a & b                       # Kronecker product of a and b with shape 10 x 6
```

Kronecker products can be concatenated (`a & b & c & ...`) and combined with
matrix multiplications and elementwise operations (`(a & b) @ c + d`).

### Indexed expressions

### Hybrid expressions

## Properties of tensor expressions

## Examples