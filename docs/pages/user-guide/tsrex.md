# Tensor Expressions

[Tensors and indices](../tensor-and-indices) can be combined in *tensor 
expressions* (`TsrEx`) in three ways:

- [**Indexless expressions**](#indexless-expressions) that only involve
tensors,
- [**Indexed expressions**](#indexed-expressions) that involve tensors indexed
with indices,
- [**Hybrid expressions**](#hybrid-expressions) that are a combinition of both.

This page provides a brief overview of the supported operations and illustrates
their use with some examples.

!!! note
    FunFact adopts a lazy evaluation model for tensor expressions. Upon
    generation only some [basic analysis](#properties-of-tensor-expressions)
    is performed without evaluating the full expression. Evaluation only
    happens when a model created from the tensor expression is
    [evaluated](../eval).

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
tsrex = a**b                    # elementwise exponentiation: base a, exponent b
```
Elementwise operations can also be carried out between a tensor and a scalar.
In this case the scalar is broadcasted to the same dimensions as the tensor
and the operations is performed elementwise.

```py
import funfact as ff
a = ff.tensor('a', 5, 2, 4, 6)  # a random 4-way tensor
tsrex = a + 1                   # elementwise sum of a and 1, equiv: 1 + a
tsrex = a - 3                   # elementwise difference of a and 3
tsrex = 3 - a                   # additive inverse of above
tsrex = a * 2                   # elementwise product of a and 2, equiv: 2 * a
tsrex = a / 4                   # elementwise division of a and 4
tsrex = 4 / a                   # multiplicative inverse of above
tsrex = a**3                    # elementwise cube of a
tsrex = 2**a                    # elementwise exponentiation: base 2, exponent a
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

FunFact tensor expressions can also be written as *indexed expressions* with a
syntax that is based on Einstein notation. This is similar to how
[einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)
works in NumPy. Indices that are repeated on the left-hand and right-hand side
of a binary operation are assumed to be contracting indices.
For example, the [indexless matrix inner product](#matrix-inner-product) can be 
written as follows as an indexed expression:

```py
import funfact as ff
a = ff.tensor('a', 5, 2)
b = ff.tensor('b', 2, 3)
i, j, k = ff.indices('i, j, k')
tsrex = a[i, j] * b[j, k]       # equivalent: a @ b
```

Here the index `j` appears on both the left-hand and right-hand side and is thus
contracted. The contraction is performed by an elementwise multiplication 
(as indicated by `*`) followed by a reduction through a summation along the 
contracting dimension. The elementwise operation `*` can be replaced by other 
elementwise operations that are all combined with a reduction through summation:

```py
tsrex = a[i, j] + b[j, k]       # elementwise summation,   reduction by summation
tsrex = a[i, j] / b[j, k]       # elementwise division,    reduction by summation
tsrex = a[i, j] - b[j, k]       # elementwise subtraction, reduction by summation
```

!!! warning
    TODO: describe how to change the reduction operation, 
    refer to semiring operators.

In contrast to an indexless matrix inner product, indexed expressions can be 
extended to higher-order tensors as illustrated in the following example.

!!! example
    A [Tucker decomposition](https://en.wikipedia.org/wiki/Tucker_decomposition)
    for a 3-way tensor,

    \[
    \mathcal{T} = T \times_1 U^{(1)} \times_2 U^{(2)} \times_3 U^{(3)},
    \]

    can be written in FunFact as follows:

    ```py
    T = ff.tensor('T', 4, 5, 6)
    U1 = ff.tensor('U_1', 4, 10)
    U2 = ff.tensor('U_2', 5, 20)
    U3 = ff.tensor('U_3', 6, 30)
    i, j, k, l, m, n = ff.indices('i, j, k, l, m, n')
    tsrex = T[i, j, k] * U1[i, l] * U2[j, m] * U3[k, n] # l x m x n tensor
    ```

    We will see later how we can write a FunFact tensor expression for a 
    [Tensor rank decomposition](https://en.wikipedia.org/wiki/Tensor_rank_decomp
    osition) of a 3-way tensor using explicitly non-reducing indices.

#### Semiring Operators

!!! warning
    TODO: update the semiring implementation. Add brief description here.

#### Explicitly Non-Reducing Indices

Indices can be made explicitly non-reducing by decorating them with a `~`
symbol. This explicitly blocks a contraction along this index even if it is
repeated on both sides of a binary expression. For non-reducing indices that are
repeated on both sides of an expression only the elementwise operations are 
performed without the reduction operation. For example,
```py
import funfact as ff
a = ff.tensor('a', 5, 2)
b = ff.tensor('b', 2, 3)
i, j, k = ff.indices('i, j, k')
tsrex = a[i, ~j] * b[j, k]      # 5 x 2 x 3 tensor
``` 

!!! note
    The `~` index decorator can be added to any of the occurences of an index in
    a binary expression in order to be applied. For example, 
    `a[i, ~j] * b[j, k]`, `a[i, j] * b[~j, k]`, and
    `a[i, ~j] * b[~j, k]` all give the same result.

!!! note
    Decorating indices that are not repeated on both sides of an expression 
    with `~` has no effect.

!!! example
    The `~` index decorator can be used to write a FunFact tensor expression for
    a rank-$r$ [tensor rank decomposition](https://en.wikipedia.org/wiki/Tensor_
    rank_decomposition) of a 3-way tensor,

    \[
    \mathcal{T} = \sum_{i=1}^r a_i \otimes b_i \otimes c_i,
    \]
    
    where $\otimes$ is the tensor product of vectors; not to be confused
    with a matrix Kronecker product. This can be translated into a FunFact
    tensor expression as follows:

    ```py
    r = 5
    a = ff.tensor('a', 10, r)
    b = ff.tensor('b', 20, r)
    c = ff.tensor('c', 30, r)
    i, j, k, l = ff.indices('i, j, k, l')
    tsrex = (a[i, ~l] * b[j, l]) * c[k, l] # i x j x k tensor
    ```
    
    Here the `~` is required to avoid the reduction over the `l` index after
    the first contraction.

After a contraction involving explicitly non-reducing indices, the order of the
indices in the resulting tensor is:

1. *Free surviving* indices of the left-hand side,
2. Explicitly non-reducing indices on both left-hand and right-hand side,
3. *Free surviving* indices of the right-hand side.

The following examples illustrate this behavior. We assume that all tensors
have the appropriate shape for the specified operation.

```py
tsrex = a[i, ~j] * b[j, k]          # i x j x k tensor
tsrex = a[~j, i] * b[j, k]          # i x j x k tensor
tsrex = a[i, j] * b[k, ~j]          # i x j x k tensor
```

#### Kronecker indices

A second kind of index decorator is given by `*`. This symbol indicates that
a given index is a *Kronecker index*, which means that the index is treated
as an *in-dimension outer product*. We can use the `*` index decorator to
write the standard [Kronecker product](#matrix-kronecker-product) of
two matrices as an indexed FunFact tensor expression:

```  py
import funfact as ff
a = ff.tensor('a', 5, 2)
b = ff.tensor('b', 2, 3)
i, j = ff.indices('i, j')
tsrex = a[[*i, *j]] *b[i, j]   # Kronecker product of a and b with shape 10 x 6
```

!!! warning
    Index notations with a Kronecker product index have to written with double 
    brackets `[[*i, *j]]`. This is required because `*` operator is the list 
    unpacking operator in Python.

!!! note
    The `*` index decorator can be added to any of the occurences of an index in
    a binary expression in order to be applied. For example, 
    `a[[*i, *j]] * b[i, j]`, `a[i, j] * b[[*i, j]]`, and
    `a[[*i, *j]] * b[[*i, j]]` all give the same result.

!!! example
    The combination of Kronecker `*` indices and explicitly non-reducing `~`
    indices can be used to write a FunFact tensor expression for a
    [Khatri-Rao product](https://en.wikipedia.org/wiki/Khatri%E2%80%93Rao_produc
    t) or *columnwise Kronecker product* of two matrices
    $A \in \mathbb{R}^{m_1 \times n}$ and $B \in \mathbb{R}^{m_2 \times n}$,

    \[
        C = A \odot B 
         := [a_1 \otimes b_1 \ a_2 \otimes b_2 \ \cdots \ a_n \otimes b_n],
    \]

    here $C \in \mathbb{R}^{m_1 m_2 \times n}$ and $\otimes$ *is* the Kronecker
    product of two vectors. The Khatri-Rao product and its row-wise variant can
    be written in FunFact as:

    ``` py
    import funfact as ff
    a = ff.tensor('a', 5, 2)
    b = ff.tensor('b', 3, 2)
    c = ff.tensor('c', 5, 4)
    i, j = ff.indices('i, j')
    # (standard) Khatri-Rao product of a and b with shape 15 x 2 :
    tsrex = a[[*i, ~j]] * b[i, j]
    # row-wise Khatri-Rao product of a and c with shape 5 x 8 :
    tsrex = a[[~i, *j]] * c[i, j] 
    ```

    Notice that up to reshaping of the data, the Khatri-Rao product is
    equivalent with using a standard outer product instead of an in-dimension
    outer product:

    ```py
    tsrex = a[i, ~j] * b[k, ~j]     # 5 x 2 x 3 tensor
    ```

#### Index transposition

Indexed tensor expressions can be transposed with the `>>` operator.
Consider the following indexed tensor expression:

```py
import funfact as ff
a = ff.tensor('a', 4, 3, 2, 6)
b = ff.tensor('b', 2, 4, 5, 6)
i, j, k, l, m = ff.indices('i, j, k, l, m')
tsrex = a[i, j, k, ~m] * b[k, i, l, m]
```

In `tsrex` the `i` and `k` indices are contracted as they are repeated on both
sides; the `m` index is also repeated but is decorated with `~` to indicate
that is not reduced over. Thus `tsrex` has indices `j, m, l` in that order.
If we want to reorder the indices and transpose `tsrex`, we can use the `>>`
operation:

```py
tsrex = tsrex >> [j, l, m]
```

This can also be written in a single line

```py
tsrex = a[i, j, k, ~m] * b[k, i, l, m] >> [j, l, m]
```

#### Index reassignment

Indices can also be reassigned to new indices:

```py
import funfact as ff
a = ff.tensor('a', 5, 2)
b = ff.tensor('b', 2, 3)
i, j, k = ff.indices('i, j, k')
tsrex = a[i, j] * b[j, k]           # tsrex has indices i, k
m, n = ff.indices('m, n')           # new indices    
tsrex = tsrex[m, n]                 # reassign indices to m, n
tsrex = (a[i, j] * b[j, k])[m,n]    # single-line equivalent
```

This functionality is particularly useful in combination with indexless
expression in order to create hybdrid expressions.

### Hybrid expressions

Hybrid tensor expressions consist of both indexless and indexed sub expressions.
The following is an illustrative example:

```py
import funfact as ff
a = ff.tensor('a', 5, 2)
b = ff.tensor('b', 2, 3)
c = ff.tensor('c', 5, 4, 6)
d = ff.tensor('d', 3, 4, 6)
i, j, k, l = ff.indices('i, j, k, l')
tsrex = ((a @ b)[i, j] * c[i, k, l]) + d
```

The following operations happen in `tsrex`:

1. Matrix product of `a` and `b`, resulting in an indexless $5 \times 3$ tensor
2. Renaming the indices of this tensor to `[i, j]` 
3. An indexed contraction with `c`, resulting in a $3 \times 4 \times 6$ tensor
   with indices `[j, k, l]`
4. An indexless elementwise summation with `d`

The result `tsrex` is thus an indexless $3 \times 4 \times 6$ tensor.

## Non-linearities

FunFact tensor expressions can be enhanced by using non-linearities on
sub-expressions. Non-linear operations can be found in the `math` module, 
which is modeled after NumPy. All non-linearities are elementwise operations.

## Properties of tensor expressions

Every FunFact tensor expressions comes with out-of-the-box functionality
to help understanding its structure. We illustrate this functionality
for the following example:

```py
import funfact as ff
a = ff.tensor('a', 5, 2, 4)
b = ff.tensor('b', 2, 3)
i, j, k, l = ff.indices('i, j, k, l')
tsrex = a[i, j, l] * b[j, k]
```

- **LaTeX style** rendering of tensor expressions is available in Jupyter
notebooks. Running
```py
tsrex
```
renders as:

\[
    a_{ijl} \times b_{jk}
\]

- **asciitree** representation of a tensor expression:
```py
tsrex.asciitree
```
returns:
```
 binary: multiply 
 ├── index_notation: [i,j,l] 
 │   ├── tensor: a 
 │   ╰── indices: i,j,l 
 │       ├── index: i 
 │       ├── index: j 
 │       ╰── index: l 
 ╰── index_notation: [j,k] 
     ├── tensor: b 
     ╰── indices: j,k 
         ├── index: j 
         ╰── index: k 
```
- **shape** of a tensor expression:
```py
tsrex.shape        # returns (5, 4, 3)
```

- **dimensionality** of a tensor expression:
```py
tsrex.ndim        # returns 3
```

- **Einsum**-like specification of the top level of the tensor expression:
```py
tsrex.einspec        # returns 'abc,bd->acd|'
```

- **Surviving indices** of the tensor expression:
```py
tsrex.live_indices 
# returns [AbstractIndex(i), AbstractIndex(l), AbstractIndex(k)]
```

