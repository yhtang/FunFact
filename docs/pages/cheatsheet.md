# Cheatsheet

## Import

``` py title="Recommended way of importing funfact"
import funfact as ff
from funfact import active_backend as ab  # use it like numpy!
```

## Choose [backend](../user-guide/backends/)

| **Syntax** | **Description** |
| ------------------- | -------------------------------------------------------------------- |
| `ff.use('jax')` | Use JAX as backend |
| `ff.use('torch')` | Use PyTorch as backend |
| `ff.use('numpy')` | Use NumPy as backend, forward calculations only |


## Declare [abstract indices](../user-guide/tensor-and-indices#abstract-indices)

| **Syntax** | **Description** |
| ------------------- | -------------------------------------------------------------------- |
| `ff.index()` | Create an anonymous index |
| `ff.index('i')` | Create an index with name $i$ |
| `ff.index('j_1')` | Create an index with name $j_1$ |
| `ff.indices('i j k')` | Creates a tuple of 3 indices named $i$, $j$, and $k$ |
| `ff.indices('i, j, k')` | Same as above |
| `ff.indices(5)` | Create a tuple of 5 anonymous indices |

## Declare [abstract tensors](../user-guide/tensor-and-indices#abstract-tensors)

| **Syntax** | **Description** |
| ------------------- | -------------------------------------------------------------------- |
| `ff.tensor('T', 2, 3, 4)` | A $2 \times 3 \times 4$ random tensor $T$ with normally initialized values |
| `ff.tensor('T', np.array([[1, 2], [3, 4]]))` | A concrete matrix $T = \begin{bmatrix}1&2\\3&4\end{bmatrix}$ |
| `ff.tensor(2, 3, 4)` | A $2 \times 3 \times 4$ nameless random tensor with normally initialized values |
| `ff.tensor(np.array([[1, 2], [3, 4]]))` | A concrete nameless matrix $\begin{bmatrix}1&2\\3&4\end{bmatrix}$ |
| <div style="color:#808080;">**Optimizability**</div> || 
| `ff.tensor('T', 2, 3, 4)` | Pure abstract tensors will be optimized by default |
| `ff.tensor('T', 2, 3, 4, optimizable=False)` | No optimization after random initialization |
| `ff.tensor('T', np.random.randn(2, 3, 4))` | Concrete data disables optimization |
| `ff.tensor('T', np.random.randn(2, 3, 4), optimizable=True)` | Force optimizable |
| <div style="color:#808080;">**Initializers**</div> || 
| `ff.tensor('T', 2, 3, 4, initializer=ff.initializers.Zeros)` | Initialize all elements as 0 |
| `ff.tensor('T', 2, 3, 4, initializer=ff.initializers.Ones)` | Initialize all elements as 1 |
| `ff.tensor('T', 4, 5, initializer=ff.initializers.Eye)` | Initialize as the identity matrix (2-way tensors only) |
| `ff.tensor('T', 2, 3, 4, initializer=ff.initializers.Normal)` | Initialize all elements as i.i.d. samples from the [normal distribution](../../api/initializers/#funfact.initializers.Normal) |
| `ff.tensor('T', 2, 3, 4, initializer=ff.initializers.Uniform)` | Initialize all elements as i.i.d. samples from the [uniform distribution](../../api/initializers/#funfact.initializers.Uniform) |
| <div style="color:#808080;">**Conditions**</div> |
| `ff.tensor('T', 2, 3, 4, prefer=ff.conditions.Unitary)` | Penalize the tensor if not orthonormal/unitary |
| `ff.tensor('T', 2, 3, 4, prefer=ff.conditions.UpperTriangular)` | Penalize lower triangular elements from being nonzero |
| `ff.tensor('T', 2, 3, 4, prefer=ff.conditions.NonNegative)` | Penalize tensor elements for being negative |
| <div style="color:#808080;">**Special Tensors**</div> |
| `ff.zeros(2, 3, 4)` | Shortcut for `ff.tensor(2, 3, 4, initializer=ff.initializers.Zeros)` |
| `ff.ones(2, 3, 4)` | Shortcut for `ff.tensor(2, 3, 4, initializer=ff.initializers.Ones)` |
| `ff.eye(2, 3)` | Shortcut for `ff.tensor(2, 3, initializer=ff.initializers.Eye)` |

## Arithmetics

| **Syntax** | **Description** |
| ------------------- | -------------------------------------------------------------------- |
| `a[i] + b[j]` | $\boldsymbol{\lambda}_{ij} = \boldsymbol{a}_{i}  -\boldsymbol{b}_{j}$, pairwise addition between two vectors |
| `a[i] - b[j]` | $\boldsymbol{\lambda}_{ij} = \boldsymbol{a}_{i}  -\boldsymbol{b}_{j}$, pairwise subtraction between two vectors |
| `a[i] * b[j]` | $\boldsymbol{a} \boldsymbol{b}^\mathsf{T}$, pairwise multiplication, i.e. outer product between two vectors |
| `a[i] / b[j]` | $\boldsymbol{\lambda}_{ij} = \boldsymbol{a}_{i}  -\boldsymbol{b}_{j}$, pairwise division between two vectors |
| `a[i] * b[i]` | $\boldsymbol{a} \cdot \boldsymbol{b}$, inner product between two vectors |
| `A[i, j] * B[j, k]` | $A \cdot B$, inner product between two matrices |
| `A[i, j] * B[k, j]` | $A \cdot B^\mathsf{T}$, inner product with transposition |
| `a[i, j, k] * b[r, k, s]` | Contraction between the 3rd dimension of $a$ and the 2nd dimension of $b$ |
| <div style="color:#808080;">**Contraction suppression**</div> |
| `A[i, ~j] * B[~j, k]` | $\boldsymbol{\lambda}_{ijk} = \boldsymbol{A}_{ij} \boldsymbol{B}_{jk}$, no contraction along $j$ |
| `A[i, ~j] * B[ j, k]` | same as above |
| `A[i,  j] * B[ ~j, k]` | same as above |
| <div style="color:#808080;">**Output order**</div> |
| `A[i, j, k] >> [k, i, j]` | Transposition/axis permutation |
| `A[i, j] * B[j, k] >> [k, i]` | $(A B)^\mathsf{T}$, reorder output axes |
| <div style="color:#808080;">**Kronecker product**</div> |
| `a[[*i]] * b[[*i]]` | $\boldsymbol{\lambda}_{i \times n'+i'} = \boldsymbol{a}_{i} \boldsymbol{b}_{i'}$, Kronecker product between vectors
| `a[[i, *j]] * b[[*j, k]]` | $\boldsymbol{\lambda}_{i \times n'+i'} = \boldsymbol{a}_{i} \boldsymbol{b}_{i'}$, Kronecker product within the $j$ axis
| <div style="color:#808080;">**Indexless**</div> |
| `a+b`, `a-b`, `a*b`, `a/b`, `a**b` | elementwise addition, subtraction, multiplication, division, exponentiation |
| `a @ b` | matrix multiplication |
| `a & b` | Kronecker product |

## Math functions

| **Syntax** | **Description** |
| ------------------- | -------------------------------------------------------------------- |
| `ff.sin(a)` | $\sin \boldsymbol{a}$ |
| `ff.sin(a[i, j])` | Equivalent to `ff.sin(a)[i, j]` |

## [Factorization](../user-guide/eval/) objects

``` py title="context"
T = ff.tensor('T', n1, n2, n3)
U_1 = ff.tensor('U_1', n1, m1, optimizable=False)
U_2 = ff.tensor('U_2', n2, m2)
U_3 = ff.tensor('U_3', n3 ,m3)
i, j, k, l, m, n = ff.indices('i, j, k, p, q, r')
tsrex = T[i, j, k] * U_1[i, p] * U_2[j, q] * U_3[k, r]
target = ab.normal(0.0, 1.0, (m1, m2, m3))
```

| **Syntax** | **Description** |
| ------------------- | -------------------------------------------------------------------- |
| `fac = ff.Factorization.from_tsrex(tsrex)` | Initialize a factorization model without optimization |
| `fac = ff.factorize(tsrex, target, **kwargs)` | Factorize `target` using `tsrex` |
| `fac()` | Obtain the approximate tensor from a factorization model |
| `fac[2, 4, 5]` | Access a single element at `[2, 4, 5]` from the approximation |
| `fac[2, :]` | Access the slice at `[2, :]` from the approximation |
| `fac['U_1']` | Read-only access of factor $U_1$ |
| `fac.factors` | A list of all optimizable factors, i.e. $T$, $U_2$, and $U_3$ |
| `fac.all_factors` | A list of all factors, i.e. $T$, $U_1$, $U_2$, and $U_3$ |
