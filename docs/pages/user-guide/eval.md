# Factorization model

After creating a FunFact [tensor expression](../tsrex), it can be evaluated by
means of a FunFact *factorization* model. The essential approach is 
straightfoward:

```py
import funfact as ff
# initialize tensors, indices, and create your favorite tensor expression
tsrex = ...
fac = ff.Factorization.from_tsrex(tsrex)
```

This instantiates a factorization model based on the input tensor expression.
The `.from_tsrex` factory method optionally takes two additional keyword 
arguments:

- `dtype`: the numerical type of the model, the available types depend on the
[backend](../backend) that is used.
- `initialize`: boolean that indicates if the leaf tensors are initialized in
Factorization model. By default this is set to `True`.

!!! warning
    Direct initialization of a factorization model (`ff.Factorization(tsrex)`)
    is **NOT** recommended and can lead to undesired behavior.

## Properties of factorization models

A factorization model has properties that can be queried. We illustrate
their usage for the following concrete example of a factorization
model for a $3 \times 2$ rank-1 matrix initialized from concrete data:

```py 
import funfact as ff
import numpy as np
# instantiate data arrays:
a = np.array([1.0, 2.0, 3.0])
b = np.array([-1.0, 4.0])
# instantiate tensors from concrete data:
a = ff.tensor('a', a, optimizable=False)
b = ff.tensor('b', b, optimizable=True, prefer=ff.conditions.NonNegative())
i, j = ff.indices('i, j')
# create tensor expression and factorization model:
tsrex = a[i] * b[j]
fac = ff.Factorization.from_tsrex(tsrex)
```

- The **tensor expression** stored in the factorization model can be retrieved
through the `.tsrex` property. This tensor expression has its own set of 
[properties](../tsrex#properties-of-tensor-expressions).
For our example, running
```py
fac.tsrex
```
renders a LaTeX representation for the tensor expression in a Jupyter notebook:

\[
    \mathbf{a}_i \mathbf{b}_j
\]

!!! warning
    The tensor expression is *compiled* upon instantiation of the factorization
    model. For more complicated expression, the internal `fac.tsrex` can differ
    from the input, but always is functionally equivalent.

- The **shape** of a factorization model can be accessed with the `.shape`
property:
```py
fac.shape       # returns (3, 2)
```

- The **dimensionality** of a factorization model can be accessed through the
`.ndim` property:
```py
fac.ndim        # returns 2
```

!!! note
    Both `.shape` and `.ndim` properties are forwarded from the tensor 
    expression of the factorization model.

- The **optimizable factor** (or *leaf*) tensors in the factorization model can
be accessed through the `.factors` property:
```py
factors = fac.factors
factors
# >>>  <'data' field of tensor b>
factors[0]
# >>> DeviceArray([-1.,  4.], dtype=float32)
```
Only the data for the `b` tensor is returned as `a` is not an optimizable
tensor.

!!! warning
    The `.factors` can be assigned to new data as a `setter` method is
    implemented. However, this is **NOT** recommended and only used during the
    optimization loop when [factorizing](../factorize) a model for a target
    tensor.

- **All factor** tensors irrespective of the optimizable flag, can be accessed
through the `.all_factors` property:
```py
factors = fac.all_factors
factors
# >>>  <'data' fields of tensors a, b>
factors[0]
# >>> DeviceArray([1., 2., 3.], dtype=float32)
factors[1]
# >>> DeviceArray([-1.,  4.], dtype=float32)
```

## Evaluation

A FunFact factorization model can be evaluated simply by calling it or 
equivalenty by using the forward method:
```py
fac()
fac.forward()
```
This performs all the operations and contractions as specified in the tensor 
expression that was used when instantiating the model based on the tensor data 
in the leaf tensors and returns the end result. For our example, this returns:
```
DeviceArray([[-1.,  4.],
             [-2.,  8.],
             [-3., 12.]], dtype=float32)
```
### Partial evaluation

A factorization model can also be evaluated for just certain elements or slices
of the full model. FunFact follows largely the same pattern as NumPy array
slicing:
```py
fac[0,0]
# >>> DeviceArray([[-1.]], dtype=float32)
fac[-1,-1]
# >>> DeviceArray([[12.]], dtype=float32)
fac[1:3,-1]
# >>>  DeviceArray([[ 8.],
#                   [12.]], dtype=float32)
```

!!! note
    The requested array slices are analyzed and only the necessary parts of the
    factor tensors are used to compute the result. This avoids evaluation of the
    full model.

### Factors

The factors of a factorization model can also be accessed by name with the `[]`
operation:
```py
fac['a']
# >>> DeviceArray([1., 2., 3.], dtype=float32)
fac['b']
# >>> DeviceArray([-1.,  4.], dtype=float32)
```

## Penalties

The **penalties** on the optimizable factors of a factorization model can
evaluated through the `.penalties` method:
```py
fac.penalty()
# >>> DeviceArray(1., dtype=float32)
```

This method takes two keyword arguments:

- `sum_leafs`: a boolean flag indicating if the penalties are summed over
all the leaf tensors.
- `sum_vec`: a boolean flag indicating if the penalties are summed over all
instances in a *vectorized* model, see the [factorization page](../factorize)
for more information about vectorization.