---
title: 'FunFact: Build Your Own Tensor Decomposition Model in a Breeze'
tags:
  - Python
  - tensor decomposition
  - machine learning
  - quantum computing
authors:
  - name: Daan Camps
    orcid: 0000-0003-0236-4353
    affiliation: "1"
    email: dcamps@lbl.gov
  - name: Yu-Hang Tang^[corresponding author]
    orcid: 0000-0001-7424-5439
    affiliation: "1"
    email: Tang.Maxin@gmail.com
affiliations:
 - name: Applied Mathematics and Computational Research Division, Lawrence Berkeley National Laboratory, Berkeley, CA 94720, USA
   index: 1
date: 14 February, 2022
bibliography: paper.bib
---

# Summary

`FunFact` is a Python package that aims to simplify the design of matrix and tensor factorization algorithms. It features a powerful programming interface that augments the NumPy API with Einstein notations for writing concise tensor expressions. Given an arbitrary forward calculation scheme, the package will solve the corresponding inverse problem using stochastic gradient descent, automatic differentiation, and multi-replica vectorization. Its application areas include tensor decomposition, quantum circuit synthesis, and neural network compression. It is GPU- and parallelization-ready thanks to modern numerical linear algebra (NLA) backends such as `JAX` [@jax] and `PyTorch` [@pytorch].

# Statement of Need

Tensor factorizations have found numerous applications in various domains [@tbook], [@treview]. Among the most prominent are tensor networks in quantum physics [@tnetwork], tensor decompositions in machine learning [@tensorly] and signal processing [@tml], [@bss], and quantum computation [@qc].

Thus far, most tensor factorization models are solved by special-purpose algorithms designed to factor the target data into a model with the prescribed structure. Furthermore, the models that are being used are often limited to linear contractions between the factor tensors, such as standard inner and outer products, elementwise multiplications, and matrix Kronecker products. Extending such a special-purpose solver to more generalized models can be daunting, especially if nonlinear operations are considered.

`FunFact` solves this problem and fills the gap. It offers an embedded Domain Specific Language (eDSL) in Python for creating nonlinear tensor algebra expressions that use generalized Einstein operations. Using the eDSL, users can create custom tensor expressions and immediately use them to solve the corresponding inverse factorization problem. `FunFact` solves this inverse problem by combining stochastic gradient descent, automatic differentiation, and model vectorization for multi-replica learning. This combination achieves instantaneous time-to-algorithm for all conceivable tensor factorization models. It allows the user to explore the entire universe of nonlinear tensor factorization models. 

![Tensor rank, Tucker, tensor network, and singular value decompositions are among the most popular factorization models that have found numerous applications. However, the popular models studied in the literature only form a small subset of all possible tensor factorization models that can be constructed from generalized contractions, semiring operations, nonlinearities, and more. `FunFact` allows users to probe this vastly larger universe of models through an eDSL for tensor expressions. From the forward computation defined by a tensor expression, `FunFact` can solve the inverse factorization problem using a combination of techniques such as lazy evaluation, automatic differentiation, and stochastic gradient descent.](docs/assets/overview.pdf)

# Functionality

`FunFact`'s core functionality consists of three parts:

1. A rich and flexible eDSL to express complicated tensor factorization models with a concise notation.
2. Forward evaluation of user-defined tensor expressions.
3. Using backpropagation and automatic differentiation to compute the model gradients and to optimize the factorization model for a target tensor using stochastic gradient descent.


## eDSL for tensor expressions

The central notions in the `FunFact` eDSL are `tensor` and `index` objects that can be used to construct tensor expressions.
Indices are used to label the dimensions of tensor expressions. Repeated indices are contracted over in an Einstein operation between two tensor expressions. Abstract `tensor` objects can be initialized either (1) based on their shape or (2) from concrete numerical data.
Optional arguments for `tensor` objects include

* a human-readable label,
* an `initializer` that provides a generator for a particular distribution,
* a `condition` that the tensor is expected to satisfy, such as nonnegativity and orthogonality, which is implemented as a penalty term during the optimization process, and
* an `optimizable` flag that indicates if the tensor should be updated during the optimization process.

Tensor expressions can be indexed, indexless, or hybrid. `FunFact` implements a tensor algebra language model based on a [context-free grammar](https://funfact.readthedocs.io/en/latest/pages/user-guide/cfg/). Index decorators, explicit output index specification, generalized contractions with semiring operations, nonlinearities, and other features make the `FunFact` language rich and flexible. 

## Forward evaluation

In `FunFact`, tensor expressions are handled by a lazy evaluation model. Only basic analyses are performed after the user defines a tensor expression, such as shape and dimensionality checking. After that, the computational graph of the expression is stored for later use. A tensor expression can be explicitly evaluated in the forward direction, *i.e.,* from leaf tensors to the result, using the `Factorization` class, which serves as an interpreter of tensor expression.
 
## Optimizing for a target tensor

The central capability of `FunFact` is implemented in the `factorize` method, which:

1. run the model in the forward direction, and then
2. run a backpropagation pass with automatic differentiation to find the gradients of a given cost function with regard to the leaf tensors in the tensor expression, and then
3. update the leaf tensors using a stochastic gradient descent algorithm. 

The `factorize` method allows a user to optimize a model as defined by *any* tensor expression towards a target tensor, thereby solving the inverse problem. The method has many knobs that the user can fine-tune for the problem at hand to achieve faster and better convergence. These include the learning rate, the optimization integrator, the cost function, the weights of the penalty terms, and any of the numerous hyperparameters.

# Example

We illustrate the use and flexibility of `FunFact` by providing reference tensor expressions for a few matrix and tensor decomposition models. Upper-case symbols are assumed to be abstract tensors of the appropriate dimensions, while lower-case symbols are abstract indices.

| Tensor Expression | Description |
| ----------------- | ----------- |
| `low_rank = U[i, r] * V[j, r]` | Rank-$r$ decomposition of $(i, j)$ matrix |
| `tucker = Z[r1, r2, r3] * S1[r1, i1] * S2[r2, i2] * S3[r3, i3]` | Rank-$(r_1, r_2, r_3)$ Tucker decomposition of $(i_1, i_2, i_3)$ tensor [@treview] |
| `tensor_rank = (A[i1, ~r] * B[i2, r]) * C[i3, r]` | Rank-$r$ tensor rank decomposition of $(i_1, i_2, i_3)$ tensor [@treview] |
| `tensor_train = G1[i1, r1] * G2[i2, r1, r2] * G3[i3, r2, r3] * G4[i4, r3]` | Tensor train decomposition of $(i_1, i_2, i_3, i_4)$ tensor [@ttd] |
| `rbf = ff.exp(-(U[i, ~k] - V[j, ~k])**2) * A[k] + B[[]]` | RBF kernel decomposition with $r$ terms of $(i, j)$ matrix [@rbf] |
| `quantum_gate = ff.eye(2**i) & ff.tensor(4, 4, prefer=cond.Unitary)` | Two-qubit unitary quantum gate [@nc] |

# Related research and software

`FunFact` is closely related to several other software packages that provide Einstein notations and Domain Specific Languages (DSL) for tensor algebra. Notable examples are `TensorOperations.jl` [@to] that provides Einstein index notations in julia, `Tensor Comprehensions` [@tc] that provides a DSL to automatically synthesize high-performance machine learning kernels, `einops` [@einops] that enables tensor operations through readable and reliable code, `TACO` [@TACO]: the tensor algebra compiler, and `COMET` [@comet] which is designed for high-performance contractions of sparse tensors. `FunFact` is distinct from all of the aforementioned projects in that it aims to solve the inverse decomposition problem from the model description as a nonlinear tensor algebra expression. Additionally, `FunFact` offers increased generality compared to other tensor decomposition software libraries such as `Tensorly` [@tensorly], `Tensor Toolbox` [@ttoolbox] or `Tensorlab` [@tlab] which only provide specialized implementations for computing fixed-form tensor decompositions such as Tucker or tensor rank decompositions.

# Acknowledgement

The authors thank Liza Rebrova for her input on this work.
This work was supported by the Laboratory Directed Research and Development Program of Lawrence Berkeley National Laboratory under U.S. Department of Energy Contract No. DE-AC02-05CH11231.

# References
