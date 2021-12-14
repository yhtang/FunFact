!!! warning

    Page under construction
    
# RBF Approximation

FunFact makes it easy to compute the *functional factorization* of algebraic tensors, a.k.a. multidimensional arrays. A functional factorization, in this context, is a generalization of the (linear) factorization of tensors. By generalization, we meant to replace the standard inner/outer product between the factor tensors with some nonlinear operations.

For example, a rank-1 matrix can be factored into the outer product between a column vector and a row vector:

$$
M \approx \mathbf{u} \mathbf{v}^\mathsf{T},
$$

where $M$ is an $n \times m$ matrix, $\mathbf{u}$ is a $n$-dimensional column vector, and $\mathbf{v}$ is a $m$-dimensional row vector. This can be equivalently represented in indexed notation as

$$
M_{ij} \approx \mathbf{u}_i \mathbf{v}_j.
$$

Moreover, if we relace the standard multiplication operation between $\mathbf{u}_i$ and $\mathbf{v}_j$ by an RBF function $\kappa(x, y) = \exp\left[-(x - y)^2\right]$, we then obtain an *RBF approximation* of $M$ such that:

$$
M_{ij} \approx \kappa(\mathbf{u}_i, \mathbf{v}_j).
$$

Given the rich expressivity of nonlinear operators and functional forms, we expect that a proper functional factorization of a tensor can yield representations that are more compact than what is possible withtin the existing linear framework. However, there is (obviously) no free lunch. the challenges to obtain the functional factorization of a tensor is two fold and involves
- Finding the most appropriate **functional form** given a specific piece of data,
- Finding the **component tensors** given the functional form for a specific data.
