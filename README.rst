FunFact
=======

FunFact is a library for computing the *functional factorization* of
algebraic tensors, a.k.a. multidimensional arrays. A functional
factorization, in our context, is a generalization of the (linear)
factorization of tensors. By generalization, we meant to replace the
standard inner/outer product between the factor tensors with nonlinear
operations.

For example, a rank-1 matrix can be factored into the outer product
between a column vector and a row vector:

.. math::


   M \approx \mathbf{u} \mathbf{v}^\mathsf{T},

where :math:`M` is an :math:`n \times m` matrix, :math:`\mathbf{u}` is a
:math:`n`-dimensional column vector, and :math:`\mathbf{v}` is a
:math:`m`-dimensional row vector. This can be equivalently represented
in indexed notation as

.. math::


   M_{ij} \approx \mathbf{u}_i \mathbf{v}_j.

Moreover, if we relace the standard multiplication operation between
:math:`\mathbf{u}_i` and :math:`\mathbf{v}_j` by an RBF function
:math:`\kappa(x, y) = \exp\left[-(x - y)^2\right]`, we then obtain an
`RBF approximation <https://arxiv.org/abs/2106.02018>`__ of :math:`M`
such that:

.. math::


   M_{ij} \approx \kappa(\mathbf{u}_i, \mathbf{v}_j).

Given the rich expressivity of nonlinear operators and functional forms,
we expect that a proper functional factorization of a tensor can yield
representations that are more compact than what is possible withtin the
existing linear framework. However, there is (obviously) no free lunch.
the challenges to obtain the functional factorization of a tensor is two
fold and involves - Finding the most appropriate **functional form**
given a specific piece of data, - Finding the **component tensors**
given the functional form for a specific data.

The two points above are exactly what we aim to facilitate using
FunFact.

Copyright
=========

FunFact Copyright (c) 2021, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of any
required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this
software, please contact Berkeley Lab’s Intellectual Property Office at
IPO@lbl.gov.

NOTICE. This Software was developed under funding from the U.S.
Department of Energy and the U.S. Government consequently retains
certain rights. As such, the U.S. Government has been granted for itself
and others acting on its behalf a paid-up, nonexclusive, irrevocable,
worldwide license in the Software to reproduce, distribute copies to the
public, prepare derivative works, and perform publicly and display
publicly, and to permit others to do so.
