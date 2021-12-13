# Welcome to the documentation of FunFact!

[![CI](https://github.com/yhtang/FunFact/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/yhtang/FunFact/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/yhtang/839011f3f7a6bab680b18cbd9a45d2d3/raw/coverage-develop.json)](https://badge.fury.io/py/funfact)
[![PyPI version](https://badge.fury.io/py/funfact.svg)](https://badge.fury.io/py/funfact)
[![Documentation Status](https://readthedocs.org/projects/funfact/badge/?version=latest)](https://funfact.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Overview

[FunFact](https://github.com/yhtang/FunFact.git) is a Python package that
enables flexible and concise expressions of tensor algebra through an Einstein
notation-based syntax. A particular emphasis is on automating the design of
matrix and tensor factorization models.  It’s areas of applications include
quantum circuit synthesis, tensor decomposition, and neural network
compression. It is GPU- and parallelization-ready thanks to modern numerical
linear algebra backends such as JAX/TensorFlow and PyTorch.
<!-- To this end, it leverages randomized combinatorial optimization
and stochastic gradient-based methods. -->

## Quick start guide

Install from pip:

```
pip install funfact
```

First, we will define some tensors and indices:

``` py
import funfact as ff
import numpy as np
a = ff.tensor('a', 5, 2, 4)                      # a random 3-way tensor
b = ff.tensor('b', np.arange(10).reshape(2, 5))  # a 2-way tensor (matrix)
u = ff.tensor('u', 4)                            # a random vectorvector
v = ff.tensor(5)                                 # an anonymous random vector
c = ff.tensor('c')                               # a scalar, i.e. 0-tensor
i, j, k = ff.indices('i, j, k')                  # define indices
```

Next, we create some tensor expressions. This only specifies the computation but does **not** carry out the evaluation
immediately.

``` py
a[i, j, k] * b[j, i]  # (1) contraction
u[i] * v[j]           # (2) outer product
u[i] - v[j]           # (3) pairwise substraction
```

1. Equivalent to `numpy.einsum('ijk,ji', a, b)`
2. Equivalent to `numpy.outer(u, v)`
3. Equivalent to `np.subtract.outer(u, v)`


<!-- 
b[i, ~j] * v[~j]      # multiply v with each row of b (3)
3. Equivalent to `numpy.einsum('ij,j->ij', b, v)` or `b * v[None, :]` -->

<!-- ## Indices and tables

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` -->


## How to cite

If you use this package for a publication (either in-paper or electronically), please cite it using the following DOI: https://doi.org/10.11578/dc.20210922.1

## Contributors

Current developers:

- [Yu-Hang "Maxin" Tang](https://github.com/yhtang)
- [Daan Camps](https://github.com/campsd)

Previou contributors:

- [Elizaveta Rebrova](https://github.com/erebrova)


## Copyright

FunFact Copyright (c) 2021, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of
any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.

## Funding Acknowledgment

This work was supported by the Laboratory Directed Research and Development Program of Lawrence Berkeley National Laboratory under U.S. Department of Energy Contract No. DE-AC02-05CH11231.



<!-- For full documentation visit [mkdocs.org](https://www.mkdocs.org).

## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files. -->
