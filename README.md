# FunFact: Build Your Own Tensor Decomposition Model in a Breeze

[![CI](https://github.com/yhtang/FunFact/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/yhtang/FunFact/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/yhtang/839011f3f7a6bab680b18cbd9a45d2d3/raw/coverage-master.json)](https://badge.fury.io/py/funfact)
[![PyPI version](https://badge.fury.io/py/funfact.svg)](https://badge.fury.io/py/funfact)
[![Documentation Status](https://readthedocs.org/projects/funfact/badge/?version=latest)](https://funfact.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

[FunFact](https://github.com/yhtang/FunFact.git) is a Python package that aims to simplify the design of matrix and tensor factorization algorithms. It features a powerful programming interface that augments the NumPy API with Einstein notations for writing concise tensor expressions. Given an arbitrary forward calculation scheme, the package will solve the corresponding inverse problem using stochastic gradient descent, automatic differentiation, and multi-replica vectorization. Its application areas include quantum circuit synthesis, tensor decomposition, and neural network compression. It is GPU- and parallelization-ready thanks to modern numerical linear algebra backends such as JAX/TensorFlow and PyTorch.

## Quick start example: semi-nonnegative CP decomposition

Install from pip:

``` bash 
pip install -U funfact
```

Package import:

``` py
import funfact as ff
import numpy as np
```

Create target tensor:

``` py
T = np.arange(60, dtype=np.float32).reshape(3, 4, 5); T
```

Define abstract tensors and indices:

``` py
R = 2
a = ff.tensor('a', T.shape[0], R, prefer=ff.conditions.NonNegative())
b = ff.tensor('b', T.shape[1], R)
c = ff.tensor('c', T.shape[2], R)
i, j, k, r = ff.indices('i, j, k, r')
```

Create a tensor expression (only specifies the algebra but does **not** carry out the computation immediately):

``` py
tsrex = (a[i, ~r] * b[j, r]) * c[k, r]; tsrex
```

Find rank-2 approximation:

``` py
>>> fac = ff.factorize(tsrex, T, max_steps=1000, nvec=8, penalty_weight=10)
>>> fac.factors
100%|██████████| 1000/1000 [00:03<00:00, 304.00it/s]
<'data' fields of tensors a, b, c>
```

Reconstruction:
    
``` py
>>> fac()
DeviceArray([[[-0.234,  0.885,  2.004,  3.123,  4.243],
              [ 4.955,  5.979,  7.002,  8.025,  9.049],
              [10.145, 11.072, 12.   , 12.927, 13.855],
              [15.335, 16.167, 16.998, 17.83 , 18.661]],

             [[20.025, 21.014, 22.003, 22.992, 23.981],
              [25.019, 26.01 , 27.001, 27.992, 28.983],
              [30.013, 31.006, 31.999, 32.992, 33.985],
              [35.007, 36.002, 36.997, 37.992, 38.987]],

             [[40.281, 41.14 , 41.999, 42.858, 43.716],
              [45.082, 46.04 , 46.999, 47.958, 48.917],
              [49.882, 50.941, 51.999, 53.058, 54.117],
              [54.682, 55.841, 56.999, 58.158, 59.316]]], dtype=float32)
```
    
Examine factors:

``` py
>>> fac['a']
DeviceArray([[1.788, 1.156],
             [3.007, 0.582],
             [4.226, 0.008]], dtype=float32)
```
    
``` py
>>> fac['b']
DeviceArray([[-2.923, -4.333],
             [-3.268, -3.541],
             [-3.614, -2.749],
             [-3.959, -1.957]], dtype=float32)
```

``` py
>>> fac['c']
DeviceArray([[-3.271,  3.461],
             [-3.341,  3.309],
             [-3.41 ,  3.158],
             [-3.479,  3.006],
             [-3.548,  2.855]], dtype=float32)
```
    
## How to cite

If you use this package for a publication (either in-paper or electronically), please cite it using the following DOI: [https://doi.org/10.11578/dc.20210922.1](https://doi.org/10.11578/dc.20210922.1)

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
