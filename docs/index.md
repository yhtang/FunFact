# FunFact: Build Your Own Tensor Decomposition Model in a Breeze

[![CI](https://github.com/yhtang/FunFact/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/yhtang/FunFact/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/yhtang/839011f3f7a6bab680b18cbd9a45d2d3/raw/coverage-develop.json)](https://badge.fury.io/py/funfact)
[![PyPI version](https://badge.fury.io/py/funfact.svg)](https://badge.fury.io/py/funfact)
[![Documentation Status](https://readthedocs.org/projects/funfact/badge/?version=latest)](https://funfact.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

[FunFact](https://github.com/yhtang/FunFact.git) is a Python package that aims to simplify the design of matrix and tensor factorization algorithms. It features a powerful programming interface that augments the NumPy API with Einstein notations for writing concise tensor expressions. Given an arbitrary forward calculation scheme, the package will solve the corresponding inverse problem using stochastic gradient descent, automatic differentiation, and multi-replica vectorization. Its application areas include quantum circuit synthesis, tensor decomposition, and neural network compression. It is GPU- and parallelization-ready thanks to modern numerical linear algebra backends such as JAX/TensorFlow and PyTorch.

## Quick start example: semi-nonnegative CP decomposition

!!! example "Install from pip"

    ``` bash 
    pip install -U funfact
    ```

!!! example "Package import"

    ``` py
    import funfact as ff
    import numpy as np
    ```

!!! example "Create target tensor"

    === "in"

        ``` py
        T = np.arange(60, dtype=np.float32).reshape(3, 4, 5); T
        ```

    === "out"

        ```
        array([[[ 0.,  1.,  2.,  3.,  4.],
                [ 5.,  6.,  7.,  8.,  9.],
                [10., 11., 12., 13., 14.],
                [15., 16., 17., 18., 19.]],

            [[20., 21., 22., 23., 24.],
                [25., 26., 27., 28., 29.],
                [30., 31., 32., 33., 34.],
                [35., 36., 37., 38., 39.]],

            [[40., 41., 42., 43., 44.],
                [45., 46., 47., 48., 49.],
                [50., 51., 52., 53., 54.],
                [55., 56., 57., 58., 59.]]], dtype=float32)
        ```

!!! example "Define abstract tensors and indices"

    ``` py
    R = 2
    a = ff.tensor('a', T.shape[0], R, prefer=ff.conditions.NonNegative())
    b = ff.tensor('b', T.shape[1], R)
    c = ff.tensor('c', T.shape[2], R)
    i, j, k, r = ff.indices('i, j, k, r')
    ```

!!! example "Create a tensor expression"

    !!! note inline end
        This only specifies the algebra but does **not** carry out the computation immediately)

    === "in"

        ``` py
        tsrex = (a[i, ~r] * b[j, r]) * c[k, r]; tsrex
        ```

    === "out"

        $${{{\boldsymbol{a}}_{{{i}}{{\widetilde{r}}}}}  {{\boldsymbol{b}}_{{{j}}{{r}}}}}  {{\boldsymbol{c}}_{{{k}}{{r}}}}$$


!!! example "Find rank-2 approximation"

    === "in"
    
        ``` py
        fac = ff.factorize(tsrex, T, max_steps=1000, nvec=8, penalty_weight=10)
        fac.factors
        ```
    
    === "out"
    
        ```
        100%|██████████| 1000/1000 [00:03<00:00, 304.00it/s]
        <'data' fields of tensors a, b, c>
        ```

!!! example "Reconstruction"

    === "in"
    
        ``` py
        fac()
        ```
    
    === "out"
    
        ```
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

!!! example "Factors"

    === "in"
    
        ``` py
        fac['a']
        ```
    
    === "out"
    
        ```
        DeviceArray([[1.788, 1.156],
                    [3.007, 0.582],
                    [4.226, 0.008]], dtype=float32)
        ```


    ===! "in"
    
        ``` py
        fac['b']
        ```
    
    === "out"
    
        ```
        DeviceArray([[-2.923, -4.333],
                    [-3.268, -3.541],
                    [-3.614, -2.749],
                    [-3.959, -1.957]], dtype=float32)
        ```

    
    ===! "in"
    
        ``` py
        fac['c']
        ```
    
    === "out"
    
        ```
        DeviceArray([[-3.271,  3.461],
                    [-3.341,  3.309],
                    [-3.41 ,  3.158],
                    [-3.479,  3.006],
                    [-3.548,  2.855]], dtype=float32)
        ```
    
## Statement of Need

Tensor factorizations have found numerous applications in a variety of domains [@tbook], [@treview]. Among the most prominent are tensor networks in quantum physics [@tnetwork], tensor decompositions in machine learning [@tensorly] and signal processing [@tml], [@bss], and quantum computation [@qc].

Thus far, most tensor factorization models are solved by special purpose algorithms designed to factor the target data into a model with the prescribed structure. Furthermore, the models that are being used are often limited to linear contractions between the factor tensors such as standard inner and outer products, elementwise multiplications, and matrix Kronecker products. Extending such a special-purpose solver to more generalized models can be a daunting task, especially if nonlinear operations are considered.

`FunFact` solves this problem and fills the gap. It offers an embedded Domain Specific Language (eDSL) in Python for creating nonlinear tensor algebra expressions that are based on generalized Einstein operations. User-defined tensor expressions can be immediately used to solve the corresponding factorization problem. `FunFact` solves this inverse problem by combining stochastic gradient descent, automatic differentiation, and model vectorization for multi-replica learning. This combination achieves instantaneous time-to-algorithm for all conceivable tensor factorization models and allows the user to explore the full universe of nonlinear tensor factorization models.

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
