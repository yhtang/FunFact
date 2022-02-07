#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import funfact as ff
import numpy as np
from funfact import active_backend as ab


@pytest.fixture
def my_arrays():
    a = ab.tensor(np.reshape(np.arange(0, 6), (2, 3)), dtype=ab.float32)
    b = ab.tensor(np.reshape(np.arange(6, 14), (2, 4)), dtype=ab.float32)
    c = ab.tensor(np.reshape(np.arange(15, 75), (3, 4, 5)), dtype=ab.float32)
    return a, b, c


@pytest.fixture
def my_tsrexs(my_arrays):
    a_arr, b_arr, c_arr = my_arrays
    a = ff.tensor('a', 2, 3, initializer=a_arr)
    b = ff.tensor('b', 2, 4, initializer=b_arr)
    c = ff.tensor('c', 3, 4, 5, initializer=c_arr)
    i, j, k, p, q = ff.indices('i, j, k, p, q')
    return [
        a[k, i] * b[k, j],
        a * a,
        a[~k, ~i] * a[k, i],
        a[k, i] * a[k, i],
        a[k, i] * b[k, j] >> [j, i],
        a[k, i] * b[k, j] >> [p, q],
        a[k, i] * c[i, j, p]
    ]


@pytest.fixture
def my_expected(my_arrays):
    a, b, c = my_arrays
    return [
        ((3, 4), ab.transpose(a, (1, 0)) @ b),
        ((2, 3), a * a),
        ((2, 3), a * a),
        ((),     ab.sum(a * a)),
        ((4, 3), ab.transpose(b, (1, 0)) @ a),
        SyntaxError,
        ((2, 4, 5), ab.einsum('ki, ijp->kjp', a, c))
    ]


def test_tsrex_shape_result(my_tsrexs, my_expected):
    for tsrex, expected in zip(my_tsrexs, my_expected):
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                tsrex.shape
        else:
            shape, result = expected
            assert tsrex.shape == shape
            fac = ff.Factorization.from_tsrex(tsrex)
            assert ab.allclose(fac(), result)
