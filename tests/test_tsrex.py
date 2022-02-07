#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import funfact as ff
import numpy as np
from funfact import active_backend as ab


@pytest.fixture
def my_arrays():
    a = np.reshape(np.arange(0, 6), (2, 3))
    b = np.reshape(np.arange(6, 14), (2, 4))
    return a, b


@pytest.fixture
def my_tensors(my_arrays):
    a_arr, b_arr = my_arrays
    a = ff.tensor('a', 2, 3, initializer=a_arr)
    b = ff.tensor('b', 2, 4, initializer=b_arr)
    return a, b


@pytest.fixture
def my_tsrexs(my_tensors):
    a, b = my_tensors
    i, j, k, p, q = ff.indices('i, j, k, p, q')
    return [
        a[k, i] * b[k, j],
        a * a,
        a[~k, ~i] * a[k, i],
        a[k, i] * a[k, i],
        a[k, i] * b[k, j] >> [j, i],
        a[k, i] * b[k, j] >> [p, q]
    ]


@pytest.fixture
def my_expected(my_arrays):
    a, b = my_arrays
    return [
        ((3, 4), ab.tensor(np.transpose(a) @ b, dtype=ab.float32)),
        ((2, 3), ab.tensor(a * a, dtype=ab.float32)),
        ((2, 3), ab.tensor(a * a, dtype=ab.float32)),
        ((),     ab.tensor(np.sum(a * a), dtype=ab.float32)),
        ((4, 3), ab.tensor(np.transpose(b) @ a, dtype=ab.float32)),
        SyntaxError
    ]


def test_fixture(my_tsrexs, my_expected):
    for tsrex, expected in zip(my_tsrexs, my_expected):
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                tsrex.shape
        else:
            shape, result = expected
            assert tsrex.shape == shape
            fac = ff.Factorization.from_tsrex(tsrex)
            assert ab.allclose(fac(), result)
