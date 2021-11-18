#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from ._factorization import Factorization
from funfact import tensor, indices


def test_elementwise():
    tol = 2 * np.finfo(np.float32).eps

    # matrix product
    A = tensor('A', 2, 2)
    B = tensor('B', 2, 2)
    i, j, k = indices('i, j, k')
    tsrex = A[i, j] * B[j, k]
    f = Factorization(tsrex)
    # one element
    idx = (1, 0)
    full = f()[idx]
    elementwise = np.squeeze(f[idx])
    for f, e in zip([full], [elementwise]):
        assert pytest.approx(e, tol) == f
    # one row
    idx = (1, slice(None))
    full = f()[idx]
    elementwise = np.squeeze(f[idx])
    for f, e in zip([full], [elementwise]):
        assert pytest.approx(e, tol) == f
    # one column
    idx = (slice(None), 0)
    full = f()[idx]
    elementwise = np.squeeze(f[idx])
    for f, e in zip([full], [elementwise]):
        assert pytest.approx(e, tol) == f

    # outer product
    A = tensor('A', 10)
    B = tensor('B', 5)
    tsrex = A[i] * B[j]
    f = Factorization(tsrex)
    f = Factorization(tsrex)
    # one element
    idx = (1, 0)
    full = f()[idx]
    elementwise = np.squeeze(f[idx])
    for f, e in zip([full], [elementwise]):
        assert pytest.approx(e, tol) == f
    # slices
    idx = (slice(1, 6), slice(2, 4))
    full = f()[idx]
    elementwise = np.squeeze(f[idx])
    for f, e in zip([full], [elementwise]):
        assert pytest.approx(e, tol) == f
