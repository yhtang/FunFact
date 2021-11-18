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
    i, j, k, m = indices('i, j, k, m')
    tsrex = A[i, j] * B[j, k]
    fac = Factorization(tsrex)
    # one element
    idx = (1, 0)
    full = fac()[idx]
    elementwise = np.squeeze(fac[idx])
    for f, e in zip([full], [elementwise]):
        assert pytest.approx(e, tol) == f
    # one row
    idx = (1, slice(None))
    full = fac()[idx]
    elementwise = np.squeeze(fac[idx])
    for f, e in zip([full], [elementwise]):
        assert pytest.approx(e, tol) == f
    # one column
    idx = (slice(None), 0)
    full = fac()[idx]
    elementwise = np.squeeze(fac[idx])
    for f, e in zip([full], [elementwise]):
        assert pytest.approx(e, tol) == f

    # outer product
    A = tensor('A', 10)
    B = tensor('B', 5)
    tsrex = A[i] * B[j]
    fac = Factorization(tsrex)
    # one element
    idx = (1, 0)
    full = fac()[idx]
    elementwise = np.squeeze(fac[idx])
    for f, e in zip([full], [elementwise]):
        assert pytest.approx(e, tol) == f
    # slices
    idx = (slice(1, 6), slice(2, 4))
    full = fac()[idx]
    elementwise = np.squeeze(fac[idx])
    for f, e in zip([full], [elementwise]):
        assert pytest.approx(e, tol) == f

    # bound index in matrix product
    A = tensor('A', 2, 3)
    B = tensor('A', 3, 4)
    tsrex = A[i, j] * B[~j, k]
    fac = Factorization(tsrex)
    # one element
    idx = (1, 0, 1)
    full = fac()[idx]
    elementwise = np.squeeze(fac[idx])
    for f, e in zip([full], [elementwise]):
        assert pytest.approx(e, tol) == f
    # slices
    idx = (slice(0, 2), slice(2, 4), 0)
    full = fac()[idx]
    elementwise = np.squeeze(fac[idx])
    for f, e in zip(full, elementwise):
        assert pytest.approx(e, tol) == f

    # combination of different contractions
    A = tensor('A', 2, 3, 4)
    B = tensor('B', 4, 3, 2)
    tsrex = A[i, j, k] * B[k, ~j, m]
    fac = Factorization(tsrex)
    idx = (0, 2, 1)
    full = fac()[idx]
    elementwise = np.squeeze(fac[idx])
    for f, e in zip([full], [elementwise]):
        assert pytest.approx(e, tol) == f
    idx = (1, slice(0, 2), 0)
    full = fac()[idx]
    elementwise = np.squeeze(fac[idx])
    for f, e in zip(full, elementwise):
        assert pytest.approx(e, tol) == f
