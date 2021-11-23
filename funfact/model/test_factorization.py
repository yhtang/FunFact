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
    assert pytest.approx(elementwise, tol) == full
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
    assert pytest.approx(elementwise, tol) == full
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
    assert pytest.approx(elementwise, tol) == full
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
    # one element
    idx = (0, 2, 1)
    full = fac()[idx]
    elementwise = np.squeeze(fac[idx])
    assert pytest.approx(elementwise, tol) == full
    # slices
    idx = (1, slice(0, 2), 0)
    full = fac()[idx]
    elementwise = np.squeeze(fac[idx])
    for f, e in zip(full, elementwise):
        assert pytest.approx(e, tol) == f

    # Tucker decomposition
    T = tensor('T', 3, 3, 3)
    u1 = tensor('u_1', 4, 3)
    u2 = tensor('u_2', 5, 3)
    u3 = tensor('u_3', 6, 3)
    i1, i2, i3, k1, k2, k3 = indices('i_1, i_2, i_3, k_1, k_2, k_3')
    tsrex = T[k1, k2, k3] * u1[i1, k1] * u2[i2, k2] * u3[i3, k3]
    fac = Factorization(tsrex)
    # one element
    idx = (0, 2, 1)
    full = fac()[idx]
    elementwise = np.squeeze(fac[idx])
    assert pytest.approx(elementwise, tol) == full
    # slices
    idx = (1, slice(0, 2), 0)
    full = fac()[idx]
    elementwise = np.squeeze(fac[idx])
    for f, e in zip(full, elementwise):
        assert pytest.approx(e, tol) == f
    idx = (slice(0, 3), slice(None), 2)
    full = fac()[idx]
    elementwise = np.squeeze(fac[idx])
    for f, e in zip(full, elementwise):
        assert pytest.approx(e, tol) == f
