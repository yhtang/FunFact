#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from ._factorization import Factorization
from funfact import tensor, indices


def test_elementwise():
    tol = 20 * np.finfo(np.float32).eps

    # matrix product
    A = tensor('A', 2, 2)
    B = tensor('B', 2, 2)
    i, j, k, m = indices('i, j, k, m')
    tsrex = A[i, j] * B[j, k]
    fac = Factorization.from_tsrex(tsrex)

    # one element
    idx = (1, 0)
    full = fac()[idx]
    elementwise = fac[idx]
    assert pytest.approx(np.ravel(elementwise), tol) == np.ravel(full)

    # one row
    idx = (1, slice(None))
    full = fac()[idx]
    elementwise = fac[idx]
    for f, e in zip(np.ravel(full), np.ravel(elementwise)):
        assert pytest.approx(e, tol) == f

    # one column
    idx = (slice(None), 0)
    full = fac()[idx]
    elementwise = fac[idx]
    for f, e in zip(np.ravel(full), np.ravel(elementwise)):
        assert pytest.approx(e, tol) == f

    # outer product
    A = tensor('A', 10)
    B = tensor('B', 5)
    tsrex = A[i] * B[j]
    fac = Factorization.from_tsrex(tsrex)

    # one element
    idx = (1, 0)
    full = fac()[idx]
    elementwise = fac[idx]
    assert pytest.approx(np.ravel(elementwise), tol) == np.ravel(full)

    # slices
    idx = (slice(1, 6), slice(2, 4))
    full = fac()[idx]
    elementwise = fac[idx]
    for f, e in zip(np.ravel(full), np.ravel(elementwise)):
        assert pytest.approx(e, tol) == f

    # bound index in matrix product
    A = tensor('A', 2, 3)
    B = tensor('B', 3, 4)
    tsrex = A[i, j] * B[~j, k]
    fac = Factorization.from_tsrex(tsrex)

    # one element
    idx = (1, 0, 1)
    full = fac()[idx]
    elementwise = fac[idx]
    assert pytest.approx(np.ravel(elementwise), tol) == np.ravel(full)

    # slices
    idx = (slice(0, 2), slice(2, 3), 0)
    full = fac()[idx]
    elementwise = fac[idx]
    for f, e in zip(np.ravel(full), np.ravel(elementwise)):
        assert pytest.approx(e, tol) == f

    # combination of different contractions
    A = tensor('A', 2, 3, 4)
    B = tensor('B', 4, 3, 2)
    tsrex = A[i, j, k] * B[k, ~j, m]
    fac = Factorization.from_tsrex(tsrex)

    # one element
    idx = (0, 2, 1)
    full = fac()[idx]
    elementwise = fac[idx]
    assert pytest.approx(np.ravel(elementwise), tol) == np.ravel(full)

    # slices
    idx = (1, slice(0, 2), 0)
    full = fac()[idx]
    elementwise = fac[idx]
    for f, e in zip(np.ravel(full), np.ravel(elementwise)):
        assert pytest.approx(e, tol) == f

    # Tucker decomposition
    T = tensor('T', 3, 3, 3)
    u1 = tensor('u_1', 4, 3)
    u2 = tensor('u_2', 5, 3)
    u3 = tensor('u_3', 6, 3)
    i1, i2, i3, k1, k2, k3 = indices('i_1, i_2, i_3, k_1, k_2, k_3')
    tsrex = T[k1, k2, k3] * u1[i1, k1] * u2[i2, k2] * u3[i3, k3]
    fac = Factorization.from_tsrex(tsrex)

    # one element
    idx = (0, 2, 1)
    full = fac()[idx]
    elementwise = fac[idx]
    assert pytest.approx(np.ravel(elementwise), tol) == np.ravel(full)

    # slices
    idx = (1, slice(0, 2), 0)
    full = fac()[idx]
    elementwise = fac[idx]
    for f, e in zip(np.ravel(full), np.ravel(elementwise)):
        assert pytest.approx(e, tol) == f
    idx = (slice(0, 3), slice(None), 2)
    full = fac()[idx]
    elementwise = fac[idx]
    for f, e in zip(np.ravel(full), np.ravel(elementwise)):
        assert pytest.approx(e, tol) == f


def test_Kronecker():
    tol = 2 * np.finfo(np.float32).eps
    dataA = np.reshape(np.arange(0, 6), (2, 3))
    dataB = np.reshape(np.arange(6, 15), (3, 3))
    A = tensor('A', dataA)
    B = tensor('B', dataB)
    i, j, k = indices('i, j, k')

    # regular Kronecker product
    tsrex = A[[*i, *j]] * B[i, j]
    fac = Factorization.from_tsrex(tsrex)
    out = fac()
    expected_shape = (6, 9)
    for o, f, e in zip(out.shape, fac.shape, expected_shape):
        assert o == e
        assert o == f
    ref = np.kron(dataA, dataB)
    assert np.allclose(out, ref, tol)

    # Kronecker product along first axis (Khatri-Rao)
    tsrex = A[[*i, ~j]] * B[i, j]
    fac = Factorization.from_tsrex(tsrex)
    out = fac()
    expected_shape = (6, 3)
    for o, f, e in zip(out.shape, fac.shape, expected_shape):
        assert o == e
        assert o == f
    ref = np.vstack([np.kron(dataA[:, k], dataB[:, k]) for k in
                    range(dataB.shape[1])]).T
    assert np.allclose(out, ref, tol)

    # Kronecker product along first axis, reduction second
    tsrex = A[[*i,  j]] * B[i, j]
    fac = Factorization.from_tsrex(tsrex)
    out = fac()
    expected_shape = (6,)
    for o, f, e in zip(out.shape, fac.shape, expected_shape):
        assert o == e
        assert o == f

    # Matrix product
    tsrex = A[[i,   j]] * B[j, k]
    fac = Factorization.from_tsrex(tsrex)
    out = fac()
    expected_shape = (2, 3)
    for o, f, e in zip(out.shape, fac.shape, expected_shape):
        assert o == e
        assert o == f
    ref = dataA @ dataB
    assert np.allclose(out, ref, tol)

    # No reduction
    tsrex = A[[i,  ~j]] * B[j, k]
    fac = Factorization.from_tsrex(tsrex)
    out = fac()
    expected_shape = (2, 3, 3)
    for o, f, e in zip(out.shape, fac.shape, expected_shape):
        assert o == e
        assert o == f

    # Kronecker product inner axis
    tsrex = A[[i,  *j]] * B[j, k]
    fac = Factorization.from_tsrex(tsrex)
    out = fac()
    expected_shape = (2, 9, 3)
    for o, f, e in zip(out.shape, fac.shape, expected_shape):
        assert o == e
        assert o == f
