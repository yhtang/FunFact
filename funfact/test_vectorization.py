#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from funfact import tensor, indices, Factorization
from .vectorization import vectorize, view


def test_vectorize_random():
    a = tensor('a', 2, 3)
    b = tensor('b', 3, 4)
    i, j, k = indices('i, j, k')
    tsrex = a[i, j] * b[j, k]
    nvec = 7
    tsrex_vector = vectorize(tsrex, nvec)
    assert tsrex_vector.ndim == tsrex.ndim + 1
    assert tsrex_vector.shape == (*tsrex.shape, nvec)
    assert tsrex_vector.root.lhs.indexless.abstract.shape == (*a.shape, nvec)
    assert tsrex_vector.root.rhs.indexless.abstract.shape == (*b.shape, nvec)

    fac = Factorization.from_tsrex(tsrex_vector)
    assert fac.ndim == tsrex_vector.ndim
    assert fac.all_factors[0].shape == (*a.shape, nvec)
    assert fac.all_factors[1].shape == (*b.shape, nvec)


def test_vectorize_concrete():
    a = tensor('a', 2, 3)
    b = tensor('b', np.eye(3))
    i, j, k = indices('i, j, k')
    tsrex = a[i, j] * b[j, k]
    nvec = 7
    tsrex_vector = vectorize(tsrex, nvec)
    assert tsrex_vector.ndim == tsrex.ndim + 1
    assert tsrex_vector.shape == (*tsrex.shape, nvec)
    assert tsrex_vector.root.lhs.indexless.abstract.shape == (*a.shape, nvec)
    assert tsrex_vector.root.rhs.indexless.abstract.shape == (*b.shape, nvec)

    fac = Factorization.from_tsrex(tsrex_vector)
    assert fac.ndim == tsrex_vector.ndim
    assert fac.all_factors[0].shape == (*a.shape, nvec)
    assert fac.all_factors[1].shape == (*b.shape, nvec)


def test_view_random():
    a = tensor('a', 2, 3)
    b = tensor('b', 3, 4)
    i, j, k = indices('i, j, k')
    tsrex = a[i, j] * b[j, k]
    nvec = 7
    tsrex_vector = vectorize(tsrex, nvec)
    vfac = Factorization.from_tsrex(tsrex_vector)
    for i in range(nvec):
        fac = view(vfac, tsrex, i)
        assert fac.ndim == tsrex.ndim
        assert fac.all_factors[0].shape == a.shape
        assert fac.all_factors[1].shape == b.shape

    # view last instance
    view(vfac, tsrex, -1)

    with pytest.raises(IndexError):
        view(vfac, tsrex, nvec)


def test_view_concrete():
    a = tensor('a', 2, 3)
    b = tensor('b', np.eye(3))
    i, j, k = indices('i, j, k')
    tsrex = a[i, j] * b[j, k]
    nvec = 7
    tsrex_vector = vectorize(tsrex, nvec)
    vfac = Factorization.from_tsrex(tsrex_vector)
    for i in range(nvec):
        fac = view(vfac, tsrex, i)
        assert fac.ndim == tsrex.ndim
        assert fac.all_factors[0].shape == a.shape
        assert fac.all_factors[1].shape == b.shape
        assert np.allclose(fac['b'], b.root.abstract.initializer)

    # view last instance
    view(vfac, tsrex, -1)

    with pytest.raises(IndexError):
        view(vfac, tsrex, nvec)
