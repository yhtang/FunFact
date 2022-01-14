#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest  # noqa: F401
import numpy as np
from funfact import tensor, indices, Factorization
from .vectorization import vectorize, view


@pytest.mark.parametrize('case', [
    (7, -1, slice(0, -1), True),
    (7, 0, slice(1, None), False)
])
def test_vectorize_random(case):
    a = tensor('a', 2, 3)
    b = tensor('b', 3, 4)
    i, j, k = indices('i, j, k')
    tsrex = a[i, j] * b[j, k]
    nvec, vec_ax, data_ax, append = case
    tsrex_vector = vectorize(tsrex, nvec, append)
    assert tsrex_vector.ndim == tsrex.ndim + 1
    assert tsrex_vector.shape[data_ax] == tsrex.shape
    assert tsrex_vector.shape[vec_ax] == nvec
    assert tsrex_vector.root.lhs.tensor.decl.shape[data_ax] == a.shape
    assert tsrex_vector.root.lhs.tensor.decl.shape[vec_ax] == nvec
    assert tsrex_vector.root.rhs.tensor.decl.shape[data_ax] == b.shape
    assert tsrex_vector.root.rhs.tensor.decl.shape[vec_ax] == nvec

    fac = Factorization.from_tsrex(tsrex_vector)
    assert fac.ndim == tsrex_vector.ndim
    assert fac.all_factors[0].shape[data_ax] == a.shape
    assert fac.all_factors[0].shape[vec_ax] == nvec
    assert fac.all_factors[1].shape[data_ax] == b.shape
    assert fac.all_factors[1].shape[vec_ax] == nvec


@pytest.mark.parametrize('case', [
    (7, -1, slice(0, -1), True),
    (7, 0, slice(1, None), False)
])
def test_vectorize_concrete(case):
    a = tensor('a', 2, 3)
    b = tensor('b', np.eye(3))
    i, j, k = indices('i, j, k')
    tsrex = a[i, j] * b[j, k]
    nvec, vec_ax, data_ax, append = case
    tsrex_vector = vectorize(tsrex, nvec, append)
    assert tsrex_vector.ndim == tsrex.ndim + 1
    assert tsrex_vector.shape[data_ax] == tsrex.shape
    assert tsrex_vector.shape[vec_ax] == nvec
    assert tsrex_vector.root.lhs.tensor.decl.shape[data_ax] == a.shape
    assert tsrex_vector.root.lhs.tensor.decl.shape[vec_ax] == nvec
    assert tsrex_vector.root.rhs.tensor.decl.shape[data_ax] == b.shape
    assert tsrex_vector.root.rhs.tensor.decl.shape[vec_ax] == nvec

    fac = Factorization.from_tsrex(tsrex_vector)
    assert fac.ndim == tsrex_vector.ndim
    assert fac.all_factors[0].shape[data_ax] == a.shape
    assert fac.all_factors[0].shape[vec_ax] == nvec
    assert fac.all_factors[1].shape[data_ax] == b.shape
    assert fac.all_factors[1].shape[vec_ax] == nvec


@pytest.mark.parametrize('case', [
    True,
    False
])
def test_view_random(case):
    a = tensor('a', 2, 3)
    b = tensor('b', 3, 4)
    i, j, k = indices('i, j, k')
    tsrex = a[i, j] * b[j, k]
    append = case
    nvec = 7
    tsrex_vector = vectorize(tsrex, nvec, append)
    vfac = Factorization.from_tsrex(tsrex_vector)
    for i in range(nvec):
        fac = view(vfac, tsrex, i, append)
        assert fac.ndim == tsrex.ndim
        assert fac.all_factors[0].shape == a.shape
        assert fac.all_factors[1].shape == b.shape

    # view last instance
    view(vfac, tsrex, -1, append)

    # view out-of-bound
    with pytest.raises(IndexError):
        view(vfac, tsrex, nvec, append)

    # view with wrong append flag
    with pytest.raises(IndexError):
        view(vfac, tsrex, nvec-1, not append)


@pytest.mark.parametrize('case', [
    True,
    False
])
def test_view_concrete(case):
    a = tensor('a', 2, 3)
    b = tensor('b', np.eye(3))
    i, j, k = indices('i, j, k')
    tsrex = a[i, j] * b[j, k]
    append = case
    nvec = 7
    tsrex_vector = vectorize(tsrex, nvec, append)
    vfac = Factorization.from_tsrex(tsrex_vector)
    for i in range(nvec):
        fac = view(vfac, tsrex, i, append)
        assert fac.ndim == tsrex.ndim
        assert fac.all_factors[0].shape == a.shape
        assert fac.all_factors[1].shape == b.shape
        assert np.allclose(fac['b'], b.root.decl.initializer)

    # view last instance
    view(vfac, tsrex, -1)

    # view out-of-bound
    with pytest.raises(IndexError):
        view(vfac, tsrex, nvec)

    # view with wrong append flag
    with pytest.raises(IndexError):
        view(vfac, tsrex, nvec-1, not append)
