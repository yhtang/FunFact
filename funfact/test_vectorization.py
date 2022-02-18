#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest  # noqa: F401
import numpy as np
from funfact import tensor, Factorization
from .vectorization import vectorize, view


@pytest.mark.parametrize('case', [
    (7, -1, slice(0, -1), True),
    (7, 0, slice(1, None), False)
])
def test_vectorize_random(case):
    tsrex = tensor(2, 3)
    nvec, vec_ax, data_ax, append = case
    tsrex_vector = vectorize(tsrex, nvec, append)
    assert tsrex_vector.ndim == tsrex.ndim + 1
    assert tsrex_vector.shape[data_ax] == tsrex.shape
    assert tsrex_vector.shape[vec_ax] == nvec

    fac = Factorization.from_tsrex(tsrex_vector)
    assert fac.ndim == tsrex_vector.ndim
    assert fac.all_factors[0].shape[data_ax] == tsrex.shape
    assert fac.all_factors[0].shape[vec_ax] == nvec


@pytest.mark.parametrize('case', [
    (7, -1, slice(0, -1), True),
    (7, 0, slice(1, None), False)
])
def test_vectorize_concrete(case):
    tsrex = tensor(2, 3)
    nvec, vec_ax, data_ax, append = case
    tsrex_vector = vectorize(tsrex, nvec, append)
    assert tsrex_vector.ndim == tsrex.ndim + 1
    assert tsrex_vector.shape[data_ax] == tsrex.shape
    assert tsrex_vector.shape[vec_ax] == nvec

    fac = Factorization.from_tsrex(tsrex_vector)
    assert fac.ndim == tsrex_vector.ndim
    assert fac.all_factors[0].shape[data_ax] == tsrex.shape
    assert fac.all_factors[0].shape[vec_ax] == nvec


@pytest.mark.parametrize('append', [
    True,
    False
])
def test_view_random(append):
    nvec = 7
    tsrex = tensor(2, 3)
    tsrex_vector = vectorize(tsrex, nvec, append)
    fac = Factorization.from_tsrex(tsrex)
    vfac = Factorization.from_tsrex(tsrex_vector)
    for i in range(nvec):
        fac = view(vfac.factors, fac, i, append)
        assert fac.ndim == tsrex.ndim
        assert fac.all_factors[0].shape == tsrex.shape

    # view last instance
    view(vfac.factors, fac, -1, append)


@pytest.mark.parametrize('append', [
    True,
    False
])
def test_view_concrete(append):
    nvec = 7
    tsrex = tensor('a', np.eye(3))
    tsrex_vector = vectorize(tsrex, nvec, append)
    fac = Factorization.from_tsrex(tsrex)
    vfac = Factorization.from_tsrex(tsrex_vector)
    for i in range(nvec):
        fac = view(vfac.factors, fac, i, append)
        assert fac.ndim == tsrex.ndim
        assert fac.all_factors[0].shape == tsrex.shape
        assert np.allclose(fac['a'], tsrex.root.decl.initializer)

    # view last instance
    view(vfac.factors, fac, -1, append)
