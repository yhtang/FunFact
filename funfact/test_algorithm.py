#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest  # noqa: F401
from funfact import active_backend as ab
from .lang import tensor
from .algorithm import factorize


@pytest.mark.parametrize('test_case', [
    (tensor(3, 3), ab.ones((3, 3)), [ab.ones((3, 3))]),
    (tensor(3, 3), ab.zeros((3, 3)), [ab.zeros((3, 3))]),
    (tensor(3, 4), ab.eye(3, 4), [ab.eye(3, 4)]),
])
def test_simple(test_case):

    tsrex, target, truth = test_case

    fac = factorize(tsrex, target)

    for a, b in zip(fac.factors, truth):
        assert ab.allclose(a, b, atol=1e-3)


def test_dtype():

    tsrex = tensor(2, 2)

    fac = factorize(tsrex, ab.ones((2, 2)), dtype=ab.complex64, max_steps=1)

    for f in fac.factors:
        assert f.dtype == ab.complex64
