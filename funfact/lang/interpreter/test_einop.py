#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest  # noqa: F401
from funfact.backend import active_backend as ab
from ._einop import _einop


@pytest.mark.parametrize('case', [
    ((3, 2), (), 'ab,->ab'),  # right elementwise multiplication
    ((), (3, 2), ',ab->ab'),  # left elementwise multiplication
    ((), (), ',->'),  # scalar multiplication
    ((3, 2), (3, 2), 'ab,ab->ab'),  # matrix elementwise multiplication
])
def test_einsum(case):
    tol = 20 * ab.finfo(ab.float32).eps

    lhs_shape, rhs_shape, spec = case
    lhs = ab.normal(0.0, 1.0, lhs_shape)
    rhs = ab.normal(0.0, 1.0, rhs_shape)
    truth = ab.einsum(spec, lhs, rhs)
    res = _einop(spec, lhs, rhs, 'sum', 'multiply')
    assert truth.shape == res.shape
    assert ab.allclose(truth, res, tol)
