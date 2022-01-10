#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from funfact.backend import active_backend as ab
from ._einop import _einop


def test_einop():
    tol = 20 * ab.finfo(ab.float32).eps

    # right elementwise multiplication
    lhs = ab.normal(0.0, 1.0, (3, 2))
    rhs = ab.normal(0.0, 1.0, (1,))
    spec = 'ab,->ab|'
    res = _einop(spec, lhs, rhs, 'sum', 'multiply')
    assert(res.shape == lhs.shape)
    assert ab.allclose(res, lhs * rhs, tol)

    # left elementwise multiplication
    lhs = ab.normal(0.0, 1.0, (1,))
    rhs = ab.normal(0.0, 1.0, (3, 2))
    spec = ',ab->ab|'
    res = _einop(spec, lhs, rhs, 'sum', 'multiply')
    assert(res.shape == rhs.shape)
    assert ab.allclose(res, lhs * rhs, tol)

    # scalar multiplication
    lhs = ab.normal(0.0, 1.0, (1,))
    rhs = ab.normal(0.0, 1.0, (1,))
    spec = ',->|'
    res = _einop(spec, lhs, rhs, 'sum', 'multiply')
    assert ab.allclose(res, lhs * rhs, tol)

    # matrix elementwise multiplication
    lhs = ab.normal(0.0, 1.0, (3, 2))
    rhs = ab.normal(0.0, 1.0, (3, 2))
    spec = 'ab,ab->ab|'
    res = _einop(spec, lhs, rhs, 'sum', 'multiply')
    assert(res.shape == lhs.shape)
    assert ab.allclose(res, lhs * rhs, tol)
