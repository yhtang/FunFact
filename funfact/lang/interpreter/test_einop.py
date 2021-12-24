#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import funfact as ff
from funfact.backend import active_backend as ab

from ._einop import _einop


def test_einop():
    ff.use('numpy')
    tol = 20 * np.finfo(np.float32).eps

    # right elementwise multiplication
    lhs = ab.normal(0.0, 1.0, 3, 2)
    rhs = ab.normal(0.0, 1.0, 1)
    spec = 'ab,->ab|'
    res = _einop(spec, lhs, rhs, 'sum', 'multiply')
    assert(res.shape == lhs.shape)
    assert pytest.approx(np.ravel(res), tol) == np.ravel(lhs * rhs)

    # left elementwise multiplication
    lhs = ab.normal(0.0, 1.0, 1)
    rhs = ab.normal(0.0, 1.0, 3, 2)
    spec = ',ab->ab|'
    res = _einop(spec, lhs, rhs, 'sum', 'multiply')
    assert(res.shape == rhs.shape)
    assert pytest.approx(np.ravel(res), tol) == np.ravel(lhs * rhs)

    # scalar multiplication
    lhs = ab.normal(0.0, 1.0, 1)
    rhs = ab.normal(0.0, 1.0, 1)
    spec = ',->|'
    res = _einop(spec, lhs, rhs, 'sum', 'multiply')
    assert pytest.approx(np.ravel(res), tol) == np.ravel(lhs * rhs)

    # matrix elementwise multiplication
    lhs = ab.normal(0.0, 1.0, 3, 2)
    rhs = ab.normal(0.0, 1.0, 3, 2)
    spec = 'ab,ab->ab|'
    res = _einop(spec, lhs, rhs, 'sum', 'multiply')
    assert(res.shape == lhs.shape)
    assert pytest.approx(np.ravel(res), tol) == np.ravel(lhs * rhs)
