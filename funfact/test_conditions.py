#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from .conditions import (
    UpperTriangular,
    Unitary,
    Diagonal,
    NonNegative,
    NoCondition,
    vmap
)
import numpy as np
from funfact.backend import active_backend as ab


@pytest.mark.parametrize('cond', [
    UpperTriangular,
    Unitary,
    Diagonal,
    NonNegative,
    NoCondition
])
def test_generic(cond):
    tol = 20 * np.finfo(np.float32).eps
    # with default arguments
    condition = cond()
    tensor = ab.tensor(np.eye(6, 8))
    assert pytest.approx(condition(tensor), tol) == 0.0

    # with specified arguments
    condition = cond(weight=10.0, elementwise='l1', reduction='sum')
    assert pytest.approx(condition(tensor), tol) == 0.0

    # vectorized append
    vec_condition = vmap(cond(), append=True)
    tensor = ab.tensor(np.stack([np.eye(6, 8), np.eye(6, 8)], axis=-1))
    assert pytest.approx(vec_condition(tensor), tol) == 0.0

    # vectorized prepend
    vec_condition = vmap(cond(), append=False)
    tensor = ab.tensor(np.stack([np.eye(6, 8), np.eye(6, 8)], axis=0))
    assert pytest.approx(vec_condition(tensor), tol) == 0.0
