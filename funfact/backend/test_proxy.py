#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os
import numpy as np
from . import (
    active_backend as ab,
    available_backends,
    use,
)


def test_available_backends():
    for backend in available_backends:
        assert os.path.exists(
            os.path.join(
                os.path.dirname(__file__),
                f'_{backend}.py'
            )
        )


def test_relu():
    x = ab.normal(0.0, 1.0, (10000,))
    y = ab.relu(x)
    assert ab.relu(ab.tensor(-10)) == 0
    assert ab.relu(ab.tensor(0)) == 0
    assert ab.relu(ab.tensor(10)) == 10
    assert ab.is_native(y)
    assert ab.all(y >= 0)
    assert ab.all(y[x > 0] == x[x > 0])


def test_celu():
    x = ab.normal(0.0, 1.0, (10000,))
    y = ab.celu(x)
    assert ab.celu(ab.tensor(-10)) < 0
    assert ab.celu(ab.tensor(0)) >= 0
    assert ab.celu(ab.tensor(10)) == 10
    assert ab.is_native(y)
    assert ab.all(y >= y)
    assert ab.all(y[x > 0] == x[x > 0])


def test_sigmoid():
    x = ab.normal(0.0, 1.0, (10000,))
    y = ab.sigmoid(x)
    assert ab.sigmoid(ab.tensor(-10)) == pytest.approx(0, abs=1e-4)
    assert ab.sigmoid(ab.tensor(0)) == pytest.approx(0.5, abs=1e-4)
    assert ab.sigmoid(ab.tensor(10)) == pytest.approx(1, abs=1e-4)
    assert ab.is_native(y)
    assert ab.all(y >= 0)


def test_log_sum_exp():
    x = ab.normal(0.0, 1.0, (10000,))
    y = ab.log_sum_exp(x)
    assert x.max() <= y
    assert y <= x.max() + np.log(len(x))


def test_use():
    use('numpy')
    assert 'NumPy' in repr(ab)

    with pytest.raises(ModuleNotFoundError):
        use('non-existing backend')


def test_active_backend():
    assert repr(ab) is not None
    assert repr(ab) != 'None'
