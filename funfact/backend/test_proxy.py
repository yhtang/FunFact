#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os
from . import (
    active_backend,
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
    x = active_backend.normal(0.0, 1.0, (10000,))
    y = active_backend.relu(x)
    assert active_backend.relu(-10) == 0
    assert active_backend.relu(0) == 0
    assert active_backend.relu(10) == 10
    assert active_backend.is_native(y)
    assert active_backend.all(y >= 0)
    assert active_backend.all(y[x > 0] == x[x > 0])


def test_celu():
    x = active_backend.normal(0.0, 1.0, (10000,))
    y = active_backend.celu(x)
    assert active_backend.celu(-10) < 0
    assert active_backend.celu(0) >= 0
    assert active_backend.celu(10) == 10
    assert active_backend.is_native(y)
    assert active_backend.all(y >= y)
    assert active_backend.all(y[x > 0] == x[x > 0])


def test_sigmoid():
    x = active_backend.normal(0.0, 1.0, (10000,))
    y = active_backend.sigmoid(x)
    assert active_backend.sigmoid(-10) == pytest.approx(0, abs=1e-4)
    assert active_backend.sigmoid(0) == pytest.approx(0.5, abs=1e-4)
    assert active_backend.sigmoid(10) == pytest.approx(1, abs=1e-4)
    assert active_backend.is_native(y)
    assert active_backend.all(y >= 0)


def test_log_sum_exp():
    x = active_backend.normal(0.0, 1.0, (10000,))
    y = active_backend.log_sum_exp(x)
    assert x.max() <= y
    assert y <= x.max() + active_backend.log(len(x))


def test_use():
    use('numpy')
    assert 'NumPy' in repr(active_backend)

    with pytest.raises(ModuleNotFoundError):
        use('non-existing backend')


def test_active_backend():
    assert repr(active_backend) is not None
    assert repr(active_backend) != 'None'
