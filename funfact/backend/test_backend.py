#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os
from . import (
    _use_default_backend,
    active_backend,
    available_backends,
    use,
)


# def test_import():
#     assert _active_backend is None


def test_available_backends():
    for backend, clsname in available_backends.items():
        assert os.path.exists(
            os.path.join(
                os.path.dirname(__file__),
                f'_{backend}.py'
            )
        )


def test_use():
    use('numpy')
    assert 'NumPy' in active_backend._get_active_backend().__qualname__

    with pytest.raises(RuntimeError):
        use('non-existing backend')


def test_use_default():
    with pytest.raises(RuntimeError):
        _use_default_backend({'none': None})


def test_active_backend():
    assert repr(active_backend) is not None
    assert repr(active_backend) != 'None'
