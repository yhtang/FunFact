#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import importlib
from . import (
    _active_backend,
    _use_default_backend,
    active_backend,
    available_backends,
    use,
)


def test_import():
    assert _active_backend is None


def test_available_backends():
    for backend, clsname in available_backends.items():
        assert hasattr(
            importlib.import_module(f'funfact.backend._{backend}'), clsname
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
