#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest  # noqa: F401
import numpy as np
from funfact.lang import tensor
from ._initialization import LeafInitializer


def test_none():
    x = tensor('x', 3, 3) | LeafInitializer()
    assert x.root.data.shape == (3, 3)


def test_default_dtype():
    ini = LeafInitializer(np.float64)
    assert ini.dtype == np.float64

    ini = LeafInitializer(np.float32)
    assert ini.dtype == np.float32


def test_class():
    class Distribution:
        def __init__(self, dtype):
            self.dtype = dtype

        def __call__(self, shape):
            return np.zeros(shape, dtype=self.dtype)

    x = tensor('x', 3, 3, initializer=Distribution) | LeafInitializer()
    assert x.root.data.shape == (3, 3)
    assert np.allclose(x.root.data, 0.0)


def test_obj():
    class Distribution:
        def __init__(self, dtype):
            self.dtype = dtype

        def __call__(self, shape):
            return np.zeros(shape, dtype=self.dtype)

    x = tensor('x', 3, 3, initializer=Distribution(np.float64))
    x = x | LeafInitializer()
    assert x.root.data.shape == (3, 3)
    assert x.root.data.dtype == np.float64
    assert np.allclose(x.root.data, 0.0)


def test_concrete():
    x = tensor('x', 3, 3, initializer=np.ones((3, 3)))
    x = x | LeafInitializer()
    assert x.root.data.shape == (3, 3)
    assert np.allclose(x.root.data, 1.0)


def test_broadcast():
    x = tensor('x', 6, 3, initializer=np.ones((3, 3)))
    x = x | LeafInitializer()
    assert x.root.data.shape == (6, 3)
    assert np.allclose(x.root.data, 1.0)

    with pytest.raises(ValueError):
        x = tensor('x', 5, 3, initializer=np.ones((3, 3)))
        x = x | LeafInitializer()
