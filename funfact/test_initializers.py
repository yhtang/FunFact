#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from funfact import active_backend as ab
from .initializers import (
    Ones,
    Zeros,
    Eye,
    Normal,
    Uniform,
    VarianceScaling,
    stack
)


all_initializers = [
    Ones,
    Zeros,
    Normal,
    Uniform,
    VarianceScaling
]


@pytest.mark.parametrize('init', all_initializers)
def test_generic(init):
    initializer = init()
    tensor = initializer((2, 3, 4))
    assert tensor.shape == (2, 3, 4)

    tensor = initializer((10,))
    assert tensor.shape == (10,)

    tensor = initializer(10)
    assert tensor.shape == (10,)

    if init is VarianceScaling:
        with pytest.raises(IndexError):
            initializer(())
    else:
        tensor = initializer(())
        assert tensor.shape == ()


def test_ones():
    initializer = Ones()
    for n in initializer(100):
        assert n == 1.0


def test_zeros():
    initializer = Zeros()
    for n in initializer(100):
        assert n == 0.0


def test_eye():
    initializer = Eye()
    n = 9
    t = initializer((n, n))
    for i in range(n):
        for j in range(n):
            assert t[i, j] == (1 if i == j else 0)

    with pytest.raises(Exception):
        initializer(1)
    with pytest.raises(Exception):
        initializer((1, 2, 3))


def test_normal():
    initializer = Normal(std=1.0)
    with pytest.raises(AssertionError):
        t = initializer(1000)
        assert t.min() >= -2.0 and t.max() <= 2.0

    initializer = Normal(std=1.0, truncation=True)
    t = initializer(1000)
    assert t.min() >= -2.0 and t.max() <= 2.0

    initializer = Normal(std=1.0, truncation=1.0)
    t = initializer(1000)
    assert t.min() >= -1.0 and t.max() <= 1.0


def test_variance_scaling():
    for distribution in ['uniform', 'normal', 'truncated']:
        initializer = VarianceScaling(distribution=distribution)
        tensor = initializer(1000)
        assert tensor.shape == (1000,)

    with pytest.raises(ValueError):
        initializer = VarianceScaling(distribution='nonexisting-distribution')


@pytest.mark.parametrize('init', [Eye, Ones, Zeros, Eye(), Ones(), Zeros()])
@pytest.mark.parametrize('append', [True, False])
def test_stack(init, append):

    base_initializer = init() if isinstance(init, type) else init
    stacked_initializer = stack(init, append=append)()
    stacked_tensor = stacked_initializer((2, 3, 4))
    assert stacked_tensor.shape == (2, 3, 4)
    if append is True:
        base_tensor = base_initializer((2, 3))
        for i in range(4):
            assert ab.allclose(stacked_tensor[..., i], base_tensor)
    else:
        base_tensor = base_initializer((3, 4))
        for i in range(2):
            assert ab.allclose(stacked_tensor[i, ...], base_tensor)
