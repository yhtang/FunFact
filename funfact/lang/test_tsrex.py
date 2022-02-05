#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from ._tsrex import (
    TsrEx,
    index,
    indices,
    tensor
)


def test_index():

    i = index()
    assert isinstance(i, TsrEx)


def test_indices():

    for i in indices(2):
        assert isinstance(i, TsrEx)

    for i in indices('i, j'):
        assert isinstance(i, TsrEx)

    with pytest.raises(RuntimeError):
        indices(-1)

    with pytest.raises(RuntimeError):
        indices('123')


def test_tensor():

    tensor(1),
    tensor(1, 2),
    tensor('t'),
    tensor('t', 1),
    tensor('t', 1, 2),
    tensor('t', np.ones(3)),
    tensor('t', np.ones((3, 3))),
    tensor(np.ones(3)),
    tensor(np.ones((3, 3))),


def test_tensor_exception():

    with pytest.raises(RuntimeError):
        tensor(-1, -2)


def test_property():

    t = tensor(3, 3)
    hasattr(t, 'shape')
    hasattr(t, 'ndim')
    hasattr(t, 'live_indices')
    hasattr(t, 'einspec')


def test_asciitree():
    a = tensor('a', 2, 3)
    assert hasattr(a, 'asciitree')
    assert 'a' in str(a.asciitree)
    assert 'None' in str(a.asciitree('null', hide_empty=False))
    assert 'None' in a.asciitree('null', stdout=False, hide_empty=False)
    assert 'None' not in a.asciitree('null', stdout=False, hide_empty=True)


def test_html_repr():
    a = tensor('a', 2, 3)
    assert r'\boldsymbol{a}' in a._repr_html_()


def test_binary_operators():

    a = tensor('a', 3, 4)
    b = tensor('b', 4, 5)
    c = tensor('c', 3, 4)
    i = index('i')
    j = index('j')

    def _check(tsrex):
        assert isinstance(tsrex, TsrEx)

    _check(~i)
    _check(*i)
    _check(-a)
    _check(a @ b)
    _check(a + c)
    _check(a - c)
    _check(a * c)
    _check(a ** c)
    _check(a / c)
    _check(a + 1)
    _check(a - 1)
    _check(a * 1)
    _check(a ** 1)
    _check(a / 1)
    _check(1 + a)
    _check(1 - a)
    _check(1 * a)
    _check(1 ** a)
    _check(1 / a)
    _check(a & b)
    _check(a & 2)
    _check(2 & a)
    _check(a[i, j])
    _check(a[i, j] >> [j, i])

    _check(a + c.root)
    _check(a.root + c)
    _check(a @ b.root)
    _check(a.root @ b)

    with pytest.raises(TypeError):
        a + None

    with pytest.raises(TypeError):
        a + 'xyz'
