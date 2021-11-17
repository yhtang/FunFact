#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from ._terminal import AbstractIndex, AbstractTensor


def test_abstract_index():

    i = AbstractIndex()
    j = AbstractIndex()
    k = AbstractIndex('k_1')
    assert i == i
    assert j == j
    assert k == k
    assert i != j
    assert i != k
    assert j != k

    assert hash(i) != hash(j)
    assert hash(i) != hash(k)
    assert hash(j) != hash(k)

    assert isinstance(repr(i), str)
    assert isinstance(str(i), str)
    assert isinstance(i._repr_tex_(), str)
    assert isinstance(i._repr_html_(), str)


def test_abstract_tensor():

    u = AbstractTensor(3)
    v = AbstractTensor(4, 4, initializer=np.eye(4))
    w = AbstractTensor(9, 2, initializer=np.random.rand)
    x = AbstractTensor(4, 2, 3, symbol='x')
    y1 = AbstractTensor(4, 2, 3, symbol='y_1')
    assert u.ndim == 1
    assert v.ndim == 2
    assert w.ndim == 2
    assert x.ndim == 3
    assert u.shape == (3,)
    assert v.shape == (4, 4)
    assert w.shape == (9, 2)
    assert x.shape == (4, 2, 3)

    for t in [u, v, w, x, y1]:
        assert isinstance(repr(t), str)
        assert isinstance(str(t), str)
        assert isinstance(t._repr_tex_(), str)
        assert isinstance(t._repr_html_(), str)
