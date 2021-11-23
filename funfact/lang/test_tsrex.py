#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import itertools as it
import numpy as np
from ._tsrex import index, indices, tensor


def test_abstract_tensor():
    u = tensor(np.ones(3))
    v = tensor('v', np.eye(4))
    w = tensor(4, 2, 5)
    x = tensor('x', 9, 2)
    for t in [u, v, w, x]:
        assert t.root.name == 'tensor'
        assert hasattr(t.root.abstract, 'symbol')
    # TODO: rewrite once [#31](https://github.com/yhtang/FunFact/issues/31)
    # is addressed.
    assert u.root.abstract.ndim == 1
    assert v.root.abstract.ndim == 2
    assert w.root.abstract.ndim == 3
    assert x.root.abstract.ndim == 2
    assert u.root.abstract.shape == (3,)
    assert v.root.abstract.shape == (4, 4)
    assert w.root.abstract.shape == (4, 2, 5)
    assert x.root.abstract.shape == (9, 2)


def test_transposition():

    A = tensor('A', 2, 3, 4, 5)
    i, j, k, r = indices('i, j, k, l')
    for perm in it.permutations([i, j, k, r]):
        AT = A[i, j, k, r].T[[*perm]]
        assert AT.root.name == 'tran'
        # TODO: More tests once
        # [#32](https://github.com/yhtang/FunFact/issues/32) is taken care of.
