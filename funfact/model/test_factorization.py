#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from unittest.mock import MagicMock as M
from funfact import active_backend as ab
from funfact.lang._ast import Primitives as P
from ._factorization import Factorization


def _test_factorization_props(fac):
    assert isinstance(fac, Factorization)
    assert hasattr(fac, '_tsrex')
    assert hasattr(fac, 'tsrex')
    assert hasattr(fac, 'shape')
    assert hasattr(fac, 'ndim')
    assert hasattr(fac, 'ndim')
    assert hasattr(fac, 'factors')
    assert hasattr(fac, 'all_factors')
    assert fac.forward() == fac()
    assert isinstance(fac.factors, Factorization._NodeView)
    assert isinstance(fac.all_factors, Factorization._NodeView)


def test_init_factory():
    fac = Factorization(M(), _secret='50A-2117')
    _test_factorization_props(fac)
    fac = Factorization.from_tsrex(M(), dtype=M(), initialize=M())
    _test_factorization_props(fac)


def test_nodeview():
    nodes = [M(data=0), M(data=1)]
    nv = Factorization._NodeView('data', nodes)
    assert isinstance(nv.__repr__(), str)
    assert len(nv) == 2
    for i, node in enumerate(nv):
        assert i == node
    for i in range(len(nv)):
        assert nv[i] == i
        nv[i] = i + 10
        assert nv[i] == i + 10


@pytest.mark.parametrize('test_case', [
    (0, slice(0, 1, 2), slice(0, 1, 2)),
    (0, 1, slice(1, 2, None)),
    (0, -1, slice(-1, None, None)),
    (0, Ellipsis, None),
    (0, [0, 1], (0, 1)),
    (0, 1.2, RuntimeError)
])
def test_as_slice(test_case):
    i, axis,  result = test_case
    as_slice = Factorization._as_slice
    if isinstance(result, type) and issubclass(result, Exception):
        with pytest.raises(result):
            as_slice(i, axis)
    else:
        assert as_slice(i, axis) == result


def _gen_factors_mock(data, optimizable=True):
    root = M(data=data, decl=M(optimizable=optimizable))
    root.name = 'tensor'
    return Factorization(M(root=root), _secret='50A-2117')


def test_factors():
    fac = _gen_factors_mock(1, True)
    assert len(fac.factors) == 1
    assert len(fac.all_factors) == 1
    assert fac.factors[0] == fac.all_factors[0]
    assert fac.factors[0] == 1
    fac.factors = [2]
    assert fac.factors[0] == 2
    fac = _gen_factors_mock(1, False)
    assert len(fac.factors) == 0
    assert len(fac.all_factors) == 1
    assert fac.all_factors[0] == 1


def test_penalties():
    def _prefer(data, *args):
        return ab.sum(data)

    root = M(data=ab.tensor([1, 2, 3]),
             decl=M(optimizable=True, prefer=_prefer))
    root.name = 'tensor'
    fac = Factorization(M(root=root), _secret='50A-2117')
    assert fac.penalty() == 6


def test_get_set_item():
    root = M(data=1, decl=M(symbol='a', optimizable=True), ndim=1)
    root.name = 'tensor'
    fac = Factorization(M(root=root), _secret='50A-2117')
    assert fac['a'] == 1
    fac['a'] = 2
    assert fac['a'] == 2
    with pytest.raises(AttributeError):
        fac['unknown-factor']
    with pytest.raises(AttributeError):
        fac['unknown-factor'] = 2
    with pytest.raises(IndexError):
        fac[0, 0]
    with pytest.raises(IndexError):
        fac[0, ...]


def test_duplicate_factors():
    a = P.tensor(decl=M(symbol='a', optimizable=True, prefer=lambda *_: 0))
    b = P.tensor(decl=M(symbol='b', optimizable=True, prefer=lambda *_: 0))

    fac1 = Factorization(M(root=P.elem(a, a, 0, 'add')), _secret='50A-2117')
    fac2 = Factorization(M(root=P.elem(a, b, 0, 'add')), _secret='50A-2117')

    assert len(fac1.factors) == 1
    assert len(fac2.factors) == 2

    fac1.factors = [None]
    with pytest.raises(IndexError):
        fac2.factors = [None]
    fac2.factors = [None, None]

    assert len(fac1.penalty(sum_leafs=False)) == 1
    assert len(fac2.penalty(sum_leafs=False)) == 2
