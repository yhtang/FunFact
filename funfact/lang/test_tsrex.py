#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from copy import deepcopy
from unittest.mock import MagicMock
import numpy as np
from ._tsrex import (
    _BaseEx,
    TsrEx,
    index,
    indices,
    tensor
)


# def test_baseex():

#     expr = tensor('T', 2, 3, 4)

#     assert isinstance(expr, _BaseEx)
#     assert hasattr(expr, 'asciitree')
#     assert hasattr(expr, '_static_analyzed')
#     assert hasattr(expr, 'shape')
#     assert hasattr(expr, 'live_indices')
#     assert hasattr(expr, 'ndim')
#     assert hasattr(expr, 'einspec')
#     assert 'testtensor' in repr(expr.asciitree)
#     assert callable(expr.asciitree)
#     assert isinstance(expr.asciitree(stdout=False), str)
#     assert expr.asciitree(stdout=True) is None
#     assert expr.asciitree('name') is None
#     assert expr._repr_html_().startswith('$$')
#     assert expr._repr_html_().endswith('$$')


# # def test_arithmetics():

# #     lhs = ArithmeticMixin()
# #     rhs = ArithmeticMixin()
# #     lhs.root = MagicMock(name='lhs')
# #     rhs.root = MagicMock(name='rhs')

# #     for ex in [
# #         lhs + rhs, lhs + 1, 1 + rhs,
# #         lhs - rhs, lhs - 1, 1 - rhs,
# #         lhs * rhs, lhs * 1, 1 * rhs,
# #         lhs / rhs, lhs / 1, 1 / rhs,
# #     ]:
# #         assert isinstance(ex, _BaseEx)
# #         assert isinstance(ex, TsrEx)
# #         assert isinstance(ex, EinopEx)
# #         assert isinstance(ex, ArithmeticMixin)

# #     for ex in [
# #         lhs + rhs, lhs - rhs, lhs * rhs, lhs / rhs,
# #     ]:
# #         assert repr(lhs.root) in repr(ex.root)
# #         assert repr(rhs.root) in repr(ex.root)

# #     for ex in [
# #         -lhs,
# #         -(lhs + rhs),
# #         -(lhs + 1),
# #         -(1 + rhs),
# #         -(1 * rhs)
# #     ]:
# #         assert isinstance(ex, _BaseEx)
# #         assert isinstance(ex, TsrEx)
# #         assert isinstance(ex, ArithmeticMixin)
# #         assert ex.root.name == 'neg'

# #     for ex in [
# #         lhs**rhs,
# #         (lhs + 2)**rhs,
# #         lhs**1,
# #         1**rhs,
# #         lhs**(1 - rhs)
# #     ]:
# #         assert isinstance(ex, _BaseEx)
# #         assert isinstance(ex, TsrEx)
# #         assert isinstance(ex, ArithmeticMixin)
# #         assert ex.root.name == 'pow'


# # YT: this test below should go into the shape analyzer
# # since we are only testing for the API here, not behavior
# '''
# def test_shape():
#     A = tensor('A', 2, 3)
#     B = tensor('B', 3, 4)
#     i, j = indices('i, j')
#     tsrex = A[[i, j]] * B[[i, j]]
#     with pytest.raises(SyntaxError):
#         tsrex.shape
#     tsrex = A[[*i, j]] * B[[*i, j]]
#     with pytest.raises(SyntaxError):
#         tsrex.shape
#     tsrex = A[[*i, *j]] * B[[*i, *j]]
#     expected_shape = (6, 12)
#     for t, e in zip(tsrex.shape, expected_shape):
#         assert t == e
# '''


# def test_index_renaming_mixin():

#     a = tensor('a', 2, 3)
#     b = tensor('b', 3, 4)
#     i, j, k, p, q = indices('i, j, k, p, q')
#     e1 = a[i, j] * b[j, k]
#     e2 = e1[p, q]
#     assert isinstance(e2, TsrEx)
#     assert e2.root.name == e1.root.name
#     assert e1.live_indices[0] == i.root.item
#     assert e1.live_indices[1] == k.root.item
#     assert e2.live_indices[0] == p.root.item
#     assert e2.live_indices[1] == q.root.item

#     e3 = e1[j, q]
#     assert e3.root.name == e1.root.name
#     assert e3.live_indices[0] == j.root.item
#     assert e3.live_indices[1] == q.root.item

#     e4 = e1[j, i]
#     assert e4.root.name == e1.root.name
#     assert e4.live_indices[0] == j.root.item
#     assert e4.live_indices[1] == i.root.item

#     with pytest.raises(SyntaxError):
#         e1[i]
#     with pytest.raises(SyntaxError):
#         e1[i, j, k]
#     with pytest.raises(SyntaxError):
#         e1[i, b]
#     with pytest.raises(SyntaxError):
#         e1[b, b]


# # def test_transposition_mixin():
# #     a = TranspositionMixin()
# #     a.root = MagicMock(name='mock_tensor')
# #     for n in range(0, 9):
# #         indices = [MagicMock(root='mock_index') for _ in range(n)]
# #         b = a >> indices
# #         assert isinstance(b, TsrEx)
# #         assert b.root.name == 'tran'
# #         assert repr(a.root) in repr(b.root)
# #         for i in indices:
# #             assert repr(i.root) in repr(b.root)


# def test_indexex():
#     i = index()
#     for j in [~i, *i]:
#         assert isinstance(j, TsrEx)

#     j = ~i
#     assert j.root.bound is True
#     assert j.root.kron is False

#     k, = [*i]
#     assert k.root.bound is False
#     assert k.root.kron is True

#     m, = [*(~i)]
#     assert m.root.bound is False
#     assert m.root.kron is True

#     n, = [*i]
#     n = ~n
#     assert n.root.bound is True
#     assert n.root.kron is False


# def test_tensorex():
#     t = tensor('u', 2, 3, 4)
#     i, j, k = indices(3)
#     ex = t[i, j, k]
#     assert isinstance(ex, TsrEx)
#     assert repr(t.root) in repr(ex.root)
#     assert repr(i.root) in repr(ex.root)
#     assert repr(j.root) in repr(ex.root)
#     assert repr(k.root) in repr(ex.root)


# # def test_einop_ex():
# #     e1 = EinopEx(MagicMock(root=MagicMock(outidx=[])))
# #     e1_copy = deepcopy(e1)
# #     i, j = [MagicMock(root='i'), MagicMock(root='j')]
# #     e2 = e1 >> [i, j]
# #     assert isinstance(e2, EinopEx)
# #     assert e1.root == e1_copy.root
# #     assert e2.root != e1.root
# #     assert e2.root.outidx.items[0] == 'i'
# #     assert e2.root.outidx.items[1] == 'j'


# def test_index():
#     i = index()
#     assert isinstance(i, TsrEx)
#     assert i.root.bound is False
#     assert i.root.kron is False

#     j = index('j')
#     assert isinstance(j, TsrEx)
#     assert str(j.root.item.symbol) == 'j'
#     assert j.root.bound is False
#     assert j.root.kron is False


# def test_indices():
#     i, = indices('i')
#     assert isinstance(i, TsrEx)

#     for n in range(10):
#         J = indices(n)
#         assert len(J) == n
#         for j in J:
#             assert isinstance(j, TsrEx)

#     p, q = indices('p, q')
#     assert isinstance(p, TsrEx)
#     assert isinstance(q, TsrEx)

#     r, s = indices('r s')
#     assert isinstance(r, TsrEx)
#     assert isinstance(s, TsrEx)

#     K = indices('a, b, c, d, e, f, g')
#     assert len(K) == 7

#     with pytest.raises(RuntimeError):
#         indices(-1)
#     with pytest.raises(RuntimeError):
#         indices(1.5)
#     with pytest.raises(RuntimeError):
#         indices('p+q')
#     with pytest.raises(RuntimeError):
#         indices(None)


# def test_tensor():

#     for n in range(1, 10):
#         u = tensor(np.ones(n))
#         assert isinstance(u, TsrEx)
#         assert u.ndim == 1
#         assert u.shape == (n,)
#         assert u.root.decl.optimizable is False

#         v = tensor(np.eye(n))
#         assert isinstance(v, TsrEx)
#         assert v.ndim == 2
#         assert v.shape == (n, n)
#         assert v.root.decl.optimizable is False

#         w = tensor(n)
#         assert isinstance(w, TsrEx)
#         assert w.ndim == 1
#         assert w.shape == (n,)
#         assert w.root.decl.optimizable is True


# @pytest.mark.parametrize(
#     'spec', [
#         ('x', np.random.randn(4, 3, 2)),
#         (np.random.randn(4, 3, 2),),
#         ('x', 4, 3, 2),
#         (4, 3, 2)
#     ]
# )
# def test_tensor_creation(spec):

#     t = tensor(*spec, optimizable=True)
#     assert isinstance(t, TsrEx)
#     assert t.shape == (4, 3, 2)
#     assert t.ndim == 3
#     assert t.root.decl.optimizable is True


# def test_tensor_0d():
#     t = tensor('x')
#     assert isinstance(t, TsrEx)
#     assert t.shape == ()
#     assert t.ndim == 0


# @pytest.mark.parametrize(
#     'spec', [
#         (1.5,),
#         (2, 'x', 3, 4),
#         (2, 3, 'y')
#     ]
# )
# def test_tensor_creation_failure(spec):

#     with pytest.raises(RuntimeError):
#         tensor(*spec)
