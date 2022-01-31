#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest  # noqa: F401
from collections import namedtuple
from funfact.backend import active_backend as ab
from ._einop import _einop


_einspec = namedtuple(
    'einspec', [
        'op_reduce', 'op_elementwise', 'tran_lhs', 'tran_rhs', 'index_lhs',
        'index_rhs', 'ax_contraction'
    ]
)


@pytest.mark.parametrize('case', [
    # scalar multiplication
    (
        (), (), (),
        ',->',
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(),
            tran_rhs=(),
            index_lhs=(),
            index_rhs=(),
            ax_contraction=()),
    ),
    # right elementwise multiplication
    (
        (3,), (), (3,),
        'a,->a',
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(0,),
            tran_rhs=(),
            index_lhs=(slice(None),),
            index_rhs=(None,),
            ax_contraction=()),
    ),
    (
        (3,), (), (3,),
        'a,->a',
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(0,),
            tran_rhs=(),
            index_lhs=(slice(None),),
            index_rhs=(None,),
            ax_contraction=()),
    ),
    (
        (3,), (), (3,),
        'a,',
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(0,),
            tran_rhs=(),
            index_lhs=(slice(None),),
            index_rhs=(None,),
            ax_contraction=()),
    ),
    (
        (3, 2), (), (3, 2),
        'ab,->ab',
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(0, 1),
            tran_rhs=(),
            index_lhs=(slice(None), slice(None)),
            index_rhs=(None, None),
            ax_contraction=()),
    ),
    (
        (3, 2), (), (3, 2),
        'ab,',
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(0, 1),
            tran_rhs=(),
            index_lhs=(slice(None), slice(None)),
            index_rhs=(None, None),
            ax_contraction=()),
    ),
    # left elementwise multiplication
    (
        (), (3,), (3,),
        ',a->a',
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(),
            tran_rhs=(0,),
            index_lhs=(None,),
            index_rhs=(slice(None),),
            ax_contraction=()),
    ),
    (
        (), (3,), (3,),
        ',a->a',
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(),
            tran_rhs=(0,),
            index_lhs=(None,),
            index_rhs=(slice(None),),
            ax_contraction=()),
    ),
    (
        (), (3,), (3,),
        ',a',
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(),
            tran_rhs=(0,),
            index_lhs=(None,),
            index_rhs=(slice(None),),
            ax_contraction=()),
    ),
    (
        (), (3, 2), (3, 2),
        ',ab->ab',
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(),
            tran_rhs=(0, 1),
            index_lhs=(None, None),
            index_rhs=(slice(None), slice(None)),
            ax_contraction=()),
    ),
    # vector dot product
    (
        (10,), (10,), (),
        'i,i',
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(0,),
            tran_rhs=(0,),
            index_lhs=(slice(None),),
            index_rhs=(slice(None),),
            ax_contraction=(0,)),
    ),
    (
        (10,), (10,), (10,),
        'i,i->i',
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(0,),
            tran_rhs=(0,),
            index_lhs=(slice(None),),
            index_rhs=(slice(None),),
            ax_contraction=()),
    ),
    # matrix elementwise multiplication
    (
        (3, 2), (3, 2), (3, 2),
        'ab,ab->ab',
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(0, 1),
            tran_rhs=(0, 1),
            index_lhs=(slice(None), slice(None)),
            index_rhs=(slice(None), slice(None)),
            ax_contraction=()),
    ),
    # inner product and contractions
    (
        (10, 3), (3, 10), (10, 10),
        'ij,jk',
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(0, 1),
            tran_rhs=(1, 0),
            index_lhs=(slice(None), None, slice(None)),
            index_rhs=(None, slice(None), slice(None)),
            ax_contraction=(2,)),
    ),
    (
        (10, 3), (3, 10), (10, 3, 10),
        'ij,jk->ijk',
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(0, 1),
            tran_rhs=(0, 1),
            index_lhs=(slice(None), slice(None), None),
            index_rhs=(None, slice(None), slice(None)),
            ax_contraction=()),
    ),
    (
        (2, 3, 4), (3, 4, 5), (2, 5),
        'ijk,jkl',
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(0, 1, 2),
            tran_rhs=(2, 0, 1),
            index_lhs=(slice(None), None, slice(None), slice(None)),
            index_rhs=(None, slice(None), slice(None), slice(None)),
            ax_contraction=(2, 3)),
    ),
    (
        (2, 3, 4), (3, 4), (2,),
        'ijk,jk',
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(0, 1, 2),
            tran_rhs=(0, 1),
            index_lhs=(slice(None), slice(None), slice(None)),
            index_rhs=(None, slice(None), slice(None)),
            ax_contraction=(1, 2)),
    ),
    (
        (2, 3, 4), (5, 4, 3), (2, 5),
        'ijk,lkj',
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(0, 1, 2),
            tran_rhs=(0, 2, 1),
            index_lhs=(slice(None), None, slice(None), slice(None)),
            index_rhs=(None, slice(None), slice(None), slice(None)),
            ax_contraction=(2, 3)),
    ),
    (
        (4, 3, 2), (5, 4, 3), (2, 5),
        'ijk,lij',
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(2, 0, 1),
            tran_rhs=(0, 1, 2),
            index_lhs=(slice(None), None, slice(None), slice(None)),
            index_rhs=(None, slice(None), slice(None), slice(None)),
            ax_contraction=(2, 3)),
    ),
])
def test_einsum(case):
    tol = 100 * ab.finfo(ab.float32).eps

    lhs_shape, rhs_shape, result_shape, npspec, ffspec = case
    lhs = ab.normal(0.0, 1.0, lhs_shape)
    rhs = ab.normal(0.0, 1.0, rhs_shape)
    truth = ab.einsum(npspec, lhs, rhs)
    res = _einop(lhs, rhs, ffspec, result_shape)
    assert truth.shape == res.shape
    assert ab.allclose(truth, res, tol)


@pytest.mark.parametrize('case', [
    (
        (10,), (10,), (10, 10),
        'add',
        _einspec(
            op_reduce='sum',
            op_elementwise='add',
            tran_lhs=(0,),
            tran_rhs=(0,),
            index_lhs=(slice(None), None),
            index_rhs=(None, slice(None)),
            ax_contraction=()),
    ),
    (
        (10,), (10,), (10, 10),
        'subtract',
        _einspec(
            op_reduce='sum',
            op_elementwise='subtract',
            tran_lhs=(0,),
            tran_rhs=(0,),
            index_lhs=(slice(None), None),
            index_rhs=(None, slice(None)),
            ax_contraction=()),
    ),
    (
        (10,), (10,), (10, 10),
        'multiply',
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(0,),
            tran_rhs=(0,),
            index_lhs=(slice(None), None),
            index_rhs=(None, slice(None)),
            ax_contraction=()),
    ),
    (
        (10,), (10,), (10, 10),
        'divide',
        _einspec(
            op_reduce='sum',
            op_elementwise='divide',
            tran_lhs=(0,),
            tran_rhs=(0,),
            index_lhs=(slice(None), None),
            index_rhs=(None, slice(None)),
            ax_contraction=()),
    ),
])
def test_generic_outer(case):
    tol = 20 * ab.finfo(ab.float32).eps

    lhs_shape, rhs_shape, result_shape, pairwise, ffspec = case
    lhs = ab.normal(0.0, 1.0, lhs_shape)
    rhs = ab.normal(0.0, 1.0, rhs_shape)
    truth = getattr(ab, pairwise)(lhs[:, None], rhs[None, :])
    res = _einop(lhs, rhs, ffspec, result_shape)
    assert truth.shape == res.shape
    assert ab.allclose(truth, res, tol)


@pytest.mark.parametrize('case', [
    (
        (10,), (10,), (),
        'sum', 'add',
        _einspec(
            op_reduce='sum',
            op_elementwise='add',
            tran_lhs=(0,),
            tran_rhs=(0,),
            index_lhs=(slice(None),),
            index_rhs=(slice(None),),
            ax_contraction=(0,)),
    ),
    (
        (10,), (10,), (),
        'sum', 'subtract',
        _einspec(
            op_reduce='sum',
            op_elementwise='subtract',
            tran_lhs=(0,),
            tran_rhs=(0,),
            index_lhs=(slice(None),),
            index_rhs=(slice(None),),
            ax_contraction=(0,)),
    ),
    (
        (10,), (10,), (),
        'sum', 'multiply',
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(0,),
            tran_rhs=(0,),
            index_lhs=(slice(None),),
            index_rhs=(slice(None),),
            ax_contraction=(0,)),
    ),
    (
        (10,), (10,), (),
        'sum', 'divide',
        _einspec(
            op_reduce='sum',
            op_elementwise='divide',
            tran_lhs=(0,),
            tran_rhs=(0,),
            index_lhs=(slice(None),),
            index_rhs=(slice(None),),
            ax_contraction=(0,)),
    ),
    (
        (10,), (10,), (),
        'min', 'add',
        _einspec(
            op_reduce='min',
            op_elementwise='add',
            tran_lhs=(0,),
            tran_rhs=(0,),
            index_lhs=(slice(None),),
            index_rhs=(slice(None),),
            ax_contraction=(0,)),
    ),
    (
        (10,), (10,), (),
        'min', 'subtract',
        _einspec(
            op_reduce='min',
            op_elementwise='subtract',
            tran_lhs=(0,),
            tran_rhs=(0,),
            index_lhs=(slice(None),),
            index_rhs=(slice(None),),
            ax_contraction=(0,)),
    ),
    (
        (10,), (10,), (),
        'min', 'multiply',
        _einspec(
            op_reduce='min',
            op_elementwise='multiply',
            tran_lhs=(0,),
            tran_rhs=(0,),
            index_lhs=(slice(None),),
            index_rhs=(slice(None),),
            ax_contraction=(0,)),
    ),
    (
        (10,), (10,), (),
        'min', 'divide',
        _einspec(
            op_reduce='min',
            op_elementwise='divide',
            tran_lhs=(0,),
            tran_rhs=(0,),
            index_lhs=(slice(None),),
            index_rhs=(slice(None),),
            ax_contraction=(0,)),
    ),
    (
        (10,), (10,), (),
        'max', 'add',
        _einspec(
            op_reduce='max',
            op_elementwise='add',
            tran_lhs=(0,),
            tran_rhs=(0,),
            index_lhs=(slice(None),),
            index_rhs=(slice(None),),
            ax_contraction=(0,)),
    ),
    (
        (10,), (10,), (),
        'max', 'subtract',
        _einspec(
            op_reduce='max',
            op_elementwise='subtract',
            tran_lhs=(0,),
            tran_rhs=(0,),
            index_lhs=(slice(None),),
            index_rhs=(slice(None),),
            ax_contraction=(0,)),
    ),
    (
        (10,), (10,), (),
        'max', 'multiply',
        _einspec(
            op_reduce='max',
            op_elementwise='multiply',
            tran_lhs=(0,),
            tran_rhs=(0,),
            index_lhs=(slice(None),),
            index_rhs=(slice(None),),
            ax_contraction=(0,)),
    ),
    (
        (10,), (10,), (),
        'max', 'divide',
        _einspec(
            op_reduce='max',
            op_elementwise='divide',
            tran_lhs=(0,),
            tran_rhs=(0,),
            index_lhs=(slice(None),),
            index_rhs=(slice(None),),
            ax_contraction=(0,)),
    ),
])
def test_generic_inner(case):
    tol = 20 * ab.finfo(ab.float32).eps

    lhs_shape, rhs_shape, result_shape, reduction, pairwise, ffspec = case
    lhs = ab.normal(0.0, 1.0, lhs_shape)
    rhs = ab.normal(0.0, 1.0, rhs_shape)
    truth = getattr(ab, reduction)(getattr(ab, pairwise)(lhs, rhs))
    res = _einop(lhs, rhs, ffspec, result_shape)
    assert truth.shape == res.shape
    assert ab.allclose(truth, res, tol)


@pytest.mark.parametrize('case', [
    (
        (3,), (3,), (9,),
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(0,),
            tran_rhs=(0,),
            index_lhs=(slice(None), None),
            index_rhs=(None, slice(None)),
            ax_contraction=()),
    ),
    (
        (1,), (5,), (5,),
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(0,),
            tran_rhs=(0,),
            index_lhs=(slice(None), None),
            index_rhs=(None, slice(None)),
            ax_contraction=()),
    ),
    (
        (2, 2), (2, 2), (4, 4),
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(0, 1),
            tran_rhs=(0, 1),
            index_lhs=(slice(None), None, slice(None), None),
            index_rhs=(None, slice(None), None, slice(None)),
            ax_contraction=()),
    ),
    (
        (2, 2), (2, 3), (4, 6),
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(0, 1),
            tran_rhs=(0, 1),
            index_lhs=(slice(None), None, slice(None), None),
            index_rhs=(None, slice(None), None, slice(None)),
            ax_contraction=()),
    ),
    (
        (2, 2), (3, 3), (6, 6),
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(0, 1),
            tran_rhs=(0, 1),
            index_lhs=(slice(None), None, slice(None), None),
            index_rhs=(None, slice(None), None, slice(None)),
            ax_contraction=()),
    ),
    (
        (2, 2), (3, 4), (6, 8),
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(0, 1),
            tran_rhs=(0, 1),
            index_lhs=(slice(None), None, slice(None), None),
            index_rhs=(None, slice(None), None, slice(None)),
            ax_contraction=()),
    ),
    (
        (2, 2), (2, 2), (4, 4),
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(0, 1),
            tran_rhs=(0, 1),
            index_lhs=(slice(None), None, slice(None), None),
            index_rhs=(None, slice(None), None, slice(None)),
            ax_contraction=()),
    ),
    (
        (2, 3), (2, 2), (4, 6),
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(0, 1),
            tran_rhs=(0, 1),
            index_lhs=(slice(None), None, slice(None), None),
            index_rhs=(None, slice(None), None, slice(None)),
            ax_contraction=()),
    ),
    (
        (3, 3), (2, 2), (6, 6),
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(0, 1),
            tran_rhs=(0, 1),
            index_lhs=(slice(None), None, slice(None), None),
            index_rhs=(None, slice(None), None, slice(None)),
            ax_contraction=()),
    ),
    (
        (3, 4), (2, 2), (6, 8),
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(0, 1),
            tran_rhs=(0, 1),
            index_lhs=(slice(None), None, slice(None), None),
            index_rhs=(None, slice(None), None, slice(None)),
            ax_contraction=()),
    ),
    (
        (2, 3, 4), (4, 1, 5), (8, 3, 20),
        _einspec(
            op_reduce='sum',
            op_elementwise='multiply',
            tran_lhs=(0, 1, 2),
            tran_rhs=(0, 1, 2),
            index_lhs=(
                slice(None), None, slice(None), None, slice(None), None
            ),
            index_rhs=(
                None, slice(None), None, slice(None), None, slice(None)
            ),
            ax_contraction=()),
    ),
])
def test_kron(case):
    tol = 100 * ab.finfo(ab.float32).eps

    lhs_shape, rhs_shape, result_shape, ffspec = case
    lhs = ab.normal(0.0, 1.0, lhs_shape)
    rhs = ab.normal(0.0, 1.0, rhs_shape)
    truth = ab.kron(lhs, rhs)
    res = _einop(lhs, rhs, ffspec, result_shape)
    assert truth.shape == res.shape
    assert ab.allclose(truth, res, tol)
