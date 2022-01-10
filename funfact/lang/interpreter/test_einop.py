#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest  # noqa: F401
from funfact.backend import active_backend as ab
from ._einop import _einop


@pytest.mark.parametrize('case', [
    # scalar multiplication
    ((), (), ',->'),
    # right elementwise multiplication
    ((3,), (), 'a,->a'),
    ((3,), (), 'a,->a'),
    ((3,), (), 'a,'),
    ((3, 2), (), 'ab,->ab'),
    ((3, 2), (), 'ab,'),
    # left elementwise multiplication
    ((), (3,), ',a->a'),
    ((), (3,), ',a->a'),
    ((), (3,), ',a'),
    ((), (3, 2), ',ab->ab'),
    # vector dot product
    ((10,), (10,), 'i,i'),
    ((10,), (10,), 'i,i->i'),
    # matrix elementwise multiplication
    ((3, 2), (3, 2), 'ab,ab->ab'),
    # inner product and contractions
    ((10, 3), (3, 10), 'ij,jk'),
    ((10, 3), (3, 10), 'ij,jk->ijk'),
    ((2, 3, 4), (3, 4, 5), 'ijk,jkl'),
    ((2, 3, 4), (3, 4), 'ijk,jk'),
    ((2, 3, 4), (5, 4, 3), 'ijk,lkj'),
    ((4, 3, 2), (5, 4, 3), 'ijk,lij'),
])
def test_einsum(case):
    tol = 100 * ab.finfo(ab.float32).eps

    lhs_shape, rhs_shape, spec = case
    lhs = ab.normal(0.0, 1.0, lhs_shape)
    rhs = ab.normal(0.0, 1.0, rhs_shape)
    truth = ab.einsum(spec, lhs, rhs)
    res = _einop(spec, lhs, rhs, 'sum', 'multiply')
    assert truth.shape == res.shape
    assert ab.allclose(truth, res, tol)


@pytest.mark.parametrize('case', [
    ((10,), (10,), 'i,j', 'sum', 'add'),
    ((10,), (10,), 'i,j', 'sum', 'subtract'),
    ((10,), (10,), 'i,j', 'sum', 'multiply'),
    ((10,), (10,), 'i,j', 'sum', 'divide'),
])
def test_generic_outer(case):
    tol = 20 * ab.finfo(ab.float32).eps

    lhs_shape, rhs_shape, spec, reduction, pairwise = case
    lhs = ab.normal(0.0, 1.0, lhs_shape)
    rhs = ab.normal(0.0, 1.0, rhs_shape)
    truth = getattr(ab, pairwise)(lhs[:, None], rhs[None, :])
    res = _einop(spec, lhs, rhs, reduction, pairwise)
    assert truth.shape == res.shape
    assert ab.allclose(truth, res, tol)


@pytest.mark.parametrize('case', [
    ((10,), (10,), 'i,i', 'sum', 'add'),
    ((10,), (10,), 'i,i', 'sum', 'subtract'),
    ((10,), (10,), 'i,i', 'sum', 'multiply'),
    ((10,), (10,), 'i,i', 'sum', 'divide'),
    ((10,), (10,), 'i,i', 'min', 'add'),
    ((10,), (10,), 'i,i', 'min', 'subtract'),
    ((10,), (10,), 'i,i', 'min', 'multiply'),
    ((10,), (10,), 'i,i', 'min', 'divide'),
    ((10,), (10,), 'i,i', 'max', 'add'),
    ((10,), (10,), 'i,i', 'max', 'subtract'),
    ((10,), (10,), 'i,i', 'max', 'multiply'),
    ((10,), (10,), 'i,i', 'max', 'divide'),
])
def test_generic_inner(case):
    tol = 20 * ab.finfo(ab.float32).eps

    lhs_shape, rhs_shape, spec, reduction, pairwise = case
    lhs = ab.normal(0.0, 1.0, lhs_shape)
    rhs = ab.normal(0.0, 1.0, rhs_shape)
    truth = getattr(ab, reduction)(getattr(ab, pairwise)(lhs, rhs))
    res = _einop(spec, lhs, rhs, reduction, pairwise)
    assert truth.shape == res.shape
    assert ab.allclose(truth, res, tol)


@pytest.mark.parametrize('case', [
    ((3,), (3,), 'i,i->i|i'),
    ((1,), (5,), 'i,i->i|i'),
    ((2, 2), (2, 2), 'ij,ij->ij|ij'),
    ((2, 2), (2, 3), 'ij,ij->ij|ij'),
    ((2, 2), (3, 3), 'ij,ij->ij|ij'),
    ((2, 2), (3, 4), 'ij,ij->ij|ij'),
    ((2, 2), (2, 2), 'ij,ij->ij|ij'),
    ((2, 3), (2, 2), 'ij,ij->ij|ij'),
    ((3, 3), (2, 2), 'ij,ij->ij|ij'),
    ((3, 4), (2, 2), 'ij,ij->ij|ij'),
    ((2, 3, 4), (4, 1, 5), 'ijk,ijk->ijk|ijk'),
])
def test_kron(case):
    tol = 100 * ab.finfo(ab.float32).eps

    lhs_shape, rhs_shape, spec = case
    lhs = ab.normal(0.0, 1.0, lhs_shape)
    rhs = ab.normal(0.0, 1.0, rhs_shape)
    truth = ab.kron(lhs, rhs)
    res = _einop(spec, lhs, rhs, 'sum', 'multiply')
    assert truth.shape == res.shape
    assert ab.allclose(truth, res, tol)
