#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest  # noqa: F401
from funfact import active_backend as ab
from .loss import (
    Loss,
    MSE,
    L1,
    KLDivergence,
    mse,
    l1,
    kl_divergence,
)


@pytest.mark.parametrize('loss_cls', [
    MSE,
    L1,
])
def test_seminorms(loss_cls):

    loss = loss_cls()
    for shape in [(3,), (3, 3), (3, 4), (3, 4, 5), (2, 3, 4, 5)]:
        for _ in range(64):
            a = ab.normal(0.0, 1.0, shape)
            b = ab.normal(0.0, 1.0, shape)
            assert loss(a, a) == pytest.approx(0, abs=1e-6)
            assert loss(b, b) == pytest.approx(0, abs=1e-6)
            assert ab.allclose(loss(a, b), loss(b, a), atol=1e-6)
            assert loss(a, b) >= 0
            assert loss(b, a) >= 0


def test_kl_divergence():

    loss = KLDivergence()
    for shape in [(3,), (3, 3), (3, 4), (3, 4, 5), (2, 3, 4, 5)]:
        for _ in range(64):
            a = ab.uniform(1e-6, 1.0, shape)
            b = ab.uniform(1e-6, 1.0, shape)
            a = a / a.sum()
            b = b / b.sum()
            assert loss(a, a) == pytest.approx(0, abs=1e-6)
            assert loss(b, b) == pytest.approx(0, abs=1e-6)
            assert loss(a, b) >= loss(b, b)
            assert loss(b, a) >= loss(a, a)


@pytest.mark.parametrize('loss_cls', [
    MSE,
    L1,
    KLDivergence
])
def test_vectorization(loss_cls):

    n, m = 13, 25
    nvec = 8

    loss = loss_cls()
    pred = ab.normal(0.0, 1.0, (n, m))
    ref = ab.normal(0.0, 1.0, (n, m))
    assert loss(pred, ref, sum_vec=False).shape == ()
    with pytest.raises(ValueError):
        loss(pred, ab.normal(0.0, 1.0, (n, m + 1)))
    with pytest.raises(ValueError):
        loss(pred, ab.normal(0.0, 1.0, (n, m, 1, 1)))
    with pytest.raises(ValueError):
        loss(ab.normal(0.0, 1.0, (n, m, 1, 1)), ref)
    with pytest.raises(ValueError):
        loss(ab.normal(0.0, 1.0, (2,)), ab.normal(0.0, 1.0, (2, 3, 4)))
    with pytest.raises(ValueError):
        loss(ab.normal(0.0, 1.0, (2, 3, 4)), ab.normal(0.0, 1.0, (2,)))

    pred = ab.normal(0.0, 1.0, (n, m, nvec))
    ref = ab.normal(0.0, 1.0, (n, m))
    assert loss(
        pred, ref, sum_vec=False, vectorized_along_last=True
    ).shape == (nvec,)
    with pytest.raises(ValueError):
        loss(pred, ab.normal(0.0, 1.0, (n, m + 1)))

    loss = loss_cls()
    pred = ab.normal(0.0, 1.0, (n, m, nvec))
    ref = ab.normal(0.0, 1.0, (n, m))
    assert loss(
        pred, ref, sum_vec=True, vectorized_along_last=True
    ).shape == ()

    loss = loss_cls()
    pred = ab.normal(0.0, 1.0, (nvec, n, m))
    ref = ab.normal(0.0, 1.0, (n, m))
    assert loss(
        pred, ref, sum_vec=False, vectorized_along_last=False
    ).shape == (nvec,)

    loss = loss_cls()
    pred = ab.normal(0.0, 1.0, (nvec, n, m))
    ref = ab.normal(0.0, 1.0, (n, m))
    assert loss(
        pred, ref, sum_vec=True, vectorized_along_last=False
    ).shape == ()


@pytest.mark.parametrize('loss_cls', [
    MSE,
    L1,
    KLDivergence
])
def test_reduction(loss_cls):

    loss_sum = loss_cls(reduction='sum')
    loss_mean = loss_cls(reduction='mean')
    with pytest.raises(Exception):
        loss_cls(reduction='abrakadabra')
    a = ab.ones((3, 3))
    b = ab.ones((3, 3)) * 2
    assert loss_sum(a, b) == loss_mean(a, b) * len(a.ravel())


@pytest.mark.parametrize('loss', [
    mse,
    l1,
    kl_divergence,
])
def test_predefined(loss):
    assert isinstance(loss, Loss)
