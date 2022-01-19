#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest  # noqa: F401
import numpy as np
from .loss import (
    MSE,
    L1,
    KLDivergence,
)


@pytest.mark.parametrize('loss_cls', [
    MSE,
    L1,
])
def test_seminorms(loss_cls):

    loss = loss_cls()
    for shape in [(3,), (3, 3), (3, 4), (3, 4, 5), (2, 3, 4, 5)]:
        for _ in range(64):
            a = np.random.randn(*shape)
            b = np.random.randn(*shape)
            assert loss(a, a) == pytest.approx(0, abs=1e-6)
            assert loss(b, b) == pytest.approx(0, abs=1e-6)
            assert loss(a, b) >= 0
            assert loss(b, a) >= 0


def test_kl_divergence():

    loss = KLDivergence()
    for shape in [(3,), (3, 3), (3, 4), (3, 4, 5), (2, 3, 4, 5)]:
        for _ in range(64):
            a = np.random.rand(*shape) + 1e-6
            b = np.random.rand(*shape) + 1e-6
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
    pred = np.random.randn(n, m)
    ref = np.random.randn(n, m)
    assert loss(pred, ref, sum_vec=False).shape == ()
    with pytest.raises(ValueError):
        loss(pred, np.random.randn(n, m + 1))
    with pytest.raises(ValueError):
        loss(pred, np.random.randn(n, m, 1, 1))
    with pytest.raises(ValueError):
        loss(np.random.randn(n, m, 1, 1), ref)
    with pytest.raises(ValueError):
        loss(np.random.randn(2), np.random.randn(2, 3, 4))
    with pytest.raises(ValueError):
        loss(np.random.randn(2, 3, 4), np.random.randn(2))

    pred = np.random.randn(n, m, nvec)
    ref = np.random.randn(n, m)
    assert loss(
        pred, ref, sum_vec=False, vectorized_along_last=True
    ).shape == (nvec,)
    with pytest.raises(ValueError):
        loss(pred, np.random.randn(n, m + 1))

    loss = loss_cls()
    pred = np.random.randn(n, m, nvec)
    ref = np.random.randn(n, m)
    assert loss(
        pred, ref, sum_vec=True, vectorized_along_last=True
    ).shape == ()

    loss = loss_cls()
    pred = np.random.randn(nvec, n, m)
    ref = np.random.randn(n, m)
    assert loss(
        pred, ref, sum_vec=False, vectorized_along_last=False
    ).shape == (nvec,)

    loss = loss_cls()
    pred = np.random.randn(nvec, n, m)
    ref = np.random.randn(n, m)
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
    a = np.ones((3, 3))
    b = np.ones((3, 3)) * 2
    assert loss_sum(a, b) == loss_mean(a, b) * a.size
