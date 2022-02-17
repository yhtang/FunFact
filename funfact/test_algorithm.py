#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest  # noqa: F401
from funfact import active_backend as ab
from .lang import tensor
from .algorithm import factorize


@pytest.mark.parametrize('test_case', [
    (tensor(3, 3), ab.ones((3, 3)), [ab.ones((3, 3))]),
    (tensor(3, 3), ab.zeros((3, 3)), [ab.zeros((3, 3))]),
    (tensor(3, 4), ab.eye(3, 4), [ab.eye(3, 4)]),
])
def test_simple(test_case):

    tsrex, target, truth = test_case

    fac = factorize(tsrex, target)

    for a, b in zip(fac.factors, truth):
        assert ab.allclose(a, b, atol=1e-3)


def test_dtype():

    tsrex = tensor(2, 2)

    fac = factorize(tsrex, ab.ones((2, 2)), dtype=ab.complex64, max_steps=1)

    for f in fac.factors:
        assert f.dtype == ab.complex64


def test_exception():

    with pytest.raises(RuntimeError):
        factorize(tensor(2), ab.ones(2), optimizer='non-existing')

    with pytest.raises(RuntimeError):
        class Optimizer:
            def __init__(self):
                pass

        factorize(tensor(2), ab.ones(2), optimizer=Optimizer)

    with pytest.raises(RuntimeError):
        factorize(tensor(2), ab.ones(2), loss='non-existing')

    with pytest.raises(RuntimeError):
        class Loss:
            def __call__(self, only_one):
                pass

        factorize(tensor(2), ab.ones(2), loss=Loss)

    with pytest.raises(RuntimeError):
        factorize(tensor(2), ab.ones(2), returns='Everything!')

    with pytest.raises(RuntimeError):
        factorize(tensor(2), ab.ones(2), returns=0)

    with pytest.raises(RuntimeError):
        factorize(tensor(2), ab.ones(2), returns=-1)

    with pytest.raises(RuntimeError):
        factorize(tensor(2), ab.ones(2), stop_by='Never')

    with pytest.raises(RuntimeError):
        factorize(tensor(2), ab.ones(2), stop_by=0)

    with pytest.raises(RuntimeError):
        factorize(tensor(2), ab.ones(2), stop_by=-1)


@pytest.mark.parametrize('stop_by', ['first', 2, None])
@pytest.mark.parametrize('vec_axis', [0, -1])
def test_kwargs(stop_by, vec_axis):

    fac = factorize(
        tensor(2), ab.ones(2), vec_size=4, stop_by=stop_by, vec_axis=vec_axis,
        max_steps=100
    )

    assert fac().shape == (2,)


def test_returns():

    fac = factorize(
        tensor(2), ab.ones(2), vec_size=4, max_steps=100, returns='best'
    )
    assert not isinstance(fac, list)

    fac = factorize(
        tensor(2), ab.ones(2), vec_size=4, max_steps=100, returns=2
    )
    assert isinstance(fac, list)

    fac = factorize(
        tensor(2), ab.ones(2), vec_size=4, max_steps=100, returns='all'
    )
    assert isinstance(fac, list)


def test_penalty_weight():
    fac = factorize(tensor(2), ab.ones(2), penalty_weight=1.0)
    assert ab.allclose(fac(), ab.ones(2), atol=1e-3)

    fac = factorize(tensor(2), ab.ones(2), penalty_weight=0.0)
    assert ab.allclose(fac(), ab.ones(2), atol=1e-3)
