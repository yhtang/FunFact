#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest  # noqa: F401
import numpy as np
from funfact import active_backend as ab
from .optim import (
    Adam,
    RMSprop
)


@pytest.mark.parametrize('opt_class', [
    Adam,
    RMSprop
])
def test_generic_init_and_step(opt_class):
    X = [ab.normal(0.0, 1.0, (2, 2)), ab.normal(0.0, 1.0, (3, 3))]
    lr = 10
    opt = opt_class(X, lr=lr)
    assert np.allclose(opt.X[0], X[0])
    assert np.allclose(opt.X[1], X[1])
    assert opt.lr == lr
    assert len(opt.V) == 2
    assert opt.V[0].shape == X[0].shape
    assert opt.V[1].shape == X[1].shape
    grad = [ab.normal(0.0, 1.0, (2, 2)), ab.normal(0.0, 1.0, (3, 3))]
    opt.step(grad)
    assert opt.X[0].shape == X[0].shape
    assert opt.X[1].shape == X[1].shape
    grad = [ab.normal(0.0, 1.0, (3, 2))]
    with pytest.raises(Exception):
        opt.step(grad)


def test_rmsprop_init_and_step():
    X = [ab.normal(0.0, 1.0, (2, 4)), ab.normal(0.0, 1.0, (3, 2))]
    lr = 10
    opt = RMSprop(X, lr,
                  alpha=0.1,
                  epsilon=1e-4,
                  weight_decay=0.2,
                  momentum=0.3,
                  centered=True
                  )
    assert np.allclose(opt.X[0], X[0])
    assert np.allclose(opt.X[1], X[1])
    assert opt.lr == lr
    assert opt.alpha == 0.1
    assert opt.epsilon == 1e-4
    assert opt.weight_decay == 0.2
    assert opt.momentum == 0.3
    assert opt.centered
    for val in [opt.V, opt.B, opt.G]:
        assert len(val) == 2
        assert val[0].shape == X[0].shape
        assert val[1].shape == X[1].shape
        assert np.allclose(val[0], 0.0)
        assert np.allclose(val[1], 0.0)
    grad = [ab.normal(0.0, 1.0, (2, 4)), ab.normal(0.0, 1.0, (3, 2))]
    opt.step(grad)
    assert opt.X[0].shape == X[0].shape
    assert opt.X[1].shape == X[1].shape
