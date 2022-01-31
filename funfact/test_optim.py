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


@pytest.mark.parametrize('opt_class', [
    Adam,
    RMSprop
])
def test_convergence(opt_class):

    class Model():
        def __init__(self, a, b, target):
            self.a = a
            self.b = b
            self.target = target
            self.X = [self._DataTensor(ab.tensor([0.16])),
                      self._DataTensor(ab.tensor([-2.34]))]

        def __call__(self):
            return (self.a * self.X[0].data**2 + self.b * self.X[1].data -
                    self.target)**2

        def grad(self):
            g = 2*(self.a * self.X[0].data**2 + self.b * self.X[1].data -
                   self.target)
            return [g*2*self.a*self.X[0].data, g*self.b]

        @property
        def factors(self):
            return self._NodeView(
                'data', self.X
            )

        @factors.setter
        def factors(self, tensors):
            for i, n in enumerate(self.X):
                n.data = tensors[i]

        class _DataTensor:
            def __init__(self, tensor):
                self.data = tensor

        class _NodeView:
            def __init__(self, attribute: str, nodes):
                self.attribute = attribute
                self.nodes = nodes

            def __getitem__(self, i):
                return getattr(self.nodes[i], self.attribute)

            def __setitem__(self, i, value):
                setattr(self.nodes[i], self.attribute, value)

            def __iter__(self):
                for n in self.nodes:
                    yield getattr(n, self.attribute)

    tol = 0.001
    model = Model(1.5, -0.25, 3.0)
    opt = opt_class(model.factors)
    for i in range(100):
        opt.step(model.grad())
    assert model() < tol
