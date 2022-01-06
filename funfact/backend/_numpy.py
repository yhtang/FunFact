#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from ._meta import BackendMeta


class NumPyBackend(metaclass=BackendMeta):

    _nla = np
    _rng = np.random.default_rng()

    native_t = np.ndarray
    tensor_t = (np.ndarray,)

    @classmethod
    def tensor(cls, array, optimizable=False, **kwargs):
        return np.asarray(array, **kwargs)

    @classmethod
    def to_numpy(cls, tensor):
        return np.asarray(tensor)

    @classmethod
    def seed(cls, key):
        cls._rng = np.random.default_rng(seed=key)

    @classmethod
    def normal(cls, mean, std, shape, dtype=np.float32):
        return cls._rng.normal(mean, std, shape)

    @classmethod
    def uniform(cls, low, high, shape, dtype=np.float32):
        return cls._rng.uniform(low, high, shape)

    @staticmethod
    def loss_and_grad(loss_fn, example_model, example_target):
        raise TypeError('NumPy backend does not support autograd and backward '
                        'mode; use JAX or PyTorch backend instead.')

    def autograd_decorator(ob):
        return ob

    class AutoGradMixin():
        pass

    def no_grad():
        pass

    @classmethod
    def set_optimizable(cls, x: native_t, optimizable: bool):
        return x
