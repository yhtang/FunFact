#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''[NumPy](https://numpy.org/)'''
import numpy as np


__name__ = 'NumPyBackend'

nla = np
native_t = np.ndarray
'''The native type for tensor data used by the backend.'''
tensor_t = (np.ndarray,)
'''Types acceptable by the backend API as 'tensors'.'''

_rng = np.random.default_rng()


def tensor(array, optimizable=False, **kwargs):
    return np.asarray(array, **kwargs)


def to_numpy(tensor):
    return np.asarray(tensor)


def seed(key):
    global _rng
    _rng = np.random.default_rng(seed=key)


def normal(mean, std, shape, dtype=np.float32):
    return _rng.normal(mean, std, shape)


def uniform(low, high, shape, dtype=np.float32):
    return _rng.uniform(low, high, shape)


def loss_and_grad(loss_fn, example_model, example_target):
    raise TypeError('NumPy backend does not support autograd and backward '
                    'mode; use JAX or PyTorch backend instead.')


def add_autograd(cls):
    raise NotImplementedError('NumPy backend does not support AutoGrad.')


def no_grad():
    pass


def set_optimizable(x: native_t, optimizable: bool):
    return x
