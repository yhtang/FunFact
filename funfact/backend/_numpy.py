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
    def tensor(cls, array, optimizable, **kwargs):
        return np.asarray(array, **kwargs)

    @classmethod
    def seed(cls, key):
        cls._rng = np.random.default_rng(seed=key)

    @classmethod
    def normal(cls, mean, std, *shape, optimizable, dtype=np.float32):
        return cls._rng.normal(mean, std, shape)
