#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from ._meta import BackendMeta


class NumpyBackend(metaclass=BackendMeta):

    _nla = np
    _rng = np.random.default_rng()

    tensor_t = np.ndarray

    @classmethod
    def as_tensor(cls, array, **kwargs):
        return np.asarray(array, **kwargs)

    @classmethod
    def seed(cls, key):
        cls._rng = np.random.default_rng(seed=key)

    @classmethod
    def normal(cls, mean, std, *shape, dtype=np.float32):
        return cls._rng.normal(mean, std, shape)
