#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from ._meta import BackendMeta


class NumpyBackend(metaclass=BackendMeta):

    _nla = np

    tensor_t = np.ndarray

    @classmethod
    def as_tensor(cls, array, **kwargs):
        return np.asarray(array, **kwargs)
