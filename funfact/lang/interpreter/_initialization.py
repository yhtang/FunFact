#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from ._base import TranscribeInterpreter


class LeafInitializer(TranscribeInterpreter):
    '''Creates numeric tensors for the leaf nodes in an AST.'''

    _key = 'data'

    def scalar(self, value, **kwargs):
        return (self._key, None)

    def tensor(self, value, **kwargs):
        if value.initializer is not None:
            ini = value.initializer
        else:
            def ini(shape):
                return np.random.randn(*shape)
        return (self._key, ini(value.shape))

    def index(self, value, **kwargs):
        return (self._key, None)

    def index_notation(self, tensor, indices, **kwargs):
        return (self._key, None)

    def call(self, f, x, **kwargs):
        return (self._key, None)

    def pow(self, base, exponent, **kwargs):
        return (self._key, None)

    def neg(self, x, **kwargs):
        return (self._key, None)

    def div(self, lhs, rhs, **kwargs):
        return (self._key, None)

    def mul(self, lhs, rhs, **kwargs):
        return (self._key, None)

    def add(self, lhs, rhs, **kwargs):
        return (self._key, None)

    def sub(self, lhs, rhs, **kwargs):
        return (self._key, None)
