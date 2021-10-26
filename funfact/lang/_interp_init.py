#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from ._interp_base import TranscribeInterpreter


class InitializationInterpreter(TranscribeInterpreter):

    def scalar(self, value, payload):
        return None

    def tensor(self, value, payload):
        if value.initializer is not None:
            ini = value.initializer
        else:
            def ini(shape):
                return np.random.randn(*shape)
        return ini(value.shape)

    def index(self, value, payload):
        return None

    def index_notation(self, tensor, indices, payload):
        return None

    def call(self, f, x, payload):
        return None

    def pow(self, base, exponent, payload):
        return None

    def neg(self, x, payload):
        return None

    def div(self, lhs, rhs, payload):
        return None

    def mul(self, lhs, rhs, payload):
        return None

    def add(self, lhs, rhs, payload):
        return None

    def sub(self, lhs, rhs, payload):
        return None
