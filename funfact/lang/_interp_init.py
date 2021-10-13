#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from ._interp_base import TranscribeInterpreter


class InitializationInterpreter(TranscribeInterpreter):

    def scalar(self, leaf):
        return None

    def tensor(self, leaf):
        if leaf.initializer is not None:
            ini = leaf.initializer
        else:
            def ini(shape):
                return np.random.randn(*shape)
        return ini(leaf.shape)

    def index(self, leaf):
        return None

    def index_notation(self, tensor, *indices):
        return None

    def call(self, tsrex, f):
        return None

    def pow(self, base, exponent):
        return None

    def neg(self, tsrex):
        return None

    def div(self, lhs, rhs):
        return None

    def mul(self, lhs, rhs):
        return None

    def add(self, lhs, rhs):
        return None

    def sub(self, lhs, rhs):
        return None
