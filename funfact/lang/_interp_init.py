#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from ._interp_base import TranscribeInterpreter


class InitializationInterpreter(TranscribeInterpreter):

    def no_op(f):
        '''Returns without evaluating a node.'''
        def do_nothing(*operands):
            return None
        return do_nothing

    @no_op
    def scalar(self, leaf):
        pass

    def tensor(self, leaf):
        if leaf.initializer is not None:
            ini = leaf.initializer
        else:
            def ini(shape):
                return np.random.randn(*shape)
        return ini(leaf.shape)

    @no_op
    def index(self, leaf):
        pass

    @no_op
    def index_notation(self, tensor, *indices):
        pass

    @no_op
    def call(self, tsrex, f):
        pass

    @no_op
    def pow(self, base, exponent):
        pass

    @no_op
    def neg(self, tsrex):
        pass

    @no_op
    def div(self, lhs, rhs):
        pass

    @no_op
    def mul(self, lhs, rhs):
        pass

    @no_op
    def add(self, lhs, rhs):
        pass

    @no_op
    def sub(self, lhs, rhs):
        pass
