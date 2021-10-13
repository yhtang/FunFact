#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._interp_base import TranscribeInterpreter


class ASCIIInterpreter(TranscribeInterpreter):
    '''Creates ASCII representations for tensor expressions.'''

    def scalar(self, leaf):
        return str(leaf)

    def tensor(self, leaf):
        return leaf.symbol

    def index(self, leaf):
        return leaf.symbol

    def index_notation(self, tensor, *indices):
        return '{}[{}]'.format(
            tensor.payload,
            ','.join([i.payload for i in indices])
        )

    def call(self, tsrex, f):
        return f

    def pow(self, base, exponent):
        return 'pow'

    def neg(self, tsrex):
        return '-'

    def div(self, lhs, rhs):
        return '/'

    def mul(self, lhs, rhs):
        return '*'

    def add(self, lhs, rhs):
        return '+'

    def sub(self, lhs, rhs):
        return '-'
