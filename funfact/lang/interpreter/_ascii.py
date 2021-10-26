#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._base import TranscribeInterpreter


class ASCIIRenderer(TranscribeInterpreter):
    '''Creates ASCII representations for tensor expressions.'''

    def scalar(self, value, payload):
        return str(value)

    def tensor(self, value, payload):
        return value.symbol

    def index(self, value, payload):
        return value.symbol

    def index_notation(self, tensor, indices, payload):
        return '{}[{}]'.format(
            tensor.payload,
            ','.join([i.payload for i in indices])
        )

    def call(self, f, x, payload):
        return f

    def pow(self, base, exponent, payload):
        return 'pow'

    def neg(self, x, payload):
        return '-'

    def div(self, lhs, rhs, payload):
        return '/'

    def mul(self, lhs, rhs, payload):
        return '*'

    def add(self, lhs, rhs, payload):
        return '+'

    def sub(self, lhs, rhs, payload):
        return '-'
