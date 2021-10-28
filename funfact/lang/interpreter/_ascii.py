#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._base import TranscribeInterpreter


class ASCIIRenderer(TranscribeInterpreter):
    '''Creates ASCII representations for tensor expressions.'''

    _key = 'ascii'

    def scalar(self, value, **kwargs):
        return (self._key, str(value))

    def tensor(self, value, **kwargs):
        return (self._key, value.symbol)

    def index(self, value, **kwargs):
        return (self._key, value.symbol)

    def index_notation(self, tensor, indices, **kwargs):
        return (
            self._key,
            '{}[{}]'.format(
                tensor.ascii, ','.join([i.ascii for i in indices])
            )
        )

    def call(self, f, x, **kwargs):
        return (self._key, f)

    def pow(self, base, exponent, **kwargs):
        return (self._key, 'pow')

    def neg(self, x, **kwargs):
        return (self._key, '-')

    def div(self, lhs, rhs, **kwargs):
        return (self._key, '/')

    def mul(self, lhs, rhs, **kwargs):
        return (self._key, '*')

    def add(self, lhs, rhs, **kwargs):
        return (self._key, '+')

    def sub(self, lhs, rhs, **kwargs):
        return (self._key, '-')
