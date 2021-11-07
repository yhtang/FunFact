#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._base import TranscribeInterpreter


class ASCIIRenderer(TranscribeInterpreter):
    '''Creates ASCII representations for tensor expressions.'''

    as_payload = TranscribeInterpreter.as_payload('ascii')

    @as_payload
    def scalar(self, value, **kwargs):
        return str(value)

    @as_payload
    def tensor(self, value, **kwargs):
        return value.symbol

    @as_payload
    def index(self, value, **kwargs):
        return value.symbol

    @as_payload
    def index_notation(self, tensor, indices, **kwargs):
        return '{}[{}]'.format(
            tensor.ascii, ','.join([i.ascii for i in indices])
        )

    @as_payload
    def call(self, f, x, **kwargs):
        return f

    @as_payload
    def pow(self, base, exponent, **kwargs):
        return 'pow'

    @as_payload
    def neg(self, x, **kwargs):
        return '-'

    @as_payload
    def div(self, lhs, rhs, **kwargs):
        return '/'

    @as_payload
    def mul(self, lhs, rhs, **kwargs):
        return '*'

    @as_payload
    def add(self, lhs, rhs, **kwargs):
        return '+'

    @as_payload
    def sub(self, lhs, rhs, **kwargs):
        return '-'

    @as_payload
    def let(self, dst, src, **kwargs):
        return '=='
