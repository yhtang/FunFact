#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._interp_base import Interpreter


class LatexInterpreter(Interpreter):

    def __call__(self, expr):
        parts = []
        for subexpr in expr.args:
            part = self(subexpr)
            if expr.p.precedence < subexpr.p.precedence:
                part = fr'\left({part}\right)'
            parts.append(part)
        return getattr(self, expr.p.name)(*parts, **expr.params)

    def scalar(self, value):
        return str(value)

    def tensor(self, value):
        return value._repr_tex_()

    def index(self, value):
        return value._repr_tex_()

    def index_notation(self, tensor, *indices):
        return fr'''{{{tensor}}}_{{{''.join(map(str, indices))}}}'''

    def call(self, input, f):
        return fr'\operatorname{{{f}}}{{{input}}}'

    def pow(self, base, exponent):
        return fr'{{{base}}}^{{{exponent}}}'

    def neg(self, input):
        return fr'-{input}'

    def div(self, lhs, rhs):
        return fr'{lhs} / {rhs}'

    def mul(self, lhs, rhs):
        return fr'{lhs} \times {rhs}'

    def add(self, lhs, rhs):
        return fr'{lhs} + {rhs}'

    def sub(self, lhs, rhs):
        return fr'{lhs} - {rhs}'
