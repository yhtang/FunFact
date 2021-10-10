#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._interp_base import FunctionalInterpreter


class LatexInterpreter(FunctionalInterpreter):

    def __call__(self, expr, parent=None):
        '''Decorate the base evaluation result with an optional pair of
        parentheses conditional on the relative precedence between the parent
        and child nodes.'''
        value = super().__call__(expr, parent)
        if parent is not None and expr.p.precedence > parent.p.precedence:
            return fr'\left({value}\right)'
        else:
            return value

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
