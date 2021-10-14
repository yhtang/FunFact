#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._interp_base import FunctionalInterpreter


class LatexInterpreter(FunctionalInterpreter):

    def __call__(self, expr, parent=None):
        '''Decorate the base evaluation result with an optional pair of
        parentheses conditional on the relative precedence between the parent
        and child nodes.'''
        value = super().__call__(expr, parent)
        if parent is not None and \
           expr.primitive.precedence > parent.primitive.precedence:
            return fr'\left({value}\right)'
        else:
            return value

    def scalar(self, leaf):
        return str(leaf)

    def tensor(self, leaf):
        return leaf._repr_tex_()

    def index(self, leaf):
        return leaf._repr_tex_()

    def index_notation(self, tensor, *indices):
        return fr'''{{{tensor}}}_{{{''.join(map(str, indices))}}}'''

    def call(self, tsrex, f):
        return fr'\operatorname{{{f}}}{{{tsrex}}}'

    def pow(self, base, exponent):
        return fr'{{{base}}}^{{{exponent}}}'

    def neg(self, tsrex):
        return fr'-{tsrex}'

    def div(self, lhs, rhs):
        return fr'{lhs} / {rhs}'

    def mul(self, lhs, rhs):
        return fr'{lhs} \times {rhs}'

    def add(self, lhs, rhs):
        return fr'{lhs} + {rhs}'

    def sub(self, lhs, rhs):
        return fr'{lhs} - {rhs}'
