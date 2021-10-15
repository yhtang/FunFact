#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._interp_base import ROOFInterpreter


class LatexInterpreter(ROOFInterpreter):

    def __call__(self, node, parent=None):
        '''Decorate the base evaluation result with an optional pair of
        parentheses conditional on the relative precedence between the parent
        and child nodes.'''
        value = super().__call__(node, parent)
        if parent is not None and node.precedence > parent.precedence:
            return fr'\left({value}\right)'
        else:
            return value

    def scalar(self, value, payload):
        return str(value)

    def tensor(self, value, payload):
        return value._repr_tex_()

    def index(self, value, payload):
        return value._repr_tex_()

    def index_notation(self, tensor, indices, payload):
        return fr'''{{{tensor}}}_{{{''.join(indices)}}}'''

    def call(self, f, x, payload):
        return fr'\operatorname{{{f}}}{{{x}}}'

    def pow(self, base, exponent, payload):
        return fr'{{{base}}}^{{{exponent}}}'

    def neg(self, x, payload):
        return fr'-{x}'

    def div(self, lhs, rhs, payload):
        return fr'{lhs} / {rhs}'

    def mul(self, lhs, rhs, payload):
        return fr'{lhs} \times {rhs}'

    def add(self, lhs, rhs, payload):
        return fr'{lhs} + {rhs}'

    def sub(self, lhs, rhs, payload):
        return fr'{lhs} - {rhs}'
