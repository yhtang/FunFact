#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import NamedTuple, Any
from ._primitive import Primitives


class Evaluated(NamedTuple):
    expr: Any
    value: Any


class InterpreterBase:
    def lazy_abstract(f):
        '''A lazy abstract method is similar to an abstract method except
        for that it only produces an error when being called, rather than
        prohibiting the instantiation of objects.'''
        def undefined_handler(*args, **kwargs):
            raise RuntimeError(f'Method {f.__name__} is lazy_abstract.')

        return undefined_handler

    @lazy_abstract
    # @set_precedence(0)
    def _lit(self):
        '''literal value'''

    @lazy_abstract
    def _idn(self):
        '''indexed notation for a single tensor'''

    @lazy_abstract
    def _call(self):
        '''nonlinear function call'''

    @lazy_abstract
    def _pow(self):
        '''raise to power'''

    @lazy_abstract
    def _neg(self):
        '''elementwise negation'''

    @lazy_abstract
    def _einsum(self):
        '''Einstein summation'''

    @lazy_abstract
    def _mul(self):
        '''elementwise multiplication, Hadamard product'''

    @lazy_abstract
    def _div(self):
        '''elementwise division'''

    @lazy_abstract
    def _add(self):
        '''elementwise addition'''

    @lazy_abstract
    def _sub(self):
        '''elementwise subtraction'''


class LatexReprInterpreter(Primitives):

    def __call__(self, expr):
        if hasattr(expr, '_repr_tex_'):
            value = expr._repr_tex_()
        else:
            rule = self._get_rule(expr.p)

            parts = []
            for arg in map(self, expr.args):
                part = arg.value

                try:
                    if _precedence[expr.p] < _precedence[arg.expr.p]:
                        part = fr'\left({part}\right)'
                except AttributeError:
                    pass

                parts.append(part)

            value = rule(*parts, **expr.params)

        return Evaluated(expr, value)

    def _get_rule(self, symbol):
        return getattr(self, f'_{symbol}')

    def _idx(self, tensor, *indices):
        return fr'''{{{tensor}}}_{{{''.join(map(str, indices))}}}'''

    def _lit(self, value):
        return str(value)

    def _call(self, input, f):
        return fr'\operatorname{{{f}}}{{{input}}}'

    def _pow(self, base, exponent):
        return fr'{{{base}}}^{{{exponent}}}'

    def _neg(self, input):
        return fr'-{input}'

    def _div(self, lhs, rhs):
        return fr'{lhs} / {rhs}'

    def _mul(self, lhs, rhs):
        return fr'{lhs} \times {rhs}'

    def _add(self, lhs, rhs):
        return fr'{lhs} + {rhs}'

    def _sub(self, lhs, rhs):
        return fr'{lhs} - {rhs}'




# class TraceInterpreter:
#     '''A trace interpreter is a meta-interpreter that invokes another
#     interpreter to generate an entire evaluated syntax tree.

#     '''

#     rules = {}
#     rules['idx'] = lambda tensor, *indices:\
#         fr'''{{{tensor}}}_{{{''.join(map(str, indices))}}}'''
#     rules['lit'] = lambda value: str(value)
#     rules['call'] = lambda input, f: fr'\operatorname{{{f}}}{{{input}}}'
#     rules['square'] = lambda input: fr'{input}^{2}'
#     rules['neg'] = lambda input: fr'-{input}'
#     rules['mul'] = lambda lhs, rhs: fr'{lhs} \times {rhs}'
#     rules['div'] = lambda lhs, rhs: f'{lhs} / {rhs}'
#     rules['add'] = lambda lhs, rhs: f'{lhs} + {rhs}'
#     rules['sub'] = lambda lhs, rhs: f'{lhs} - {rhs}'

#     def __call__(self, expr, interpret):
#         if hasattr(expr, '_repr_tex_'):
#             value = expr._repr_tex_()
#         else:
#             rule = self.rules[expr.p]
#             precedence = self.precedence[expr.p]

#             parts = []
#             for arg in map(self, expr.args):
#                 part = arg.value

#                 try:
#                     if self.precedence[arg.expr.p] > precedence:
#                         part = fr'\left({part}\right)'
#                 except AttributeError:
#                     pass

#                 parts.append(part)

#             value = rule(*parts, **expr.params)

#         return Evaluated(expr, value)

