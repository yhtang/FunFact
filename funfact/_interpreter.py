#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import NamedTuple, Any
from ._primitive import precedence


class Evaluated(NamedTuple):
    expr: Any
    value: Any


class LatexReprInterpreter:

    rules = {}
    rules['idx'] = lambda tensor, *indices:\
        fr'''{{{tensor}}}_{{{''.join(map(str, indices))}}}'''
    rules['lit'] = lambda value: str(value)
    rules['call'] = lambda input, f: fr'\operatorname{{{f}}}{{{input}}}'
    rules['square'] = lambda input: fr'{input}^{2}'
    rules['neg'] = lambda input: fr'-{input}'
    rules['mul'] = lambda lhs, rhs: fr'{lhs} \times {rhs}'
    rules['div'] = lambda lhs, rhs: f'{lhs} / {rhs}'
    rules['add'] = lambda lhs, rhs: f'{lhs} + {rhs}'
    rules['sub'] = lambda lhs, rhs: f'{lhs} - {rhs}'

    def __call__(self, expr):
        if hasattr(expr, '_repr_tex_'):
            value = expr._repr_tex_()
        else:
            rule = self.rules[expr.p]
            pr = precedence[expr.p]

            parts = []
            for arg in map(self, expr.args):
                part = arg.value

                try:
                    if precedence[arg.expr.p] > pr:
                        part = fr'\left({part}\right)'
                except AttributeError:
                    pass

                parts.append(part)

            value = rule(*parts, **expr.params)

        return Evaluated(expr, value)


class TraceInterpreter:
    '''A trace interpreter is a meta-interpreter that invokes another
    interpreter to generate an entire evaluated syntax tree.

    '''

    rules = {}
    rules['idx'] = lambda tensor, *indices:\
        fr'''{{{tensor}}}_{{{''.join(map(str, indices))}}}'''
    rules['lit'] = lambda value: str(value)
    rules['call'] = lambda input, f: fr'\operatorname{{{f}}}{{{input}}}'
    rules['square'] = lambda input: fr'{input}^{2}'
    rules['neg'] = lambda input: fr'-{input}'
    rules['mul'] = lambda lhs, rhs: fr'{lhs} \times {rhs}'
    rules['div'] = lambda lhs, rhs: f'{lhs} / {rhs}'
    rules['add'] = lambda lhs, rhs: f'{lhs} + {rhs}'
    rules['sub'] = lambda lhs, rhs: f'{lhs} - {rhs}'

    def __call__(self, expr, interpret):
        if hasattr(expr, '_repr_tex_'):
            value = expr._repr_tex_()
        else:
            rule = self.rules[expr.p]
            precedence = self.precedence[expr.p]

            parts = []
            for arg in map(self, expr.args):
                part = arg.value

                try:
                    if self.precedence[arg.expr.p] > precedence:
                        part = fr'\left({part}\right)'
                except AttributeError:
                    pass

                parts.append(part)

            value = rule(*parts, **expr.params)

        return Evaluated(expr, value)

# TODO: 
# meta-interpreter, e.g. a tracer interpreter that converts any interpreter
# into one that returns an interpreted tree.
