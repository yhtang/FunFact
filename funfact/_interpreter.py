#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import NamedTuple, Any


class Evaluated(NamedTuple):
    expr: Any
    value: Any


class LatexReprInterpreter:

    precedence = dict(
        lit=0,
        idx=1,
        call=2,
        square=3,
        neg=4,
        mul=5,
        div=5,
        add=6,
        sub=6
    )

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
