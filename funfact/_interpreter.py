#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import NamedTuple, Any


class Evaluated(NamedTuple):
    expr: Any
    value: Any


class LatexReprInterpreter:

    precedence = dict(
        idx=1,
        neg=4,
        mul=5,
        div=5,
        add=6,
        sub=6
    )

    rules = {}
    rules['idx'] = lambda tensor, *indices:\
        fr'''{{{tensor}}}_{{{''.join(map(str, indices))}}}'''
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
                    if self.precedence[arg.p] > precedence:
                        part = fr'\left({part}\right)'
                except AttributeError:
                    pass

                parts.append(part)

            value = rule(*parts, **expr.params)

        return Evaluated(expr, value)

# TODO: 
# meta-interpreter, e.g. a tracer interpreter that converts any interpreter
# into one that returns an interpreted tree.


# def __str__(self):
#     lstr = str(self.lhs)
#     rstr = str(self.rhs)
#     if self.lhs.oper.precedence > self.oper.precedence:
#         lstr = fr'({lstr})'
#     if self.rhs.oper.precedence > self.oper.precedence:
#         rstr = fr'({rstr})'
#     return f'{lstr} {self.oper.symbol} {rstr}'


# index expr
def __str__(self):
    return '{tensor}[{indices}]'.format(
        tensor=str(self.tensor),
        indices=', '.join(map(str, self.indices))
    )

def __repr__(self):
    return '{tensor}[{indices}]'.format(
        tensor=repr(self.tensor),
        indices=', '.join(map(repr, self.indices))
    )

def repr_tex(self):
    idx = ''.join([i.repr_tex() for i in self.indices])
    return fr'''{{{self.tensor.repr_tex()}}}_{{{idx}}}'''

