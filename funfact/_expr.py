#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numbers
from ._interpreter import LatexReprInterpreter


class Expr:

    latex_repr_interpreter = LatexReprInterpreter()

    def __init__(self, p, *args, **params):
        self.p = p
        self.args = args
        self.params = params

    def eval(self, interpreter):
        return interpreter(self)

    def __repr__(self):
        return '{cls}({p}, {args}{params})'.format(
            cls=type(self).__name__,
            p=repr(self.p),
            args=', '.join(map(repr, self.args)),
            params=', '.join([
                f'{repr(k)}={repr(v)}' for k, v in self.params.items()
            ])
        )

    def _repr_html_(self):
        return f'''$${self.eval(self.latex_repr_interpreter).value}$$'''

    def __add__(self, rhs):
        return add(self, rhs)

    def __radd__(self, lhs):
        return add(lhs, self)

    def __sub__(self, rhs):
        return sub(self, rhs)

    def __rsub__(self, lhs):
        return sub(lhs, self)

    def __mul__(self, rhs):
        return mul(self, rhs)

    def __rmul__(self, lhs):
        return mul(lhs, self)

    def __neg__(self):
        return neg(self)


def _as_expr(value):
    if isinstance(value, Expr):
        return value
    elif isinstance(value, numbers.Real):
        return Expr('lit', value=value)
    else:
        raise RuntimeError(
            f'Value {value} of type {type(value)} not allowed in expression.'
        )


def neg(expr):
    return Expr('neg', expr)


def add(lhs, rhs):
    return Expr('add', _as_expr(lhs), _as_expr(rhs))


def sub(lhs, rhs):
    return Expr('sub', _as_expr(lhs), _as_expr(rhs))


def mul(lhs, rhs):
    return Expr('mul', _as_expr(lhs), _as_expr(rhs))


def div(lhs, rhs):
    return Expr('div', _as_expr(lhs), _as_expr(rhs))
