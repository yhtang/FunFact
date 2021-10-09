#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numbers
from ._interpreter import LatexReprInterpreter


class TsrEx:

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

    def __pow__(self, exponent):
        return pow(self, exponent)

    def __rpow__(self, base):
        return pow(base, self)


def _as_tsrex(value):
    if isinstance(value, TsrEx):
        return value
    elif isinstance(value, numbers.Real):
        return TsrEx('lit', value=value)
    else:
        raise RuntimeError(
            f'Value {value} of type {type(value)} not allowed in expression.'
        )


def add(lhs, rhs):
    return TsrEx('add', _as_tsrex(lhs), _as_tsrex(rhs))


def sub(lhs, rhs):
    return TsrEx('sub', _as_tsrex(lhs), _as_tsrex(rhs))


def mul(lhs, rhs):
    return TsrEx('mul', _as_tsrex(lhs), _as_tsrex(rhs))


def div(lhs, rhs):
    return TsrEx('div', _as_tsrex(lhs), _as_tsrex(rhs))


def neg(expr):
    return TsrEx('neg', _as_tsrex(expr))


def pow(base, exponent):
    return TsrEx('pow', _as_tsrex(base), _as_tsrex(exponent))
