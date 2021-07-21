#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import typing


class Primitive(typing.NamedTuple):
    name: str


p_idx = Primitive('idx')
p_div = Primitive('exp')
p_neg = Primitive('neg')
p_mul = Primitive('mul')
p_div = Primitive('div')
p_add = Primitive('add')
p_sub = Primitive('sub')


class Expression:

    def __init__(self, *args, **params):
        self.args = args
        self.params = params

    def __repr__(self):
        return '{cls}({args}{params})'.format(
            cls=type(self).__name__,
            args=', '.join(map(repr, self.args)),
            params=', '.join([
                f'{repr(k)}={repr(v)}' for k, v in self.params.items()
            ])
        )

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



def neg(expr): return Expression(p_neg, expr)

def add(lhs, rhs): return Expression(p_add, lhs, rhs)

def sub(lhs, rhs): return Expression(p_sub, lhs, rhs)

def mul(lhs, rhs): return Expression(p_mul, lhs, rhs)

def div(lhs, rhs): return Expression(p_div, lhs, rhs)
