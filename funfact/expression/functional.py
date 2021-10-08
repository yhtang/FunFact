#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._expr import Expr


def exp(t):
    return Expr('call', t, f='exp')


def sin(t):
    return Expr('call', t, f='sin')


def cos(t):
    return Expr('call', t, f='cos')


def pow(t, e):
    return Expr('pow', t, e)


def square(t):
    return Expr('pow', t, 2)
