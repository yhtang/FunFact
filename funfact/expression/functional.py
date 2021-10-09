#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._expr import Expr


def abs(x): return Expr('call', x, f='abs')
def exp(x): return Expr('call', x, f='exp')
def log(x): return Expr('call', x, f='log')
def sin(x): return Expr('call', x, f='sin')
def cos(x): return Expr('call', x, f='cos')
def tan(x): return Expr('call', x, f='tan')
def asin(x): return Expr('call', x, f='asin')
def acos(x): return Expr('call', x, f='acos')
def atan(x): return Expr('call', x, f='atan')
def sinh(x): return Expr('call', x, f='sinh')
def cosh(x): return Expr('call', x, f='cosh')
def tanh(x): return Expr('call', x, f='tanh')
def asinh(x): return Expr('call', x, f='asinh')
def acosh(x): return Expr('call', x, f='acosh')
def atanh(x): return Expr('call', x, f='atanh')
def erf(x): return Expr('call', x, f='erf')
def erfc(x): return Expr('call', x, f='erfc')
def min(x, y): return Expr('call', x, y, f='min')
def max(x, y): return Expr('call', x, y, f='max')
