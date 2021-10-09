#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._tsrex import TsrEx
from ._primitive import primitives as P


def abs(x): return TsrEx(P.call, x, f='abs')
def exp(x): return TsrEx(P.call, x, f='exp')
def log(x): return TsrEx(P.call, x, f='log')
def sin(x): return TsrEx(P.call, x, f='sin')
def cos(x): return TsrEx(P.call, x, f='cos')
def tan(x): return TsrEx(P.call, x, f='tan')
def asin(x): return TsrEx(P.call, x, f='asin')
def acos(x): return TsrEx(P.call, x, f='acos')
def atan(x): return TsrEx(P.call, x, f='atan')
def sinh(x): return TsrEx(P.call, x, f='sinh')
def cosh(x): return TsrEx(P.call, x, f='cosh')
def tanh(x): return TsrEx(P.call, x, f='tanh')
def asinh(x): return TsrEx(P.call, x, f='asinh')
def acosh(x): return TsrEx(P.call, x, f='acosh')
def atanh(x): return TsrEx(P.call, x, f='atanh')
def erf(x): return TsrEx(P.call, x, f='erf')
def erfc(x): return TsrEx(P.call, x, f='erfc')
def min(x, y): return TsrEx(P.call, x, y, f='min')
def max(x, y): return TsrEx(P.call, x, y, f='max')
