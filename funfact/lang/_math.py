#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._tsrex import TsrEx as _T
from ._primitive import primitives as _P


def abs(x): return _T(_P.call, x, f='abs')
def exp(x): return _T(_P.call, x, f='exp')
def log(x): return _T(_P.call, x, f='log')
def sin(x): return _T(_P.call, x, f='sin')
def cos(x): return _T(_P.call, x, f='cos')
def tan(x): return _T(_P.call, x, f='tan')
def asin(x): return _T(_P.call, x, f='asin')
def acos(x): return _T(_P.call, x, f='acos')
def atan(x): return _T(_P.call, x, f='atan')
def sinh(x): return _T(_P.call, x, f='sinh')
def cosh(x): return _T(_P.call, x, f='cosh')
def tanh(x): return _T(_P.call, x, f='tanh')
def asinh(x): return _T(_P.call, x, f='asinh')
def acosh(x): return _T(_P.call, x, f='acosh')
def atanh(x): return _T(_P.call, x, f='atanh')
def erf(x): return _T(_P.call, x, f='erf')
def erfc(x): return _T(_P.call, x, f='erfc')
def min(x, y): return _T(_P.call, x, y, f='min')
def max(x, y): return _T(_P.call, x, y, f='max')
