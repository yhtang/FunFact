#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._tsrex import TsrEx


def abs(x): return TsrEx('call', x, f='abs')
def exp(x): return TsrEx('call', x, f='exp')
def log(x): return TsrEx('call', x, f='log')
def sin(x): return TsrEx('call', x, f='sin')
def cos(x): return TsrEx('call', x, f='cos')
def tan(x): return TsrEx('call', x, f='tan')
def asin(x): return TsrEx('call', x, f='asin')
def acos(x): return TsrEx('call', x, f='acos')
def atan(x): return TsrEx('call', x, f='atan')
def sinh(x): return TsrEx('call', x, f='sinh')
def cosh(x): return TsrEx('call', x, f='cosh')
def tanh(x): return TsrEx('call', x, f='tanh')
def asinh(x): return TsrEx('call', x, f='asinh')
def acosh(x): return TsrEx('call', x, f='acosh')
def atanh(x): return TsrEx('call', x, f='atanh')
def erf(x): return TsrEx('call', x, f='erf')
def erfc(x): return TsrEx('call', x, f='erfc')
def min(x, y): return TsrEx('call', x, y, f='min')
def max(x, y): return TsrEx('call', x, y, f='max')
