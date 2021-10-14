#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._tsrex import TsrEx as _T
from ._ast import Primitives as _P


def abs(x): return _T(_P.call('abs', x))
def exp(x): return _T(_P.call('exp', x))
def log(x): return _T(_P.call('log', x))
def sin(x): return _T(_P.call('sin', x))
def cos(x): return _T(_P.call('cos', x))
def tan(x): return _T(_P.call('tan', x))
def asin(x): return _T(_P.call('asin', x))
def acos(x): return _T(_P.call('acos', x))
def atan(x): return _T(_P.call('atan', x))
def sinh(x): return _T(_P.call('sinh', x))
def cosh(x): return _T(_P.call('cosh', x))
def tanh(x): return _T(_P.call('tanh', x))
def asinh(x): return _T(_P.call('asinh', x))
def acosh(x): return _T(_P.call('acosh', x))
def atanh(x): return _T(_P.call('atanh', x))
def erf(x): return _T(_P.call('erf', x))
def erfc(x): return _T(_P.call('erfc', x))
