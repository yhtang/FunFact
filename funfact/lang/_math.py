#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._tsrex import TsrEx as _T, _BaseEx
from ._ast import Primitives as _P


def abs(x): return _T(_P.call('abs', _BaseEx(x).root))
def exp(x): return _T(_P.call('exp', _BaseEx(x).root))
def log(x): return _T(_P.call('log', _BaseEx(x).root))
def sin(x): return _T(_P.call('sin', _BaseEx(x).root))
def cos(x): return _T(_P.call('cos', _BaseEx(x).root))
def tan(x): return _T(_P.call('tan', _BaseEx(x).root))
def asin(x): return _T(_P.call('asin', _BaseEx(x).root))
def acos(x): return _T(_P.call('acos', _BaseEx(x).root))
def atan(x): return _T(_P.call('atan', _BaseEx(x).root))
def sinh(x): return _T(_P.call('sinh', _BaseEx(x).root))
def cosh(x): return _T(_P.call('cosh', _BaseEx(x).root))
def tanh(x): return _T(_P.call('tanh', _BaseEx(x).root))
def asinh(x): return _T(_P.call('asinh', _BaseEx(x).root))
def acosh(x): return _T(_P.call('acosh', _BaseEx(x).root))
def atanh(x): return _T(_P.call('atanh', _BaseEx(x).root))
def erf(x): return _T(_P.call('erf', _BaseEx(x).root))
def erfc(x): return _T(_P.call('erfc', _BaseEx(x).root))
def relu(x): return _T(_P.call('relu', _BaseEx(x).root))
def celu(x): return _T(_P.call('celu', _BaseEx(x).root))
def sigmoid(x): return _T(_P.call('sigmoid', _BaseEx(x).root))
