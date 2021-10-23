#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._tsrex import TsrEx as _T
from ._ast import Primitives as _P


def abs(x): return _T(_P.call('abs', _T._as_node(x)))
def exp(x): return _T(_P.call('exp', _T._as_node(x)))
def log(x): return _T(_P.call('log', _T._as_node(x)))
def sin(x): return _T(_P.call('sin', _T._as_node(x)))
def cos(x): return _T(_P.call('cos', _T._as_node(x)))
def tan(x): return _T(_P.call('tan', _T._as_node(x)))
def asin(x): return _T(_P.call('asin', _T._as_node(x)))
def acos(x): return _T(_P.call('acos', _T._as_node(x)))
def atan(x): return _T(_P.call('atan', _T._as_node(x)))
def sinh(x): return _T(_P.call('sinh', _T._as_node(x)))
def cosh(x): return _T(_P.call('cosh', _T._as_node(x)))
def tanh(x): return _T(_P.call('tanh', _T._as_node(x)))
def asinh(x): return _T(_P.call('asinh', _T._as_node(x)))
def acosh(x): return _T(_P.call('acosh', _T._as_node(x)))
def atanh(x): return _T(_P.call('atanh', _T._as_node(x)))
def erf(x): return _T(_P.call('erf', _T._as_node(x)))
def erfc(x): return _T(_P.call('erfc', _T._as_node(x)))