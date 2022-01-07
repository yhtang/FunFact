#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._tsrex import TsrEx as _T, _as_node
from ._ast import Primitives as _P


def abs(x): return _T(_P.call('abs', _as_node(x)))
def conj(x): return _T(_P.call('conj', _as_node(x)))
def exp(x): return _T(_P.call('exp', _as_node(x)))
def log(x): return _T(_P.call('log', _as_node(x)))
def sin(x): return _T(_P.call('sin', _as_node(x)))
def cos(x): return _T(_P.call('cos', _as_node(x)))
def tan(x): return _T(_P.call('tan', _as_node(x)))
def asin(x): return _T(_P.call('asin', _as_node(x)))
def acos(x): return _T(_P.call('acos', _as_node(x)))
def atan(x): return _T(_P.call('atan', _as_node(x)))
def sinh(x): return _T(_P.call('sinh', _as_node(x)))
def cosh(x): return _T(_P.call('cosh', _as_node(x)))
def tanh(x): return _T(_P.call('tanh', _as_node(x)))
def asinh(x): return _T(_P.call('asinh', _as_node(x)))
def acosh(x): return _T(_P.call('acosh', _as_node(x)))
def atanh(x): return _T(_P.call('atanh', _as_node(x)))
def erf(x): return _T(_P.call('erf', _as_node(x)))
def erfc(x): return _T(_P.call('erfc', _as_node(x)))
def relu(x): return _T(_P.call('relu', _as_node(x)))
def celu(x): return _T(_P.call('celu', _as_node(x)))
def sigmoid(x): return _T(_P.call('sigmoid', _as_node(x)))
