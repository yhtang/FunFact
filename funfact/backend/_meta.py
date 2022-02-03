#!/usr/bin/env python
# -*- coding: utf-8 -*-

_required_properties = [
    'native_t',   # tensor type native to the backend
    'tensor_t',   # tensor type interoperable with the backend
    'float32',    # single-precision floats
    'float64',    # double-precision floats
    'complex64',  # single-precision complex floats
    'complex128'  # double-precision complex floats
]

_required_methods = [
    # creation
    'tensor'  # Convert array-like data to a tensor
    # operators
    'add',
    'subtract',
    'multiply',
    'divide',
    # rng
    'seed',
    'normal'
    # functions
    'abs',
    'conj',
    'exp',
    'log',
    'sin',
    'cos',
    'tan',
    'asin',
    'acos',
    'atan',
    'sinh',
    'cosh',
    'tanh',
    'asinh',
    'acosh',
    'atanh',
    'erf',
    'erfc',
    'relu',
    'celu',
    'sigmoid',
    # compilation
    'jitclass',
    'jit'
]
