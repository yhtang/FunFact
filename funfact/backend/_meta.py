#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABCMeta


class BackendMeta(ABCMeta):

    _required_properties = [
        'tensor_t',  # the type of a tensor
        'float32',   # single-precision floats
        'float64',   # double-precision floats
    ]

    _required_methods = [
        # type conversion
        'as_tensor'  # Convert array-like data to a tensor
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

    def __getattr__(self, attr):
        '''By default, dispatch all requests to the underlying NLA package.'''
        return getattr(self._nla, attr)

    def is_tensor(self, array):
        '''Determine if the argument is of type tensor_t.'''
        return isinstance(array, self.tensor_t)

    def log_add_exp(self, lhs, rhs):
        return self.log(self.add(self.exp(lhs), self.exp(rhs)))

    def log_sum_exp(self, data, axis=None):
        return self.log(self.sum(self.exp(data), axis=axis))
