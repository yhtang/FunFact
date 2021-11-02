#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from ._base import ROOFInterpreter
from ._einop import _einop


class Evaluator(ROOFInterpreter):
    '''The evaluation interpreter evaluates an initialized tensor expression.
    '''

    @staticmethod
    def _binary_operator(op, lhs, rhs, spec):
        return _einop(spec, lhs, rhs, op)

    def scalar(self, value, **kwargs):
        return value

    def tensor(self, value, data, **kwargs):
        return data

    def index(self, value, **kwargs):
        return None

    def index_notation(self, tensor, indices, **kwargs):
        return tensor

    def call(self, f, x, **kwargs):
        return getattr(np, f)(x)

    def pow(self, base, exponent, **kwargs):
        return np.power(base, exponent)

    def neg(self, x, **kwargs):
        return -x

    def div(self, lhs, rhs, einspec, **kwargs):
        return self._binary_operator(np.divide, lhs, rhs, einspec)

    def mul(self, lhs, rhs, einspec, **kwargs):
        return self._binary_operator(np.multiply, lhs, rhs, einspec)

    def add(self, lhs, rhs, einspec, **kwargs):
        return self._binary_operator(np.add, lhs, rhs, einspec)

    def sub(self, lhs, rhs, einspec, **kwargs):
        return self._binary_operator(np.subtract, lhs, rhs, einspec)
