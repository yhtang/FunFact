#!/usr/bin/env python
# -*- coding: utf-8 -*-
import jax.numpy as np
from ._base import ROOFInterpreter
from ._einop import _einop


class Evaluator(ROOFInterpreter):
    '''The evaluation interpreter evaluates an initialized tensor expression.
    '''

    @staticmethod
    def _binary_operator(reduction, pairwise, lhs, rhs, spec):
        return _einop(spec, lhs, rhs, reduction, pairwise)

    def literal(self, value, **kwargs):
        # TODO: need to specialize for each literal type
        # e.g. scalar, 1, 0, delta
        return value

    def tensor(self, abstract, data, **kwargs):
        return data

    def index(self, item, bound, kron, **kwargs):
        return None

    def indices(self, items, **kwargs):
        return None

    def index_notation(self, tensor, indices, **kwargs):
        return tensor

    def call(self, f, x, **kwargs):
        return getattr(np, f)(x)

    def pow(self, base, exponent, **kwargs):
        return np.power(base, exponent)

    def neg(self, x, **kwargs):
        return -x

    def ein(self, lhs, rhs, precedence, reduction, pairwise, outidx, einspec,
            **kwargs):
        return self._binary_operator(reduction, pairwise, lhs, rhs, einspec)

    def tran(self, src, indices, einspec, **kwargs):
        in_spec, out_spec = einspec.split('->')
        return np.transpose(src, [in_spec.index(i) for i in out_spec])
