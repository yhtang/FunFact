#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.backend import active_backend as ab
from ._base import ROOFInterpreter
from ._einop import _einop


class Evaluator(ROOFInterpreter):
    '''The evaluation interpreter evaluates an initialized tensor expression.
    '''

    @staticmethod
    def _binary_operator(reduction, pairwise, lhs, rhs, spec):
        return _einop(spec, lhs, rhs, reduction, pairwise)

    def literal(self, value, **kwargs):
        return ab.tensor(value.raw)

    def tensor(self, abstract, data, **kwargs):
        return data

    def index(self, item, bound, kron, **kwargs):
        return None

    def indices(self, items, **kwargs):
        return None

    def index_notation(self, indexless, indices, **kwargs):
        return indexless

    def call(self, f, x, **kwargs):
        return getattr(ab, f)(x)

    def neg(self, x, **kwargs):
        return -x

    def matmul(self, lhs, rhs, **kwargs):
        return lhs @ rhs

    def kron(self, lhs, rhs, **kwargs):
        return ab.kron(lhs, rhs)

    def binary(self, lhs, rhs, precedence, oper, **kwargs):
        return getattr(ab, oper)(lhs, rhs)

    def ein(self, lhs, rhs, precedence, reduction, pairwise, outidx, einspec,
            **kwargs):
        return self._binary_operator(reduction, pairwise, lhs, rhs, einspec)

    def tran(self, src, indices, einspec, **kwargs):
        in_spec, out_spec = einspec.split('->')
        return ab.transpose(src, [in_spec.index(i) for i in out_spec])
