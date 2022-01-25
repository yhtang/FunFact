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

    def abstract_index_notation(self, tensor, indices, **kwargs):
        raise NotImplementedError()

    def abstract_binary(self, lhs, rhs, precedence, operator, **kwargs):
        raise NotImplementedError()

    def literal(self, value, **kwargs):
        return ab.tensor(value.raw)

    def tensor(self, decl, data, **kwargs):
        return data

    def index(self, item, bound, kron, **kwargs):
        return None

    def indices(self, items, **kwargs):
        return None

    def indexed_tensor(self, tensor, indices, **kwargs):
        return tensor

    def call(self, f, x, **kwargs):
        return getattr(ab, f)(x)

    def neg(self, x, **kwargs):
        return -x

    def elem(self, lhs, rhs, precedence, operator, **kwargs):
        return getattr(ab, operator)(lhs, rhs)

    def ein(self, lhs, rhs, precedence, reduction, pairwise, outidx, einspec,
            **kwargs):
        return self._binary_operator(reduction, pairwise, lhs, rhs, einspec)

    def tran(self, src, indices, einspec, **kwargs):
        in_spec, out_spec = einspec.split('->')
        return ab.transpose(src, [in_spec.index(i) for i in out_spec])

    def abstract_dest(self, src, indices, **kwargs):
        raise NotImplementedError()
