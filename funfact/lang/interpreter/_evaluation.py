#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.backend import active_backend as ab
from ._base import ROOFInterpreter
from ._einop import _einop


class Evaluator(ROOFInterpreter):
    '''The evaluation interpreter evaluates an initialized tensor expression.
    '''

    def abstract_index_notation(self, tensor, indices, **kwargs):
        raise NotImplementedError()

    def abstract_binary(self, lhs, rhs, precedence, operator, **kwargs):
        raise NotImplementedError()

    def literal(self, value, **kwargs):
        return ab.tensor(value.raw)

    def parametrized_tensor(self, decl, data, **kwargs):
        return decl.generator(data)

    def tensor(self, decl, data, **kwargs):
        return data

    def ellipsis(self, **kwargs):
        return None

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
            shape, **kwargs):
        return _einop(lhs, rhs, einspec, shape)

    def tran(self, src, indices, transpec, **kwargs):
        return ab.transpose(
            src, transpec.order
        )

    def abstract_dest(self, src, indices, **kwargs):
        raise NotImplementedError()
