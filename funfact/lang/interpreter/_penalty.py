#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._base import TranscribeInterpreter
from funfact.backend import active_backend as ab


class PenaltyEvaluator(TranscribeInterpreter):
    '''Evaluates the penalty terms of the leaf nodes in an AST.'''

    _traversal_order = TranscribeInterpreter.TraversalOrder.POST

    def literal(self, value, **kwargs):
        return []

    @TranscribeInterpreter.as_payload('penalty')
    def tensor(self, abstract, data, **kwargs):
        if abstract._penalty:
            return abstract._penalty(data)
        else:
            return ab.tensor(0.0)

    def index(self, item, bound, kron, **kwargs):
        return []

    def indices(self, items, **kwargs):
        return []

    def index_notation(self, indexless, indices, **kwargs):
        return []

    def call(self, f, x, **kwargs):
        return []

    def neg(self, x, **kwargs):
        return []

    def matmul(self, lhs, rhs, **kwargs):
        return []

    def kron(self, lhs, rhs, **kwargs):
        return []

    def binary(self, lhs, rhs, precedence, oper, **kwargs):
        return []

    def ein(self, lhs, rhs, precedence, reduction, pairwise, outidx, **kwargs):
        return []

    def tran(self, src, indices, **kwargs):
        return []
