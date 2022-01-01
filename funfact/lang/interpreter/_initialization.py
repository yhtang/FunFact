#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.backend import active_backend as ab
from ._base import TranscribeInterpreter


class LeafInitializer(TranscribeInterpreter):
    '''Creates numeric tensors for the leaf nodes in an AST.'''

    _traversal_order = TranscribeInterpreter.TraversalOrder.POST

    def __init__(self):
        super().__init__()

    def literal(self, value, **kwargs):
        return []

    @TranscribeInterpreter.as_payload('data')
    def tensor(self, abstract, **kwargs):
        initializer, optimizable, shape = (
            abstract.initializer, abstract.optimizable, abstract.shape
        )
        if initializer is not None:
            if not callable(initializer):
                # If optimizable, slice for each instance must be independent.
                # Otherwise, slices can share a view into the original tensor.
                f = ab.tile if optimizable else ab.broadcast_to
                return ab.tensor(
                    f(initializer, shape), optimizable=optimizable
                )
            else:
                return initializer(shape)
        else:
            return ab.normal(0.0, 1.0, *shape, optimizable=optimizable)

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

    def binary(self, lhs, rhs, precedence, oper, **kwargs):
        return []

    def ein(self, lhs, rhs, precedence, reduction, pairwise, outidx, **kwargs):
        return []

    def tran(self, src, indices, **kwargs):
        return []
