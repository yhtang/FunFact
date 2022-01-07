#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.backend import active_backend as ab
from funfact.initializers import Normal
from ._base import TranscribeInterpreter


class LeafInitializer(TranscribeInterpreter):
    '''Creates numeric tensors for the leaf nodes in an AST.'''

    _traversal_order = TranscribeInterpreter.TraversalOrder.POST

    def __init__(self, dtype):
        self.dtype = dtype or ab.float32
        super().__init__()

    def literal(self, value, **kwargs):
        return []

    @TranscribeInterpreter.as_payload('data')
    def tensor(self, abstract, **kwargs):
        initializer, optimizable, shape = (
            abstract.initializer, abstract.optimizable, abstract.shape
        )
        if initializer is None:
            initializer = Normal(dtype=self.dtype)
        elif isinstance(initializer, type):
            '''got an initializer class'''
            initializer = initializer(dtype=self.dtype)
        try:
            return ab.set_optimizable(initializer(shape), optimizable)
        except TypeError:
            # If optimizable, slice for each instance must be independent.
            # Otherwise, slices can share a view into the original tensor.
            f = ab.tile if optimizable else ab.broadcast_to
            return ab.set_optimizable(
                f(ab.tensor(initializer, dtype=self.dtype), shape),
                optimizable=optimizable
            )

    def ellipsis(self, ellipsis, **kwargs):
        return []

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
