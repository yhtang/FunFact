#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.backend import active_backend as ab
from funfact.initializers import Normal
from ._base import _as_payload, TranscribeInterpreter


class LeafInitializer(TranscribeInterpreter):
    '''Creates numeric tensors for the leaf nodes in an AST.'''

    _traversal_order = TranscribeInterpreter.TraversalOrder.POST

    def __init__(self, dtype=None):
        self.dtype = dtype or ab.float32
        super().__init__()

    def abstract_index_notation(self, tensor, indices, **kwargs):
        return []

    def abstract_binary(self, lhs, rhs, precedence, operator, **kwargs):
        return []

    def literal(self, value, **kwargs):
        return []

    @_as_payload('data')
    def tensor(self, decl, **kwargs):
        initializer, optimizable, shape = (
            decl.initializer, decl.optimizable, decl.shape
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

    def index(self, item, bound, kron, **kwargs):
        return []

    def indices(self, items, **kwargs):
        return []

    def indexed_tensor(self, tensor, indices, **kwargs):
        return []

    def call(self, f, x, **kwargs):
        return []

    def neg(self, x, **kwargs):
        return []

    def elem(self, lhs, rhs, precedence, operator, **kwargs):
        return []

    def ein(self, lhs, rhs, precedence, reduction, pairwise, outidx, **kwargs):
        return []

    def tran(self, src, indices, **kwargs):
        return []

    def abstract_dest(self, src, indices, **kwargs):
        return []
