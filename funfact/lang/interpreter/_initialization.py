#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.backend import active_backend as ab
from ._base import TranscribeInterpreter


class LeafInitializer(TranscribeInterpreter):
    '''Creates numeric tensors for the leaf nodes in an AST.'''

    def __init__(self):
        super().__init__()

    as_payload = TranscribeInterpreter.as_payload('data')

    @as_payload
    def literal(self, value, **kwargs):
        return None

    @as_payload
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

    @as_payload
    def index(self, item, bound, kron, **kwargs):
        return None

    @as_payload
    def indices(self, items, **kwargs):
        return None

    @as_payload
    def index_notation(self, tensor, indices, **kwargs):
        return None

    @as_payload
    def call(self, f, x, **kwargs):
        return None

    @as_payload
    def pow(self, base, exponent, **kwargs):
        return None

    @as_payload
    def neg(self, x, **kwargs):
        return None

    @as_payload
    def elem(self, lhs, rhs, oper, **kwargs):
        return None

    @as_payload
    def ein(self, lhs, rhs, precedence, reduction, pairwise, outidx, **kwargs):
        return None

    @as_payload
    def tran(self, src, indices, **kwargs):
        return None
