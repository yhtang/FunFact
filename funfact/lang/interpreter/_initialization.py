#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
from funfact.backend import active_backend as ab
from ._base import TranscribeInterpreter


class LeafInitializer(TranscribeInterpreter):
    '''Creates numeric tensors for the leaf nodes in an AST.'''

    def __init__(self, dtype=ab.float32):
        super().__init__()
        self.dtype = dtype

    as_payload = TranscribeInterpreter.as_payload('data')

    @as_payload
    def literal(self, value, **kwargs):
        return None

    @as_payload
    def tensor(self, abstract, **kwargs):
        if abstract.initializer is not None:
            if not callable(abstract.initializer):
                init_val = ab.as_tensor(abstract.initializer)
            else:
                init_val = abstract.initializer(abstract.shape)
        else:
            init_val = ab.normal(0.0, 1.0, *abstract.shape)
        return init_val

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
    def ein(self, lhs, rhs, precedence, reduction, pairwise, outidx, **kwargs):
        return None

    @as_payload
    def tran(self, src, indices, **kwargs):
        return None
