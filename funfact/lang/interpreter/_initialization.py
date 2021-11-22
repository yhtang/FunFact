#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import jax.numpy as np
import jax
from ._base import TranscribeInterpreter


class JaxRng:

    def __init__(self, key=0):
        self.key = jax.random.PRNGKey(key)

    def normal(self, size, scale=1, dtype=np.float32):
        self.key, subkey = jax.random.split(self.key)
        return scale * jax.random.normal(subkey, size, dtype)


class LeafInitializer(TranscribeInterpreter):
    '''Creates numeric tensors for the leaf nodes in an AST.'''

    def __init__(self, seed=0):
        self.rng = JaxRng(seed)

    as_payload = TranscribeInterpreter.as_payload('data')

    @as_payload
    def literal(self, value, **kwargs):
        return None

    @as_payload
    def tensor(self, abstract, **kwargs):
        if abstract.initializer is not None:
            if not callable(abstract.initializer):
                init_val = copy.deepcopy(abstract.initializer)
            else:
                init_val = abstract.initializer(abstract.shape)
        else:
            init_val = self.rng.normal(abstract.shape)
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
