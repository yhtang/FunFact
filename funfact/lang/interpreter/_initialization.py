#!/usr/bin/env python
# -*- coding: utf-8 -*-
import jax.numpy as np
import jax
#from jax import random
#import numpy as np
from ._base import TranscribeInterpreter



class JaxRng:
    
    def __init__(self, key=0):
        self.key = jax.random.PRNGKey(key)
        
    def normal(self, size, dtype=np.float32):
        self.key, subkey = jax.random.split(self.key)
        return jax.random.normal(subkey, size, dtype)


class LeafInitializer(TranscribeInterpreter):
    '''Creates numeric tensors for the leaf nodes in an AST.'''

    def __init__(self,seed=0):
        self.rng = JaxRng(seed)

    as_payload = TranscribeInterpreter.as_payload('data')

    @as_payload
    def scalar(self, value, **kwargs):
        return None

    @as_payload
    def tensor(self, value, **kwargs):
        if value.initializer is not None:
            ini = value.initializer
        else:
            def ini(shape):
                return self.rng.normal(shape)
        return ini(value.shape)

    @as_payload
    def index(self, value, **kwargs):
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
    def div(self, lhs, rhs, **kwargs):
        return None

    @as_payload
    def mul(self, lhs, rhs, **kwargs):
        return None

    @as_payload
    def add(self, lhs, rhs, **kwargs):
        return None

    @as_payload
    def sub(self, lhs, rhs, **kwargs):
        return None
