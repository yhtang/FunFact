#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import jax.numpy as jnp
import jax.random as jrn
import jax
from jax.tree_util import register_pytree_node_class
from ._meta import BackendMeta


class JAXBackend(metaclass=BackendMeta):

    _nla = jnp
    _key = jrn.PRNGKey(int.from_bytes(os.urandom(7), 'big'))

    native_t = jnp.ndarray
    tensor_t = (jnp.ndarray, np.ndarray)

    @classmethod
    def as_tensor(cls, array, **kwargs):
        return jnp.asarray(array, **kwargs)

    @classmethod
    def seed(cls, key):
        cls._key = jrn.PRNGKey(key)

    @classmethod
    def normal(cls, mean, std, *shape, dtype=jnp.float32):
        cls._key, subkey = jrn.split(cls._key)
        return mean + std * jrn.normal(subkey, shape, dtype)

    @staticmethod
    def grad(*args, **kwargs):
        return jax.grad(*args, **kwargs)

    @staticmethod
    def jit(*args, **kwargs):
        return jax.jit(*args, **kwargs)

    def autograd_decorator(*args, **kwargs):
        return register_pytree_node_class(*args, **kwargs)

    class AutoGradMixin():
        def tree_flatten(self):
            return list(self.factors), self.tsrex

        @classmethod
        def tree_unflatten(cls, tsrex, tensors):
            unflatten = cls(tsrex, initialize=False)
            unflatten.factors = tensors
            return unflatten
