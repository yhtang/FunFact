#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import contextlib
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
    def tensor(cls, array, optimizable=False, **kwargs):
        return jnp.asarray(array, **kwargs)

    @classmethod
    def to_numpy(cls, tensor, **kwargs):
        return np.asarray(tensor, **kwargs)

    @classmethod
    def seed(cls, key):
        cls._key = jrn.PRNGKey(key)

    @classmethod
    def normal(cls, mean, std, shape, dtype=jnp.float32):
        cls._key, subkey = jrn.split(cls._key)
        return mean + std * jrn.normal(subkey, shape, dtype)

    @classmethod
    def uniform(cls, low, high, shape, dtype=jnp.float32):
        cls._key, subkey = jrn.split(cls._key)
        return jrn.uniform(subkey, shape, dtype, minval=low, maxval=high)

    @staticmethod
    def loss_and_grad(loss_fn, example_model, example_target, **kwargs):
        loss_and_grad_fn = jax.jit(
            jax.value_and_grad(
                lambda model, target: loss_fn(model, target, **kwargs)
            )
        )

        def wrapper(model, target):
            loss, dmodel = loss_and_grad_fn(model, target)
            return loss, [jnp.conjugate(df) for df in dmodel.factors]
        return wrapper

    def autograd_decorator(*args, **kwargs):
        return register_pytree_node_class(*args, **kwargs)

    class AutoGradMixin():
        def tree_flatten(self):
            return list(self.factors), (self.tsrex,)

        @classmethod
        def tree_unflatten(cls, metadata, children):
            unflatten = cls(*metadata, initialize=False)
            unflatten.factors = children
            return unflatten

    def no_grad():
        return contextlib.nullcontext()

    @classmethod
    def set_optimizable(self, x: native_t, optimizable: bool):
        return x
