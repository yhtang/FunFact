#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Backend on top of [JAX](https://github.com/google/jax).

Arguments:
    enable_x64 (bool): Enable 64-bit floating point types.
'''
import os
import contextlib
import numpy as np
from ._context import context


# must happen before importing JAX
if context.pop('enable_x64', False):
    os.environ['JAX_ENABLE_X64'] = 'True'


import jax.numpy as jnp
import jax.random as jrn
import jax
from jax.tree_util import register_pytree_node_class

__name__ = 'JAXBackend'

nla = jnp
native_t = jnp.ndarray
'''The native type for tensor data used by the backend.'''
tensor_t = (jnp.ndarray, np.ndarray)
'''Types acceptable by the backend API as 'tensors'.'''

_key = jrn.PRNGKey(int.from_bytes(os.urandom(7), 'big'))


def tensor(array, optimizable=False, **kwargs):
    return jnp.asarray(array, **kwargs)


def to_numpy(tensor, **kwargs):
    return np.asarray(tensor, **kwargs)


def seed(key):
    global _key
    _key = jrn.PRNGKey(key)


def normal(mean, std, shape, dtype=jnp.float32):
    global _key
    _key, subkey = jrn.split(_key)
    return mean + std * jrn.normal(subkey, shape, dtype)


def uniform(low, high, shape, dtype=jnp.float32):
    global _key
    _key, subkey = jrn.split(_key)
    return jrn.uniform(subkey, shape, dtype, minval=low, maxval=high)


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


def add_autograd(cls):

    class AddAutoGrad(cls):
        def tree_flatten(self):
            return list(self.factors), (self.tsrex,)

        @classmethod
        def tree_unflatten(cls, metadata, children):
            return cls._from_jax_flatten(*metadata, children)

    return register_pytree_node_class(AddAutoGrad)


def no_grad():
    return contextlib.nullcontext()


def set_optimizable(x: native_t, optimizable: bool):
    return x
