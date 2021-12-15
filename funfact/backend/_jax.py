#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import jax.numpy as jnp
import jax.random as jrn
from ._meta import BackendMeta


class JAXBackend(metaclass=BackendMeta):

    _nla = jnp
    _key = jrn.PRNGKey(int.from_bytes(os.urandom(7), 'big'))

    tensor_t = jnp.ndarray

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
