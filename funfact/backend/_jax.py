#!/usr/bin/env python
# -*- coding: utf-8 -*-
import jax.numpy as jnp
from ._meta import BackendMeta


class JAXBackend(metaclass=BackendMeta):

    _nla = jnp

    tensor_t = jnp.ndarray

    @classmethod
    def as_tensor(cls, array, **kwargs):
        return jnp.asarray(array, **kwargs)
