#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Callable
from funfact import active_backend as ab
from funfact.lang._tsrex import TsrEx
from funfact.lang._ast import Primitives as P
from funfact.lang._terminal import ParametrizedAbstractTensor

class Generator:
    ''' A generator can generate a tensor from a set of parameters.

    Args:
        shape_param: Tuple(Int):
            Shape of the parameters to generate tensor.
        generator: Callable:
            Function to generate tensor from parameters.
    '''
    def __init__(self, shape_param, generator: Callable):
        self._shape_param = shape_param
        self._generator = generator

    def __call__(self, param):
        return self._generator(param)

    @property
    def shape_param(self):
        return self._shape_param

    def vectorize(self, n, append: bool = True):
        '''Vectorize to n replicas.'''

        def wrapper(params):
            '''Vectorized generator.'''
            def _get_instance(i):
                return params[..., i] if append else params[i, ...]
            axis = -1 if append else 0
            return ab.stack(
                [self._gen(_get_instance(i)) for i in range(n)], axis
            )

        self._generator = wrapper
        self._shape_param = (*self._shape_param, n) if append else \
                            (n, *self._shape_param)


def _gen_rotation(theta):
    return ab.tensor([[ab.cos(theta), -ab.sin(theta)],
                      [ab.sin(theta), ab.cos(theta)]])


# TODO: make it nxn on i,j?
def givens_rotation():
    return TsrEx(
        P.parametrized_tensor(
            ParametrizedAbstractTensor(
                2, 2, symbol='G', initializer=None, optimizable=True,
                generator=Generator(1, _gen_rotation)
            )
        )
    )
        
