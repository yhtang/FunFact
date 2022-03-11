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

    class VectorizedGenerator:
        def __init__(self, generator, n, append: bool = True):
            self.generator = generator
            self.n = n
            self.append = append

        def __call__(self, params):
            def _get_instance(i):
                return params[..., i] if self.append else params[i, ...]
            axis = -1 if self.append else 0
            return ab.stack(
                [self.generator(_get_instance(i)) for i in range(self.n)], axis
            )

    def vectorize(self, n, append: bool = True):
        '''Vectorize to n replicas.'''
        shape_param = (*self._shape_param, n) if append else \
                      (n, *self._shape_param)
        generator = Generator.VectorizedGenerator(self, n, append)
        return type(self)(shape_param, generator)


def givens_rotation(i, j, n):
    '''Generate an nxn parametrized Givens rotation with the rotation acting
    on the [(i,j), (i,j)] submatrix.

    Args
        i: int:
            first row/column index for Givens rotation
        j: int:
            second row/column index for Givens rotation
        n: int:
            size of rotation matrix.
    '''
    def _gen_rotation(theta):
        rot = ab.eye(n, n)
        # rot = rot.at[i, i].set(ab.cos(theta[0]))
        # rot = rot.at[i, j].set(-ab.sin(theta[0]))
        # rot = rot.at[j, j].set(ab.cos(theta[0]))
        # rot = rot.at[j, i].set(ab.sin(theta[0]))
        rot[i, i] = ab.cos(theta)
        rot[i, j] = -ab.sin(theta)
        rot[j, i] = ab.sin(theta)
        rot[j, j] = ab.cos(theta)
        return rot

    '''
    def _gen_rotation(theta):
        return ab.vstack(
            [ab.hstack([ab.cos(theta), -ab.sin(theta)]),
             ab.hstack([ab.sin(theta), ab.cos(theta)])]
    )
    '''

    return TsrEx(
        P.parametrized_tensor(
            ParametrizedAbstractTensor(
                n, n, symbol='G', initializer=None, optimizable=True,
                generator=Generator((1,), _gen_rotation)
            )
        )
    )
