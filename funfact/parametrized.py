#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import numbers
from typing import Callable
from funfact import active_backend as ab
from funfact.lang._tsrex import TsrEx
from funfact.lang._ast import Primitives as P
from funfact.lang._terminal import ParametrizedAbstractTensor


class Generator:
    ''' A generator can generate a tensor from a set of parameters.

    Args:
        shape_of_params: int...:
            Shape of the parameters to generate tensor.
        generator: Callable:
            Function to generate tensor from parameters.
    '''
    def __init__(self, generator: Callable, *shape_of_params):
        for n in shape_of_params:
            if not (isinstance(n, numbers.Integral) and n > 0):
                raise RuntimeError(
                    "Shape of the parameters must be a positive integer, "
                    f"got {n}."
                )
        self.shape_of_params = tuple(map(int, shape_of_params))
        self._generator = generator

    def __call__(self, params, slices=None):
        return self._generator(params, slices=slices)

    class VectorizedGenerator:
        def __init__(self, generator, n, append: bool = True):
            self.generator = generator
            self.n = n
            self.append = append

        def __call__(self, params, slices=None):
            def _get_instance(i):
                return params[..., i] if self.append else params[i, ...]
            axis = -1 if self.append else 0
            return ab.stack(
                [self.generator(_get_instance(i), slices=slices) for i in
                 range(self.n)], axis
            )

    def vectorize(self, n, append: bool = True):
        '''Vectorize to n replicas.'''
        shape_of_params = (*self.shape_of_params, n) if append else \
                          (n, *self.shape_of_params)
        generator = Generator.VectorizedGenerator(self, n, append)
        return type(self)(generator, *shape_of_params)


def planar_rotation(i, j, n, initializer=None, optimizable=True):
    '''Generate an n x n planar rotation parameterized by a single rotation
    angle with the rotation acting on the [(i,j), (i,j)] submatrix nof the
    n x n identity matrix.

    Args:
        i: int:
            first row/column index for Givens rotation
        j: int:
            second row/column index for Givens rotation
        n: int:
            size of rotation matrix.
        initializer (callable):
            Initialization distribution
        optimizable (boolean):
            True/False flag indicating if a tensor leaf should be optimized.
    '''

    if not (isinstance(n, numbers.Integral) and n > 0):
        raise RuntimeError(
                "Shape of the planar rotation must be a positive integer, "
                f"got {n}."
                )

    def _check_rc(idx):
        if not (isinstance(idx, numbers.Integral) and idx < n and idx >= 0):
            raise RuntimeError(
                f"Row/column index must be between [0, {n}], "
                f"got {idx}."
            )

    _check_rc(i)
    _check_rc(j)

    min_idx = min(i, j)
    max_idx = max(i, j)

    def _gen_rotation(theta, slices=None):
        if slices is not None:
            raise TypeError
        rot22 = ab.vstack(
            [ab.hstack([ab.cos(theta), -ab.sin(theta)]),
             ab.hstack([ab.sin(theta), ab.cos(theta)])]
        )
        rot = ab.vstack(
            [ab.hstack([rot22, ab.zeros([2, n-2])]),
             ab.hstack([ab.zeros([n-2, 2]), ab.eye(n-2, n-2)])]
        )
        p = list(range(-1, n-1))
        p[0] = min_idx + 0.5
        p[1] = max_idx - 0.5
        p = np.argsort(p)
        return rot[p, :][:, p]

    symbol_str = 'G_' + str(min_idx) + str(max_idx) + str(n)
    return TsrEx(
        P.parametrized_tensor(
            ParametrizedAbstractTensor(
                Generator(_gen_rotation, 1), n, n, symbol=symbol_str,
                initializer=initializer, optimizable=optimizable
            )
        )
    )
