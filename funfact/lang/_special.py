#!/usr/bin/env python
# -*- coding: utf-8 -*-
import funfact.initializers as ini
from funfact.util.iterable import flatten
from ._ast import Primitives as P
from ._terminal import AbstractTensor
from ._tsrex import TsrEx
from funfact import active_backend as ab
import numpy as np


def zeros(*shape, optimizable=False, dtype=None):
    return TsrEx(
        P.tensor(
            AbstractTensor(
                *flatten(shape), symbol=('0', None),
                initializer=ini.Zeros(dtype=dtype), optimizable=optimizable
            )
        )
    )


def ones(*shape, optimizable=False, dtype=None):
    return TsrEx(
        P.tensor(
            AbstractTensor(
                *flatten(shape), symbol=('1', None),
                initializer=ini.Ones(dtype=dtype), optimizable=optimizable
            )
        )
    )


def eye(n, m=None, optimizable=False, dtype=None):
    return TsrEx(
        P.tensor(
            AbstractTensor(
                n, m or n, symbol=('I', None),
                initializer=ini.Eye(dtype=dtype),
                optimizable=optimizable
            )
        )
    )


def proj0(dtype=None):
    '''Creates abstract tensor that represents the projection on the zero
    state: |0><0|.'''
    e0 = ab.tensor(np.array([[1, 0], [0, 0]]), dtype=dtype)
    return TsrEx(
        P.tensor(
            AbstractTensor(
                2, 2, symbol=('E', 0),
                initializer=e0,
                optimizable=False
            )
        )
    )


def proj1(dtype=None):
    '''Creates abstract tensor that represents the projection on the one
    state: |1><1|.'''
    e1 = ab.tensor(np.array([[0, 0], [0, 1]]), dtype=dtype)
    return TsrEx(
        P.tensor(
            AbstractTensor(
                2, 2, symbol=('E', 1),
                initializer=e1,
                optimizable=False
            )
        )
    )
