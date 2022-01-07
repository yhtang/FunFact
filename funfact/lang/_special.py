#!/usr/bin/env python
# -*- coding: utf-8 -*-
import funfact.initializers as ini
from funfact.util.iterable import flatten
from ._ast import Primitives as P
from ._terminal import AbstractTensor
from ._tsrex import TsrEx


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


def eye(n, optimizable=False, dtype=None):
    return TsrEx(
        P.tensor(
            AbstractTensor(
                n, n, symbol=('I', None),
                initializer=ini.Eye(dtype=dtype),
                optimizable=optimizable
            )
        )
    )
