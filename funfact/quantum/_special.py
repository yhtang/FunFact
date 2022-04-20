#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.lang._ast import Primitives as P
from funfact.lang._terminal import AbstractTensor
from funfact.lang._tsrex import TsrEx
from funfact import active_backend as ab
import numpy as np


def proj0(dtype=None):
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
