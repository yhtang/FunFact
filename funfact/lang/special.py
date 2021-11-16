#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._ast import Primitives as P
from ._tensor import _SpecialTensor
from ._tsrex import TensorEx


delta = TensorEx(P.tensor(
    _SpecialTensor('delta', r'\boldsymbol{\delta}'))
)

_0 = TensorEx(P.tensor(
    _SpecialTensor('0', r'\boldsymbol{0}'))
)

_1 = TensorEx(P.tensor(
    _SpecialTensor('1', r'\boldsymbol{1}'))
)
