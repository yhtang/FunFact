#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._ast import Primitives as P
from ._terminal import LiteralValue
from ._tsrex import TensorEx


_0 = TensorEx(P.literal(LiteralValue(0, r'\boldsymbol{0}')))

_1 = TensorEx(P.literal(LiteralValue(1, r'\boldsymbol{1}')))

delta = TensorEx(P.literal(LiteralValue('delta', r'\boldsymbol{\delta}')))
