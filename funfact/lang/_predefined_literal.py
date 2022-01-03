#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._ast import Primitives as P
from ._terminal import LiteralValue
from ._tsrex import TsrEx


_0 = TsrEx(P.literal(LiteralValue(0, r'\boldsymbol{0}')))

_1 = TsrEx(P.literal(LiteralValue(1, r'\boldsymbol{1}')))

delta = TsrEx(P.literal(LiteralValue('delta', r'\boldsymbol{\delta}')))

pi = TsrEx(P.literal(LiteralValue(3.141592653589793238462643383279, r'\pi')))
