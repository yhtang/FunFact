#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._ast import Primitives as P
from ._terminal import LiteralValue
from ._tsrex import TensorEx, TsrEx


class LiteralEx(TensorEx):
    def __getitem__(self, indices):
        if indices is Ellipsis:
            return TsrEx(P.index_notation(self.root, P.indices(tuple())))
        else:
            return super().__getitem__(indices)


_0 = LiteralEx(P.literal(LiteralValue(0, r'\boldsymbol{0}')))

_1 = LiteralEx(P.literal(LiteralValue(1, r'\boldsymbol{1}')))

delta = LiteralEx(P.literal(LiteralValue('delta', r'\boldsymbol{\delta}')))
