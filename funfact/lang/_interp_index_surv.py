#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numbers import Real
from typing import Iterable, Union, Any
from ._interp_base import TranscribeInterpreter
from ._ast import Primitives as P
from ._tensor import AbstractIndex, AbstractTensor


class IndexSurvivalInterpreter(TranscribeInterpreter):
    '''The index survival interpreter analyzes which of the indices survive in a
    contraction of two tensors.'''
    Tensorial = Union[
        P.index_notation, P.call, P.pow, P.neg, P.mul, P.div, P.add, P.sub
    ]
    Numeric = Union[Tensorial, Real]

    def scalar(self, value: Real, payload: Any):
        return None

    def tensor(self, value: AbstractTensor, payload: Any):
        return None

    def index(self, value: AbstractIndex, payload: Any):
        return value.symbol

    def index_notation(
        self, tensor: P.tensor, indices: Iterable[P.index], payload: Any
    ):
        return [i.payload for i in indices]

    def call(self, f: str, x: Tensorial, payload: Any):
        return None

    def pow(self, base: Numeric, exponent: Numeric, payload: Any):
        return None

    def neg(self, x: Numeric, payload: Any):
        return None

    def mul(self, lhs: Numeric, rhs: Numeric, payload: Any):
        diff_lhs = [x for x in lhs.payload if x not in rhs.payload]
        diff_rhs = [x for x in rhs.payload if x not in lhs.payload]
        return diff_lhs + diff_rhs

    def div(self, lhs: Numeric, rhs: Numeric, payload: Any):
        diff_lhs = [x for x in lhs.payload if x not in rhs.payload]
        diff_rhs = [x for x in rhs.payload if x not in lhs.payload]
        return diff_lhs + diff_rhs

    def add(self, lhs: Numeric, rhs: Numeric, payload: Any):
        diff_lhs = [x for x in lhs.payload if x not in rhs.payload]
        diff_rhs = [x for x in rhs.payload if x not in lhs.payload]
        return diff_lhs + diff_rhs

    def sub(self, lhs: Numeric, rhs: Numeric, payload: Any):
        diff_lhs = [x for x in lhs.payload if x not in rhs.payload]
        diff_rhs = [x for x in rhs.payload if x not in lhs.payload]
        return diff_lhs + diff_rhs
