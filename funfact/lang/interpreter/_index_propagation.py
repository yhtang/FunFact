#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numbers import Real
from typing import Iterable, Union
from ._base import TranscribeInterpreter
from funfact.lang._ast import Primitives as P
from funfact.lang._tensor import AbstractIndex, AbstractTensor


class IndexPropagator(TranscribeInterpreter):
    '''The index propagator analyzes which of the indices survive in a
    contraction of two tensors and passes them onto the parent node.'''

    Tensorial = Union[
        P.index_notation, P.call, P.pow, P.neg, P.mul, P.div, P.add, P.sub
    ]
    Numeric = Union[Tensorial, Real]

    _key = 'live_indices'

    def scalar(self, value: Real, **kwargs):
        return (self._key, None)

    def tensor(self, value: AbstractTensor, **kwargs):
        return (self._key, None)

    def index(self, value: AbstractIndex, **kwargs):
        return (self._key, value.symbol)

    def index_notation(
        self, tensor: P.tensor, indices: Iterable[P.index], **kwargs
    ):
        return (self._key, [i.live_indices for i in indices])

    def call(self, f: str, x: Tensorial, **kwargs):
        return (self._key, x.live_indices)

    def pow(self, base: Numeric, exponent: Numeric, **kwargs):
        raise NotImplementedError('The current implementation seems incorrect')
        return (self._key, None)

    def neg(self, x: Numeric, **kwargs):
        return (self._key, x.live_indices)

    def mul(self, lhs: Numeric, rhs: Numeric, **kwargs):
        diff_lhs = [x for x in lhs.live_indices if x not in rhs.live_indices]
        diff_rhs = [x for x in rhs.live_indices if x not in lhs.live_indices]
        return (self._key, diff_lhs + diff_rhs)

    def div(self, lhs: Numeric, rhs: Numeric, **kwargs):
        diff_lhs = [x for x in lhs.live_indices if x not in rhs.live_indices]
        diff_rhs = [x for x in rhs.live_indices if x not in lhs.live_indices]
        return (self._key, diff_lhs + diff_rhs)

    def add(self, lhs: Numeric, rhs: Numeric, **kwargs):
        diff_lhs = [x for x in lhs.live_indices if x not in rhs.live_indices]
        diff_rhs = [x for x in rhs.live_indices if x not in lhs.live_indices]
        return (self._key, diff_lhs + diff_rhs)

    def sub(self, lhs: Numeric, rhs: Numeric, **kwargs):
        diff_lhs = [x for x in lhs.live_indices if x not in rhs.live_indices]
        diff_rhs = [x for x in rhs.live_indices if x not in lhs.live_indices]
        return (self._key, diff_lhs + diff_rhs)
