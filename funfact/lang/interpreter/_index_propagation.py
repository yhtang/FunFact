#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools as it
from numbers import Real
from typing import Iterable, Union
from ._base import TranscribeInterpreter
from funfact.lang._ast import Primitives as P
from funfact.lang._tensor import AbstractIndex, AbstractTensor

def ordered_symmetric_difference(lhs_indices, rhs_indices):
    diff_lhs = [x for x in lhs_indices if x not in rhs_indices]
    diff_rhs = [x for x in rhs_indices if x not in lhs_indices]
    return diff_lhs + diff_rhs

class IndexPropagator(TranscribeInterpreter):
    '''The index propagator analyzes which of the indices survive in a
    contraction of two tensors and passes them onto the parent node.'''

    Tensorial = Union[
        P.index_notation, P.call, P.pow, P.neg, P.mul, P.div, P.add, P.sub
    ]
    Numeric = Union[Tensorial, Real]

    as_payload = TranscribeInterpreter.as_payload('live_indices')

    @as_payload
    def scalar(self, value: Real, **kwargs):
        return []

    @as_payload
    def tensor(self, value: AbstractTensor, **kwargs):
        return []

    @as_payload
    def index(self, value: AbstractIndex, **kwargs):
        return [value.symbol]

    @as_payload
    def index_notation(
        self, tensor: P.tensor, indices: Iterable[P.index], **kwargs
    ):
        return list(it.chain.from_iterable([i.live_indices for i in indices]))

    @as_payload
    def call(self, f: str, x: Tensorial, **kwargs):
        return x.live_indices

    @as_payload
    def pow(self, base: Numeric, exponent: Numeric, **kwargs):
        return base.live_indices + exponent.live_indices

    @as_payload
    def neg(self, x: Numeric, **kwargs):
        return x.live_indices

    @as_payload
    def mul(self, lhs: Numeric, rhs: Numeric, **kwargs):
        return ordered_symmetric_difference(lhs.live_indices, rhs.live_indices)

    @as_payload
    def div(self, lhs: Numeric, rhs: Numeric, **kwargs):
        return ordered_symmetric_difference(lhs.live_indices, rhs.live_indices)

    @as_payload
    def add(self, lhs: Numeric, rhs: Numeric, **kwargs):
        return ordered_symmetric_difference(lhs.live_indices, rhs.live_indices)

    @as_payload
    def sub(self, lhs: Numeric, rhs: Numeric, **kwargs):
        return ordered_symmetric_difference(lhs.live_indices, rhs.live_indices)