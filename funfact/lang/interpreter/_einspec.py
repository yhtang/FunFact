#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numbers import Real
from typing import Iterable, Union
from ._base import TranscribeInterpreter
from funfact.lang._ast import Primitives as P
from funfact.lang._tensor import AbstractIndex, AbstractTensor


class IndexMap:
    def __init__(self):
        self._index_map = {}

    def _map(self, idx):
        try:
            return self._index_map[idx]
        except KeyError:
            self._index_map[idx] = chr(97 + len(self._index_map))
            return self._index_map[idx]

    def __call__(self, ids):
        try:
            return ''.join([self._map(i) for i in ids])
        except TypeError:
            return self._map(ids)


class EinsteinSpecGenerator(TranscribeInterpreter):
    '''The Einstein summation specification generator creates NumPy-style spec
    strings for tensor contraction operations.'''

    Tensorial = Union[
        P.index_notation, P.call, P.pow, P.neg, P.mul, P.div, P.add, P.sub
    ]
    Numeric = Union[Tensorial, Real]

    as_payload = TranscribeInterpreter.as_payload('einspec')

    @as_payload
    def scalar(self, value: Real, **kwargs):
        return None

    @as_payload
    def tensor(self, value: AbstractTensor, **kwargs):
        return None

    @as_payload
    def index(self, value: AbstractIndex, **kwargs):
        return None

    @as_payload
    def index_notation(
        self, tensor: P.tensor, indices: Iterable[P.index],  **kwargs
    ):
        return None

    @as_payload
    def call(self, f: str, x: Tensorial, **kwargs):
        return None

    @as_payload
    def pow(self, base: Numeric, exponent: Numeric, **kwargs):
        map = IndexMap()
        return f'{map(base.live_indices)},{map(exponent.live_indices)}'

    @as_payload
    def neg(self, x: Numeric, **kwargs):
        return None

    @as_payload
    def mul(self, lhs: Numeric, rhs: Numeric, **kwargs):
        map = IndexMap()
        return f'{map(lhs.live_indices)},{map(rhs.live_indices)}'

    @as_payload
    def div(self, lhs: Numeric, rhs: Numeric, **kwargs):
        map = IndexMap()
        return f'{map(lhs.live_indices)},{map(rhs.live_indices)}'

    @as_payload
    def add(self, lhs: Numeric, rhs: Numeric, **kwargs):
        map = IndexMap()
        return f'{map(lhs.live_indices)},{map(rhs.live_indices)}'

    @as_payload
    def sub(self, lhs: Numeric, rhs: Numeric, **kwargs):
        map = IndexMap()
        return f'{map(lhs.live_indices)},{map(rhs.live_indices)}'
